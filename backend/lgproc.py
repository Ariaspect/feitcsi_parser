"""The vendored MT7921 parser's processing, over this project's reader.

The second parser in ``backend/vendor`` is worth having for its *arithmetic*,
not its I/O: it reads a whole capture with ``open().read()`` and walks the TLV
stream record by record in Python. Measured on captures/20260827_143002.bin,
135 MB: their ``load()`` takes 29.8 s and peaks at 709 MB, where reading
through ``MTKIndex`` and sampling 4096 frames takes 5.3 s and 182 MB -- 5.6x
faster on a quarter of the memory. So the file is read here the fast way and
handed to their functions unaltered.

Everything numerical below is theirs, called rather than reimplemented --
``occupancy``, ``active_bins``, ``agc_scale``, ``feature_conj``,
``feature_ratio`` and ``lag1_phase_coherence`` are pure NumPy over ``H`` and
touch no files. Reimplementing them would defeat the point: a second opinion
that has been retyped is no longer independent.

The one thing that must be got right is the axis convention, because the two
parsers disagree about it. Their ``H`` is ``(packet, rx, tx, subcarrier)``
where ``rx`` is tag 16 (our ``rpi``) and ``tx`` is tag 15 (our ``tpi``). Ours
maps ``tpi`` onto the rx axis and reads one ``rpi`` plane. So their ``H`` is
built here from two indexes, ``plane=0`` and ``plane=1``, each contributing
one ``rx`` row and both of its ``tpi`` slots -- which is also the only way to
fill the grid their ``feature_conj`` needs, since that conjugates ACROSS rx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .mtk import MTKIndex, decode_complex
from .vendor import csi_parse as cp

# Their tensor is dense: every packet carries every rx and tx cell it declares.
# Sampling frames evenly keeps a long capture's summary honest without decoding
# all of it -- occupancy and coherence are both statistics over frames, so a
# spread sample answers them, and a contiguous head would answer a different
# question about the first few seconds.
MAX_FRAMES = 4096


def build_tensor(
    path: str | Path,
    *,
    max_frames: int = MAX_FRAMES,
    interpolate: bool = False,
) -> dict[str, Any]:
    """Assemble their ``H`` and the arrays their functions expect.

    Returns ``H`` shaped ``(packets, rx, tx, subcarriers)`` to their
    convention, plus the RSSI and tx-validity masks ``agc_scale`` and
    ``active_bins`` take, and the frame ids actually decoded.

    ``interpolate`` defaults to FALSE here, unlike everywhere else in this
    project. Their ``active_bins`` finds usable subcarriers by measuring how
    often each bin is non-zero, so filling pilots and DC first would hand it a
    capture in which those bins are never null and it would report them as
    carrying data: 245 bins instead of the 234 it finds on the raw stream.
    Their processing assumes the hardware's own zeros are still there, so this
    leaves them.
    """
    path = Path(path)
    planes = [MTKIndex(path, plane=p) for p in (0, 1)]
    base = planes[0]

    # Two-stream frames only: a single-tx frame has no second cell to conjugate
    # against, and their build_tensor drops the same frames via require_full.
    usable = np.flatnonzero(base.num_rx_arr >= 2)
    if usable.size == 0:
        raise ValueError("no frame in this capture carries two streams")
    if usable.size > max_frames:
        usable = usable[np.linspace(0, usable.size - 1, max_frames).astype(np.int64)]

    per_plane = [decode_complex(path, idx, usable, interpolate=interpolate) for idx in planes]
    # (packets, rx=rpi, tx=tpi, subcarriers)
    H = np.stack(per_plane, axis=1).astype(np.complex64)

    # Their tx_valid marks which tx slots a packet actually carried. Ours are
    # NaN where a slot was absent, so finiteness is the same statement.
    tx_valid = np.isfinite(H).any(axis=(1, 3))
    # A cell that was never measured must be zero, not NaN: occupancy() counts
    # magnitudes above zero and NaN would poison the comparison.
    H = np.nan_to_num(H, nan=0.0)

    rssi = np.repeat(base.rssi_1[usable][:, None], H.shape[1], axis=1).astype(np.int16)
    return {
        "H": H,
        "rssi": rssi,
        "tx_valid": tx_valid,
        "frame_ids": usable,
        "times": base.times[usable],
        "nsub": int(base.num_subcarriers),
        "index": base,
    }


def summarise(path: str | Path, *, max_frames: int = MAX_FRAMES) -> dict[str, Any]:
    """Run their processing over a capture and reduce it to what a tab shows."""
    t = build_tensor(path, max_frames=max_frames)
    H, rssi, tx_valid = t["H"], t["rssi"], t["tx_valid"]

    occ = cp.occupancy(H, tx_valid)
    active = cp.active_bins(H, tx_valid)

    agc = cp.apply_agc(H, rssi, tx_valid, active)

    # Their phase feature conjugates across rx. Ours divides along tx. Both are
    # reported because the choice is the substantive disagreement between the
    # two parsers, and one number settles it per capture.
    coherence: dict[str, float] = {}
    if H.shape[1] >= 2:
        coherence["conj_rx"] = cp.lag1_phase_coherence(cp.feature_conj(H)[..., active])
        coherence["ratio_rx"] = cp.lag1_phase_coherence(cp.feature_ratio(H)[..., active])
    if H.shape[2] >= 2:
        tx_pair = H[:, :, 1, :] * np.conj(H[:, :, 0, :])
        coherence["conj_tx"] = cp.lag1_phase_coherence(tx_pair[..., active])
    coherence["raw"] = cp.lag1_phase_coherence(H[:, 0, 0, :][:, active])

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_db = 20 * np.log10(np.abs(H[:, 0, 0, :]))
        agc_db = 20 * np.log10(np.abs(agc[:, 0, 0, :]))

    def spectrum(x: np.ndarray) -> list[float | None]:
        # A null bin is -inf in dB and empty after masking; nanmean would warn
        # on the all-NaN column and return NaN, which is the right value but a
        # noisy way to reach it.
        m = np.where(np.isfinite(x), x, np.nan)
        keep = np.isfinite(m).any(axis=0)
        mean = np.full(m.shape[1], np.nan)
        if keep.any():
            mean[keep] = np.nanmean(m[:, keep], axis=0)
        return [None if not np.isfinite(v) else float(v) for v in mean]

    index: MTKIndex = t["index"]
    macs = index.source_macs
    census: dict[str, int] = {}
    for m in macs:
        census[m] = census.get(m, 0) + 1

    return {
        "frames": int(H.shape[0]),
        "framesInFile": int(index.count),
        "nrx": int(H.shape[1]),
        "ntx": int(H.shape[2]),
        "nsub": int(t["nsub"]),
        "activeBins": [int(b) for b in active],
        "occupancy": [float(v) for v in occ],
        "rawSpectrum": spectrum(raw_db),
        "agcSpectrum": spectrum(agc_db),
        "coherence": {k: float(v) for k, v in coherence.items()},
        "rssiMean": float(np.mean(rssi[:, 0])),
        "twoStreamFrames": int(tx_valid.all(axis=1).sum()),
        "peer": max(census, key=lambda k: census[k]) if census else None,
        "macCensus": sorted(census.items(), key=lambda kv: -kv[1])[:6],
        "tMin": float(t["times"][0]) if len(t["times"]) else 0.0,
        "tMax": float(t["times"][-1]) if len(t["times"]) else 0.0,
        "toolVersion": cp.__version__,
    }


# ---------------------------------------------------------------------- #
#  Tile planes                                                            #
# ---------------------------------------------------------------------- #
#
# The metrics below exist so the vendored parser's output can be looked at in
# the same heatmap as everything else, panned and zoomed through the same tile
# path. A summary of coherence numbers answers "which parser is better"; a
# heatmap answers "what does this parser see", which is the question a capture
# is opened to ask.

LG_METRICS = ("lg_amplitude", "lg_conj_phase", "lg_conj_amplitude")

# Both are properties of the file rather than of a block, and rebuilding either
# per block would dominate the decode. The plane-1 index costs a full scan; the
# active-bin set costs a decode of its own.
_plane1: dict[str, MTKIndex] = {}
_active: dict[str, np.ndarray] = {}


def _plane1_index(path: Path) -> MTKIndex:
    key = str(path)
    if key not in _plane1:
        _plane1[key] = MTKIndex(path, plane=1)
    return _plane1[key]


def _active_mask(path: Path, nsub: int) -> np.ndarray:
    """Their occupancy rule, measured once over a sample of the capture.

    Computed on RAW samples for the reason build_tensor documents: their rule
    counts non-zero bins, so interpolated pilots would read as carrying data.
    """
    key = str(path)
    if key not in _active:
        t = build_tensor(path, max_frames=512, interpolate=False)
        mask = np.zeros(nsub, dtype=bool)
        mask[cp.active_bins(t["H"], t["tx_valid"])] = True
        _active[key] = mask
    return _active[key]


def decode_block(
    path: Path,
    index: MTKIndex,
    frame_ids: np.ndarray,
    *,
    interpolate: bool = True,
) -> dict[str, np.ndarray]:
    """Their planes for a block of frames, shaped as the tile path expects.

    ``(frames, subcarriers)`` float32 with NaN where they have nothing to say —
    the same contract ``mtk.decode_frames`` returns, so these drop into the
    tile cache beside the ordinary metrics.

    Bins their occupancy rule rejects are NaN rather than plotted: showing a
    guard band as though it were a measurement is what the rule exists to
    prevent, and it would dominate the colour scale.
    """
    p0 = decode_complex(path, index, frame_ids, interpolate=interpolate)
    p1 = decode_complex(path, _plane1_index(path), frame_ids, interpolate=interpolate)
    H = np.stack([p0, p1], axis=1).astype(np.complex64)     # (n, rx, tx, sub)
    nsub = H.shape[3]

    tx_valid = np.isfinite(H).any(axis=(1, 3))
    H = np.nan_to_num(H, nan=0.0)
    rssi = np.repeat(index.rssi_1[frame_ids][:, None], H.shape[1], axis=1).astype(np.int16)
    active = _active_mask(path, nsub)

    agc = cp.apply_agc(H, rssi, tx_valid, np.flatnonzero(active))
    conj = cp.feature_conj(H)          # theirs: conjugated ACROSS rx

    def masked(a: np.ndarray) -> np.ndarray:
        out = np.asarray(a, dtype=np.float32).copy()
        out[:, ~active] = np.nan
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 20 * np.log10(np.abs(agc[:, 0, 0, :]))
        camp = 20 * np.log10(np.abs(conj[:, 0, 0, :]))
    amp[~np.isfinite(amp)] = np.nan
    camp[~np.isfinite(camp)] = np.nan

    return {
        "lg_amplitude": masked(amp),
        "lg_conj_amplitude": masked(camp),
        "lg_conj_phase": masked(np.angle(conj[:, 0, 0, :])),
    }
