"""Vectorised batched decoding of FeitCSI frames.

Decodes a selection of frames in one vectorised pass, reproducing CSIKit's
per-frame pipeline (payload unpack, pilot interpolation, RSSI scaling) but
across all selected frames at once. Processes in chunks to keep peak memory
bounded on large captures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from CSIKit.util.matlab import db, dbinv

from .index import FrameIndex, HEADER_BYTES

# Maximum frames decoded per internal chunk. Bounds peak memory: each frame's
# complex matrix is (num_sc, num_rx, num_tx) complex128. For 242x2x1 that is
# ~7.7 KB, so 4096 frames is ~31 MB of complex data plus temporaries.
CHUNK_SIZE = 4096

# Pilot subcarrier indices per (rate_format, channel_width), mirrored from
# FeitCSIBeamformReader.interpolate.
_PILOT_INDICES: dict[tuple[str, str], list[int]] = {
    ("HT", "20"): [7, 21, 34, 48],
    ("VHT", "20"): [7, 21, 34, 48],
    ("HT", "40"): [5, 33, 47, 66, 80, 108],
    ("VHT", "40"): [5, 33, 47, 66, 80, 108],
    ("VHT", "80"): [19, 47, 83, 111, 130, 158, 194, 222],
    ("VHT", "160"): [
        19, 47, 83, 111, 130, 158, 194, 222,
        261, 289, 325, 353, 372, 400, 436, 464,
    ],
    ("HE", "20"): [6, 32, 74, 100, 141, 167, 209, 235],
    ("HE", "40"): [
        6, 32, 74, 100, 140, 166, 208, 234,
        249, 275, 317, 343, 383, 409, 451, 477,
    ],
    ("HE", "80"): [
        32, 100, 166, 234, 274, 342, 408, 476,
        519, 587, 653, 721, 761, 829, 895, 963,
    ],
    ("HE", "160"): [
        32, 100, 166, 234, 274, 342, 408, 476,
        519, 587, 653, 721, 761, 829, 895, 963,
        1028, 1096, 1162, 1230, 1270, 1338, 1404, 1472,
        1515, 1583, 1649, 1717, 1757, 1825, 1891, 1959,
    ],
}


def _cubic(y0: np.ndarray, y1: np.ndarray, y2: np.ndarray, y3: np.ndarray, mu: float) -> np.ndarray:
    """Cubic interpolation, vectorised — mirrors ``cubicInterpolate``."""
    mu2 = mu * mu
    a0 = y3 - y2 - y0 + y1
    a1 = y0 - y1 - a0
    a2 = y2 - y0
    a3 = y1
    return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3


def _linear(y1: np.ndarray, y2: np.ndarray, mu: float) -> np.ndarray:
    """Linear interpolation, vectorised — mirrors ``linearInterpolate``."""
    return y1 * (1 - mu) + y2 * mu


def _apply_interpolation(matrix: np.ndarray, indices: list[int]) -> None:
    """Vectorised pilot interpolation, modifying ``matrix`` in place.

    ``matrix`` has shape (n_frames, num_sc, num_rx, num_tx). Each pilot index
    is processed sequentially (matching upstream), but the arithmetic is
    vectorised across (frame, rx, tx). The wrap-correction branch uses
    ``np.where`` so it is evaluated per element.
    """
    amplitude = np.abs(matrix)
    phase = np.angle(matrix)

    for i in indices:
        amp_interp = _cubic(
            amplitude[:, i - 2], amplitude[:, i - 1],
            amplitude[:, i + 1], amplitude[:, i + 2],
            0.5,
        )
        phase_interp = _linear(phase[:, i - 1], phase[:, i + 1], 0.5)
        # Upstream wrap correction: if phase[i-1] > 2 and phase[i+1] < -2,
        # interpolate from i+1, i+2 with mu=-1 instead.
        mask = (phase[:, i - 1] > 2) & (phase[:, i + 1] < -2)
        phase_wrap = _linear(phase[:, i + 1], phase[:, i + 2], -1)
        phase_interp = np.where(mask, phase_wrap, phase_interp)

        amplitude[:, i] = amp_interp
        phase[:, i] = phase_interp
        matrix[:, i] = amplitude[:, i] * np.exp(1j * phase[:, i])


def _batch_interpolate(matrix: np.ndarray, index: FrameIndex, frame_ids: np.ndarray) -> None:
    """Interpolate pilot subcarriers for all frames in the batch.

    Fast path: if all selected frames share the same (rate_format, channel_width),
    one pilot index set applies to the whole batch. Otherwise frames are grouped
    by that pair and each group is processed independently.
    """
    keys = {
        (index.rate_formats[int(fid)], index.channel_widths[int(fid)])
        for fid in frame_ids
    }
    if len(keys) <= 1:
        key = next(iter(keys)) if keys else ("unknown", "unknown")
        indices = _PILOT_INDICES.get(key, [])
        if indices:
            _apply_interpolation(matrix, indices)
    else:
        for key in keys:
            indices = _PILOT_INDICES.get(key, [])
            if not indices:
                continue
            mask = np.array([
                index.rate_formats[int(fid)] == key[0]
                and index.channel_widths[int(fid)] == key[1]
                for fid in frame_ids
            ])
            sub = matrix[mask]
            _apply_interpolation(sub, indices)
            matrix[mask] = sub


def _batch_scale(matrix: np.ndarray, index: FrameIndex, frame_ids: np.ndarray, num_sc: int) -> None:
    """Vectorised RSSI scaling, reproducing ``csitools.scale_csi_frame``.

    Per (frame, rx): scale = dbinv(rssi_1) / (sum(|csi|^2) / num_sc), then
    multiply by sqrt(scale). Vectorised over frames and rx.
    """
    rssi = index.rssi_1[frame_ids].astype(np.float64)
    rss_pwr = dbinv(rssi)  # 10^(rssi/10), shape (n,)
    # Sum |csi|^2 over (subcarrier, tx) per (frame, rx).
    csi_mag = np.sum(np.abs(matrix) ** 2, axis=(1, 3))  # (n, rx)
    norm_csi_mag = csi_mag / num_sc
    scale = rss_pwr[:, None] / norm_csi_mag  # (n, rx)
    matrix *= np.sqrt(scale)[:, None, :, None]  # broadcast (n, 1, rx, 1)


def _is_contiguous(frame_ids: np.ndarray) -> bool:
    """True if frame_ids is a strictly ascending run of step 1."""
    if len(frame_ids) <= 1:
        return True
    return bool(np.all(np.diff(frame_ids) == 1))


def _decode_chunk(
    path: Path,
    index: FrameIndex,
    frame_ids: np.ndarray,
    csi_length: int,
    num_sc: int,
    num_rx: int,
    num_tx: int,
    *,
    scaled: bool,
    interpolate: bool,
    contiguous_fast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one chunk of frames into (amplitude, phase) float32."""
    n = len(frame_ids)

    # --- Gather payload bytes into a contiguous (n, csi_length) uint8 array ---
    if contiguous_fast and index.stride is not None:
        start_off = int(index.offsets[frame_ids[0]])
        raw = np.memmap(path, dtype="<u1", mode="r", offset=start_off, shape=(n, index.stride))
        raw2d = raw  # already (n, stride)
        payload = raw2d[:, HEADER_BYTES : HEADER_BYTES + csi_length].copy()
        del raw, raw2d  # release memmap
    else:
        payload = np.empty((n, csi_length), dtype="<u1")
        with path.open("rb") as fh:
            for i, fid in enumerate(frame_ids):
                off = int(index.offsets[fid]) + HEADER_BYTES
                fh.seek(off)
                payload[i] = np.frombuffer(fh.read(csi_length), dtype="<u1")

    # --- Decode int16 -> complex ---
    # Payload layout: rx-major, then tx, then subcarrier, each as (real, imag) int16.
    flat = payload.view("<i2").reshape(n, num_rx, num_tx, num_sc, 2)
    matrix = flat[..., 0].astype(np.float64) + 1j * flat[..., 1].astype(np.float64)
    # (n, rx, tx, sc) -> (n, sc, rx, tx)
    matrix = np.ascontiguousarray(matrix.transpose(0, 3, 1, 2))

    # --- Pilot interpolation ---
    if interpolate:
        _batch_interpolate(matrix, index, frame_ids)

    # --- RSSI scaling ---
    if scaled:
        _batch_scale(matrix, index, frame_ids, num_sc)

    # --- Extract first stream and compute amplitude/phase ---
    stream = matrix[:, :, 0, 0]  # (n, num_sc)
    amplitude = db(np.abs(stream)).astype(np.float32)
    phase = np.angle(stream).astype(np.float32)

    return amplitude, phase


def decode_frames(
    path: str | Path,
    index: FrameIndex,
    frame_ids: np.ndarray,
    *,
    scaled: bool = True,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a selection of frames in one vectorised pass.

    Returns (amplitude, phase), both float32, shape (len(frame_ids), num_subcarriers).
    Processes internally in chunks of ``CHUNK_SIZE`` frames to keep peak memory
    bounded. When ``frame_ids`` is a contiguous run under a uniform stride, the
    mmap is sliced directly rather than gathering row by row.
    """
    path = Path(path)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    n = len(frame_ids)
    num_sc = index.num_subcarriers

    if n == 0:
        return (
            np.empty((0, num_sc), dtype=np.float32),
            np.empty((0, num_sc), dtype=np.float32),
        )

    csi_lengths = index.csi_lengths[frame_ids]
    csi_length = int(csi_lengths[0])
    if not np.all(csi_lengths == csi_length):
        raise ValueError("Selected frames have varying csi_length; cannot batch-decode")

    contiguous_fast = index.stride is not None and _is_contiguous(frame_ids)
    num_rx = index.num_rx
    num_tx = index.num_tx

    if n <= CHUNK_SIZE:
        return _decode_chunk(
            path, index, frame_ids, csi_length, num_sc, num_rx, num_tx,
            scaled=scaled, interpolate=interpolate, contiguous_fast=contiguous_fast,
        )

    amps: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    for start in range(0, n, CHUNK_SIZE):
        chunk_ids = frame_ids[start : start + CHUNK_SIZE]
        amp, phase = _decode_chunk(
            path, index, chunk_ids, csi_length, num_sc, num_rx, num_tx,
            scaled=scaled, interpolate=interpolate, contiguous_fast=contiguous_fast,
        )
        amps.append(amp)
        phases.append(phase)

    return np.concatenate(amps), np.concatenate(phases)
