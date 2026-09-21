"""FastAPI app serving FeitCSI parsed readings as JSON and binary tiles."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

import subprocess

from . import farsense, hybrid, lgdetect, lgproc, truth as truthmod
from .presence import CHANNELS
from .stream import get_stream
from .tiles import (
    DOPPLER_METRICS,
    TILE_METRICS,
    _presence_grid,
    compute_doppler,
    compute_presence,
    compute_tile,
    get_index,
    reset_tile_caches,
)
from .index import parse_mac_filter, parse_mimo_filter

DEFAULT_PATH = "captures/capture.dat"
DEFAULT_WINDOW = 200

CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"
# The default window a calibration reference may sit from the capture it
# scores. Named rather than only defaulted, because /api/phase1 compares
# against it to flag a caller that widened it.
_DEFAULT_REF_AGE_H = 6.0
# .dat = FeitCSI, .bin = MediaTek.
CAPTURE_SUFFIXES = (".dat", ".bin")
# Depth cap on the captures/ walk. Deep enough for any sane layout, and a
# hard stop if a directory tree turns out to be pathological.
MAX_CAPTURE_DEPTH = 8

app = FastAPI(title="FeitCSI Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # X-Tile-* headers must be explicitly exposed, otherwise browsers hide
    # them from JavaScript and the tile body is unusable.
    expose_headers=[
        "X-Tile-Width",
        "X-Tile-Height",
        "X-Tile-T0",
        "X-Tile-T1",
        "X-Tile-DT",
        "X-Tile-Level",
        "X-Capture-TMin",
        "X-Capture-TMax",
        "X-Tile-Frames",
        "X-Tile-Total",
        "X-Tile-Exact",
        "X-Tile-Anchored",
        "X-Tile-VMin",
        "X-Tile-VMax",
        "X-Tile-PLow",
        "X-Tile-PHigh",
        "X-Tile-Filled",
        "X-Doppler-Width",
        "X-Doppler-Height",
        "X-Doppler-Fs",
        "X-Doppler-FMin",
        "X-Doppler-FMax",
        "X-Doppler-Win",
        "X-Doppler-Hop",
        "X-Doppler-WinSeconds",
        "X-Doppler-Frames",
        "X-Doppler-ColT0",
        "X-Doppler-ColT1",
        "X-Doppler-Blank",
        "X-Doppler-ScaleMin",
        "X-Doppler-ScaleMax",
    ],
)


ROOTS_ENV_VAR = "FEITCSI_CAPTURE_ROOTS"


def capture_roots() -> list[Path]:
    """Directories the API is allowed to read captures from.

    ``captures/`` always, plus any ``os.pathsep``-separated paths named in
    ``$FEITCSI_CAPTURE_ROOTS``.  Empty by default, so a stock deployment
    reads from exactly one directory.

    The env var exists for deployments that keep captures on a data mount and
    would rather point at it than symlink it in; a symlink inside ``captures/``
    remains the simpler option and is still followed.
    """
    roots = [CAPTURES_DIR]
    extra = os.environ.get(ROOTS_ENV_VAR, "")
    roots.extend(Path(part) for part in extra.split(os.pathsep) if part.strip())
    return roots


def _under_root(target: Path, root: Path) -> bool:
    """True if *target* sits inside *root*, comparing both spellings.

    ``root`` is checked as written and as resolved, because ``captures/`` may
    itself be a symlink: an absolute path handed out by ``/api/captures`` is
    spelled with the unresolved root, so comparing only against the resolved
    one would reject the app's own output.
    """
    for base in {root, root.resolve()}:
        if target == base or base in target.parents:
            return True
    return False


def resolve_capture_path(path: str) -> Path:
    """Validate and resolve a capture file path.

    This is the single chokepoint for all filesystem access from the API, and
    it confines every request to :func:`capture_roots`.

    Accepted spellings, all of which ``/api/captures`` or the README produce:

    * root-relative, including nested — ``capture.dat``, ``2026-08/x.dat``
    * legacy repo-root-relative — ``captures/capture.dat`` (``DEFAULT_PATH``)
    * absolute, as long as it lies inside a root

    A ``..`` component is rejected outright rather than normalised, so no
    request can climb out of a root.  Symlinks *inside* a root are still
    followed wherever they point: placing one requires filesystem access to
    the server, which is a deliberate act by an operator, not something an
    HTTP caller can arrange.

    Rejects the empty string, anything outside every root, and anything that
    is not an existing regular file.
    """
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="path parameter is required")

    not_found = HTTPException(status_code=404, detail=f"File not found: {path}")
    roots = capture_roots()
    p = Path(path)

    if p.is_absolute():
        # Confine before touching the filesystem, and never on a resolved
        # path -- resolving first would let a symlink inside a root decide
        # the verdict, and those are allowed to point outside it.
        if not any(_under_root(p, root) for root in roots):
            raise not_found
        if not p.is_file():
            raise not_found
        return p.resolve()

    if ".." in p.parts:
        raise not_found

    candidates = [p]
    # 'captures/capture.dat' is how DEFAULT_PATH and the README spell it.
    if p.parts and p.parts[0] == CAPTURES_DIR.name:
        candidates.append(Path(*p.parts[1:]))

    for root in roots:
        for candidate in candidates:
            target = root / candidate
            if target.is_file():
                return target.resolve()

    raise not_found


@app.get("/api/snapshot")
def snapshot(
    path: str = Query(DEFAULT_PATH, description="Path to .dat file"),
    max_packets: int = Query(DEFAULT_WINDOW, ge=1, le=10000),
) -> dict:
    p = resolve_capture_path(path)

    # Cached per path: decodes only the bytes appended since the last poll,
    # so refresh cost tracks new frames rather than total capture size.
    stream = get_stream(p)
    try:
        stream.update()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse error: {exc}") from exc

    window = stream.snapshot(max_packets=max_packets)

    amp_finite = window.amplitude[np.isfinite(window.amplitude)]
    phase_finite = window.phase[np.isfinite(window.phase)]
    ratio_amp_finite = window.ratio_amplitude[np.isfinite(window.ratio_amplitude)]
    ratio_phase_finite = window.ratio_phase[np.isfinite(window.ratio_phase)]

    return {
        "filename": window.filename,
        "chipset": window.chipset,
        "bandwidth": window.bandwidth,
        "num_subcarriers": window.num_subcarriers,
        "total_packets": stream.total_frames,
        "window_packets": len(window),
        "time_seconds": window.time_seconds.tolist(),
        "amplitude": window.amplitude.tolist(),
        "phase": window.phase.tolist(),
        "ratio_amplitude": window.ratio_amplitude.tolist(),
        "ratio_phase": window.ratio_phase.tolist(),
        "amp_min": float(np.nanmin(amp_finite)) if amp_finite.size else 0.0,
        "amp_max": float(np.nanmax(amp_finite)) if amp_finite.size else 1.0,
        "phase_min": float(np.nanmin(phase_finite)) if phase_finite.size else -np.pi,
        "phase_max": float(np.nanmax(phase_finite)) if phase_finite.size else np.pi,
        "ratio_amp_min": float(np.nanmin(ratio_amp_finite)) if ratio_amp_finite.size else 0.0,
        "ratio_amp_max": float(np.nanmax(ratio_amp_finite)) if ratio_amp_finite.size else 1.0,
        "ratio_phase_min": float(np.nanmin(ratio_phase_finite)) if ratio_phase_finite.size else -np.pi,
        "ratio_phase_max": float(np.nanmax(ratio_phase_finite)) if ratio_phase_finite.size else np.pi,
    }


@app.get("/api/meta")
def meta(
    path: str = Query(..., description="Path to .dat file"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM' (e.g. '2x1', '2x2')"),
    source_mac: str | None = Query(None, description="Source MAC filter, e.g. 'd8:3a:dd:29:22:f5'"),
) -> dict:
    """Cheap metadata endpoint — index only, never decodes payloads.

    Returns capture geometry and time range.  On a 211 MB capture this returns
    in well under a second because it only builds a FrameIndex.

    ``mimo`` and ``source_mac`` filter the reported counts and time range to
    frames that match. The full capture's geometry (num_subcarriers, num_rx,
    num_tx, bandwidth, chipset) is a property of the file and is reported
    unfiltered.
    """
    p = resolve_capture_path(path)
    idx = get_index(p)

    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mac_filter = parse_mac_filter(source_mac)
    mask = idx.filter_mask(mimo=mimo_filter, source_mac=mac_filter)
    filtered_count = int(mask.sum())

    if filtered_count > 0:
        idxs = np.flatnonzero(mask)
        t_min = float(idx.times[idxs[0]])
        t_max = float(idx.times[idxs[-1]])
    else:
        t_min = 0.0
        t_max = 0.0

    return {
        "filename": p.name,
        "chipset": idx.chipset,
        "bandwidth": idx.bandwidth,
        "num_subcarriers": idx.num_subcarriers,
        "total_frames": filtered_count,
        "t_min": t_min,
        "t_max": t_max,
        "num_rx": idx.num_rx,
        "num_tx": idx.num_tx,
    }


@app.get("/api/filters")
def filters(path: str = Query(..., description="Path to .dat file")) -> dict:
    """Distinct MIMO modes and source MACs present in a capture.

    Used to populate the frontend dropdowns. Cheap (header scan only).
    """
    p = resolve_capture_path(path)
    idx = get_index(p)

    if idx.count == 0:
        return {"mimo_modes": [], "source_macs": []}

    mimo_modes = sorted({
        f"{int(rx)}x{int(tx)}"
        for rx, tx in zip(idx.num_rx_arr, idx.num_tx_arr)
    })
    # Preserve first-seen order for MACs (stable in the index).
    seen: dict[str, None] = {}
    for m in idx.source_macs:
        seen.setdefault(m, None)
    return {"mimo_modes": mimo_modes, "source_macs": list(seen)}


@app.get("/api/tile")
def tile(
    path: str = Query(..., description="Path to .dat file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    width: int = Query(1600, ge=1, description="Output columns (client plot width in pixels; capped at 4096)"),
    metric: str = Query("amplitude", description=f"One of: {', '.join(TILE_METRICS)}"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM' (e.g. '2x1', '2x2')"),
    source_mac: str | None = Query(None, description="Source MAC filter, e.g. 'd8:3a:dd:29:22:f5'"),
    interpolate: bool = Query(
        True,
        description="Linearly interpolate gaps in both axes: structural "
        "nulls (pilots, DC/guard band) along subcarrier, and sampling gaps "
        "along time. False leaves both as NaN, as decoded on the wire.",
    ),
    agc: bool = Query(
        True,
        description="Remove the receiver's own per-gain-state amplitude "
        "distortion. The NIC's AGC holds the level flat but each gain state "
        "has its own frequency response, so gain steps show as single-frame "
        "vertical stripes in the amplitude view that are the radio, not the "
        "room. Affects amplitude and the CIR built on it; the CSI ratio "
        "divides the gain out and is left alone. MediaTek captures only.",
    ),
) -> Response:
    """Pre-aggregated grid at display resolution, as raw little-endian float32.

    The body is a bare ``(num_subcarriers, width)`` float32 array, row-major,
    with row 0 = highest subcarrier index.  Metadata rides in response headers
    so the body stays a buffer the client can wrap in ``Float32Array``.

    ``X-Capture-TMin``/``TMax`` are the whole file's extent, NOT this tile's
    window -- the client already knows the window it asked for, and what it
    cannot know is how far the capture has grown since. Returning it here lets
    a live view track the newest packet without a second /api/meta round trip.
    """
    if metric not in TILE_METRICS:
        raise HTTPException(
            status_code=400,
            detail="metric must be one of: " + ", ".join(f"'{m}'" for m in TILE_METRICS),
        )

    p = resolve_capture_path(path)

    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mac_filter = parse_mac_filter(source_mac)

    grid, meta = compute_tile(
        p, t0, t1, width, metric,
        mimo=mimo_filter, source_mac=mac_filter, interpolate=interpolate,
        agc_correct=agc,
    )

    body = grid.astype("<f4", copy=False).tobytes()

    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "X-Tile-Width": str(grid.shape[1]),
            "X-Tile-Height": str(grid.shape[0]),
            # The window this tile actually covers. Columns are quantised to
            # the lattice, so the tile spans the smallest aligned range
            # containing the request and the client crops -- which is what
            # stops a pan or a live poll re-quantising the whole picture. A
            # client that ignores these and assumes it got the window it asked
            # for will draw the tile shifted by up to one column.
            "X-Tile-T0": str(meta["t0"]),
            "X-Tile-T1": str(meta["t1"]),
            "X-Tile-DT": str(meta["dt"]),
            "X-Tile-Level": str(meta["level"]),
            "X-Capture-TMin": str(meta["t_min"]),
            "X-Capture-TMax": str(meta["t_max"]),
            "X-Tile-Frames": str(meta["frames_decoded"]),
            "X-Tile-Total": str(meta["total_in_range"]),
            "X-Tile-Exact": "1" if meta["exact"] else "0",
            # 0 when a correction metric had no absolute orientation to
            # anchor to, so its polarity is not comparable with another view.
            "X-Tile-Anchored": "1" if meta["anchored"] else "0",
            # 1 when the AGC correction actually applied. 0 means the values
            # are exactly as decoded -- the toggle is off, the metric is one
            # the table does not touch, the capture is not MediaTek, or it
            # never changed gain state.
            "X-Tile-Agc": "1" if meta["agc_corrected"] else "0",
            "X-Tile-AgcStates": str(meta["agc_states"]),
            "X-Tile-VMin": str(meta["vmin"]),
            "X-Tile-VMax": str(meta["vmax"]),
            "X-Tile-PLow": str(meta["p_low"]),
            "X-Tile-PHigh": str(meta["p_high"]),
            "X-Tile-Filled": str(meta["filled_columns"]),
        },
    )


@app.get("/api/doppler")
def doppler(
    path: str = Query(..., description="Path to capture file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    metric: str = Query("csi_ratio_complex", description=f"One of: {', '.join(DOPPLER_METRICS)}"),
    win_seconds: float = Query(10.0, gt=0, le=600, description="STFT window length in seconds; clamped to the range if longer"),
    overlap: float = Query(0.5, ge=0.0, lt=1.0, description="Window overlap fraction"),
    max_gap_fraction: float = Query(0.5, gt=0.0, le=1.0, description="Blank a column once more than this fraction of its window is interpolated across dropouts"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM'"),
    source_mac: str | None = Query(None, description="Source MAC filter"),
    interpolate: bool = Query(True, description="Fill structural subcarrier nulls before transforming"),
) -> Response:
    """Subcarrier-averaged Doppler spectrogram, as raw little-endian float32.

    The body is a bare ``(win // 2 + 1, n_windows)`` array, row-major, row 0 =
    highest Doppler frequency -- the same row order ``/api/tile`` uses, so the
    same client-side renderer draws it.

    The frequency axis runs ``X-Doppler-FMin`` to ``X-Doppler-FMax``, and
    whether it is one-sided depends on the metric. ``amplitude`` and
    ``csi_ratio_phase_time_unwrapped`` are real signals: their spectra are
    conjugate-symmetric, ``FMin`` is 0, and the sign of the Doppler shift is
    not recoverable -- approaching and receding motion land on the same row.
    ``csi_ratio_complex`` is the complex ratio itself, so the axis is
    two-sided, runs about -Nyquist to +Nyquist, and the sign is real: positive
    is one direction of radial motion and negative the other. Both chains
    share an oscillator, so the carrier frequency offset that would otherwise
    bias the whole axis divides out of a ratio, and what is left is geometry.

    ``X-Doppler-Fs`` is the capture's own median frame rate over the frames in
    range, so ``FMax`` is this file's true Nyquist rather than a function of
    any requested width. Motion above it aliases: at 5 GHz a 1 m/s movement
    sits near 33 Hz, well above every capture this was built against.

    A window longer than the range holds is clamped rather than refused, so
    zooming in never blanks the panel; ``X-Doppler-WinSeconds`` reports what
    was actually used. ``X-Doppler-Blank`` counts columns dropped for being
    mostly interpolated across dropouts.

    **Values are normalised dB, not raw magnitude.** An unnormalised FFT bin
    scales with the window length -- a unit tone reads 213.5 at a 600-sample
    window and 71.3 at 200, exactly the ratio of the two -- so the same motion
    changed brightness whenever the window was clamped by a zoom. Each branch
    is divided by its taper's coherent gain, which leaves the tone's own
    amplitude, and then expressed in dB because the values span five orders of
    magnitude. ``X-Doppler-ScaleMin``/``ScaleMax`` carry the fixed colour range
    that follows from this, measured over ten September captures rather than
    chosen: it spans the complex panel's 1st-to-99th percentile and contains
    the phase panel's whole range. A client that auto-fits instead will make
    two captures of the same room look different.
    """
    p = resolve_capture_path(path)

    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        spec, meta = compute_doppler(
            p, t0, t1, metric,
            win_seconds=win_seconds,
            overlap=overlap,
            max_gap_fraction=max_gap_fraction,
            mimo=mimo_filter,
            source_mac=parse_mac_filter(source_mac),
            interpolate=interpolate,
        )
    except ValueError as exc:
        # Bad metric, or a window the requested range cannot hold. Both are
        # the caller's parameters rather than a server fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=spec.astype("<f4", copy=False).tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Doppler-Width": str(spec.shape[1]),
            "X-Doppler-Height": str(spec.shape[0]),
            "X-Doppler-Fs": str(meta["fs"]),
            "X-Doppler-FMin": str(meta["f_min"]),
            "X-Doppler-FMax": str(meta["f_max"]),
            "X-Doppler-Win": str(meta["win"]),
            "X-Doppler-Hop": str(meta["hop"]),
            "X-Doppler-WinSeconds": str(meta["win_seconds"]),
            "X-Doppler-Blank": str(meta["blank_columns"]),
            "X-Doppler-Frames": str(meta["frames_used"]),
            "X-Doppler-ColT0": str(meta["col_t0"]),
            "X-Doppler-ColT1": str(meta["col_t1"]),
            # The fixed dB range this panel should be drawn with. Values are
            # normalised by the taper's coherent gain, so the same number means
            # the same motion at every window length and every zoom -- which is
            # what makes a fixed scale possible at all.
            "X-Doppler-ScaleMin": str(meta["scale_min"]),
            "X-Doppler-ScaleMax": str(meta["scale_max"]),
            "X-Capture-TMin": str(meta["t_min"]),
            "X-Capture-TMax": str(meta["t_max"]),
            "X-Tile-VMin": str(meta["vmin"]),
            "X-Tile-VMax": str(meta["vmax"]),
            "X-Tile-PLow": str(meta["p_low"]),
            "X-Tile-PHigh": str(meta["p_high"]),
        },
    )


def _nullable(values: np.ndarray) -> list[float | None]:
    """Serialise a float array with non-finite entries as JSON ``null``.

    ``json.dumps`` writes a bare ``NaN``, which is not JSON and which
    ``JSON.parse`` rejects outright -- so a single blanked window would take
    the whole response down. ``null`` is also the right thing for a chart to
    receive: it draws a break in the line rather than a zero, which is exactly
    what a window with no verdict should look like.
    """
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


@app.get("/api/lgdetect")
def lg_detect(   # not `lgdetect`: that name is the module this calls
    path: str = Query(..., description="Path to the capture"),
    margin_s: float = Query(truthmod.DEFAULT_MARGIN_S, ge=0, le=60, description="Empty camera frames within this many seconds of a transition are not scored; 0 trusts every frame"),
    threshold: float = Query(26.0, gt=0, le=200, description="Its change-detection threshold in dB; 26 is the value the script ships with"),
    absence: float = Query(10.0, gt=0, le=600, description="Seconds without movement before it reports absence"),
) -> dict:
    """Replay the LG on-board detector over a capture and report what it said.

    The detector runs under a NumPy 1.x interpreter, matching the board. Its
    length arithmetic shifts a numpy uint8 left by 8, which NumPy 2 keeps as
    uint8 and evaluates to 0 -- the TLV walk then desynchronises at the first
    CSI field and produces frames with zeroed imaginary parts rather than an
    error. Replayed on the project venv it would be measuring the NumPy
    version rather than the detector, so it gets its own venv and a subprocess.

    A first run costs about 20 s per 5-minute capture -- it loops over 256
    subcarriers per record in pure Python -- and is cached thereafter.

    Where the capture carries webcam labels, its verdict is scored against
    them. The comparison is the point: the detector is a frame-to-frame
    amplitude-difference trigger, so it answers "is something changing", and
    the labels say whether anyone was actually there.
    """
    p = resolve_capture_path(path)
    try:
        result = lgdetect.run(p, threshold=threshold, absence=absence)
    except lgdetect.BoardEnvMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"replay timed out: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    spans = lgdetect.intervals(result["events"], result["duration"])
    out = dict(result)
    out["intervals"] = [{"t0": a, "t1": b} for a, b in spans]

    # Score against the webcam labels when the capture has them.
    out["truth"] = None
    stem = p.with_suffix("")
    cv_path = stem.parent / f"{stem.name}_cv.json"
    meta_path = stem.parent / f"{stem.name}_meta.json"
    if cv_path.exists():
        try:
            cv = json.loads(cv_path.read_text())
            base = None
            if meta_path.exists():
                base = json.loads(meta_path.read_text()).get("capture_start_utc_epoch")
            frames = cv.get("frames") or []
            if base is None and frames:
                base = frames[0].get("epoch")
            if base is not None and frames:
                times, truth = [], []
                for f in frames:
                    if f.get("epoch") is None:
                        continue
                    times.append(float(f["epoch"]) - float(base))
                    truth.append(bool(f.get("boxes")))
                said = [any(a <= t < b for a, b in spans) for t in times]
                # Empty frames within margin_s of a transition are the seconds
                # the camera cannot vouch for -- see backend.truth -- and are
                # scored in neither direction.
                ambiguous = truthmod.ambiguous_frames(
                    np.asarray(times), np.asarray(truth, dtype=bool), margin_s
                )
                scored = [(s, t) for s, t, a in zip(said, truth, ambiguous) if not a]
                tp = sum(1 for s, t in scored if s and t)
                fp = sum(1 for s, t in scored if s and not t)
                fn = sum(1 for s, t in scored if not s and t)
                tn = sum(1 for s, t in scored if not s and not t)
                n = max(len(scored), 1)
                out["truth"] = {
                    "timeS": times,
                    "present": truth,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "excluded": int(ambiguous.sum()),
                    "marginSeconds": float(margin_s),
                    "accuracy": (tp + tn) / n,
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                    # What it would score by always saying "present": the bar
                    # any detector has to clear to have said anything.
                    "baseRate": sum(t for _, t in scored) / n,
                }
        except (OSError, ValueError):
            out["truth"] = None
    return out


@app.get("/api/phase1")
def phase1(
    path: str = Query(..., description="Path to the capture"),
    grid: float = Query(1.0, gt=0.05, le=60, description="Seconds per verdict; 1 s is the camera's own rate and there is no ground truth finer"),
    margin_s: float = Query(truthmod.DEFAULT_MARGIN_S, ge=0, le=60, description="Empty camera frames within this many seconds of a transition are not scored -- the walk between door and chair, which the CSI sees and the camera's ROI does not; 0 trusts every frame"),
    k: float = Query(3.0, gt=0, le=100, description="Threshold as a multiple of the reference's dev_scale"),
    lg_threshold: float = Query(26.0, gt=0, le=200, description="LG's change-detection threshold in dB"),
    lg_absence: float = Query(10.0, gt=0, le=600, description="Seconds without movement before LG reports absence"),
    ref_age_h: float = Query(_DEFAULT_REF_AGE_H, gt=0, le=720, description="How far in time a calibration capture may sit from this one. LOAD-BEARING: the pool screen cannot replace it, see below"),
    pool: int = Query(5, ge=2, le=32, description="How many empty captures to calibrate against"),
) -> dict:
    """Both detectors on one grid, against the camera, honestly calibrated.

    The two are not comparable as they stand: ours reduces a window to a single
    verdict, LG's emits movement events and holds a state between them. So both
    are resampled onto the same grid before anything is counted.

    What makes this different from ``/api/presence`` is where the reference
    comes from. That endpoint takes ranges the caller names, and in the UI those
    are the capture's own empty stretches -- which is knowing the answer in
    advance. Here the reference is drawn only from OTHER captures the camera
    labelled completely empty, and never from this one. It also needs at least
    two of them: a single capture reports how far it wanders from itself (about
    0.04 dB overnight), not how far two captures sit apart (about 0.2 dB even
    ten minutes apart), and a threshold built on the former is roughly 5x too
    low. Measured on the 20260914 sessions, which have exactly one clean empty
    capture between them: every threshold from 0.08 to 3.7 dB gave either 100%
    recall at 0% specificity or the reverse.

    Two different failures are guarded, and NEITHER guard covers the other:

    ``ref_age_h`` bounds how far in time a reference may sit from the capture,
    and is the only thing standing between a verdict and a pool that describes
    a different room. ``screen_reference_pool`` cannot do that job: it is shown
    the reference profiles and never the capture's, so it can only ask whether
    the references agree with each OTHER. Measured -- five camera-empty captures
    from one 20260827 morning agree to 0.161 dB, give a healthy dev_scale of
    0.175 and a 0.524 dB threshold, and applied to a capture from 20260904 put
    100% of its windows above that threshold, its quietest one included, at 9x
    the threshold. The screen passes it; only the 6 h window stops it. Widening
    ref_age_h buys silently wrong verdicts, so treat the default as a limit
    rather than a starting point.

    The screen guards the other direction: a pool whose own members disagree
    makes dev_scale the gap between them instead of the wander within either.

    When no usable reference exists this returns ``calibrated: false`` and no
    verdict for our side, rather than falling back to something that would
    produce a number. LG needs no reference at all -- it compares each frame to
    the one before -- so its side is still scored, which is precisely the
    trade-off the two approaches make.
    """
    p = resolve_capture_path(path)
    out: dict = {"path": str(p), "gridSeconds": grid, "marginSeconds": margin_s}

    truth = _camera_truth(p)
    if truth is None:
        raise HTTPException(
            status_code=404,
            detail="this capture has no camera labels, so neither detector can be scored",
        )
    out["groundTruth"] = {"timeS": truth[:, 0].tolist(),
                          "present": [bool(v) for v in truth[:, 1]]}

    duration = float(truth[-1, 0])
    centres = np.arange(grid / 2, duration, grid)
    out["timeS"] = centres.tolist()

    # ---- ours, only if a reference exists that is not this capture ----------
    refs = _empty_reference_pool(p, ref_age_h, pool)
    ref_age = _nearest_reference_age_h(p, refs)
    if len(refs) < 2:
        out["ours"] = None
        out["calibrated"] = False
        out["calibrationNote"] = (
            f"needs 2 or more camera-empty captures within {ref_age_h:g} h that are "
            f"not this one; found {len(refs)}"
        )
    else:
        try:
            state, thr, scale, spread_out, min_dev = _score_ours(
                p, refs, grid, k, centres)
        except ValueError as exc:
            out["ours"] = None
            out["calibrated"] = False
            out["calibrationNote"] = str(exc)
        else:
            out["calibrated"] = True
            # ref_age_h is the only guard against a pool from another room, and
            # it is the caller's to widen. Widening it leaves no other trace --
            # 20260827 references against a 20260904 capture look healthy in
            # every reported number (poolSpread 0.283, devScale 0.445) and score
            # 0.006 specificity. So when the pool sits further out than the
            # default allows, say so in the payload rather than relying on a
            # reader to notice referenceAgeH is 197 instead of 6.
            if ref_age > _DEFAULT_REF_AGE_H:
                out["referenceWarning"] = (
                    f"nearest reference is {ref_age:.1f} h away, past the "
                    f"{_DEFAULT_REF_AGE_H:g} h default; nothing else here "
                    f"detects a pool that describes a different room"
                )
            out["ours"] = {
                "present": [bool(v) for v in state],
                "threshold": thr,
                "devScale": scale,
                "poolSpread": spread_out,
                "minDeviation": min_dev,
                # How far the capture's quietest window still sits from the
                # pool, in thresholds. DIAGNOSTIC ONLY -- it does not separate
                # a usable calibration from a broken one, and was briefly wired
                # to a warning that did. Measured over 22 calibrations from the
                # August protocol that scored 94% recall at 91% specificity, it
                # spans 0.09 to 5.65; two calibrations known to be broken
                # (20260827 references against 20260904 captures, specificity
                # 0.006) read 4.94 and 5.20, inside that range. It cannot
                # separate them because it mixes three things: how far the pool
                # is (wanted), how tight the pool is (the denominator), and how
                # occupied the capture is (the numerator -- every healthy case
                # above 5 is occupied 96-100% of the time and so is honestly far
                # from any empty reference).
                "applicability": (min_dev / thr) if thr > 0 else None,
                # Hours to the nearest reference. Unlike the two numbers above
                # this one actually guards against a pool from another room,
                # because it is the only thing that does -- see the endpoint
                # docstring.
                "referenceAgeH": ref_age,
                "references": [r.name for r in refs],
                "confusion": _confusion(centres, state, truth, grid, margin_s),
            }

    # ---- theirs, which never needs one --------------------------------------
    try:
        lg = lgdetect.run(p, threshold=lg_threshold, absence=lg_absence)
    except lgdetect.BoardEnvMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    lg_state = _lg_state_on_grid(lg.get("events") or [], centres)
    out["lg"] = {
        "present": [bool(v) for v in lg_state],
        "threshold": lg_threshold,
        "absence": lg_absence,
        "events": len(lg.get("events") or []),
        "confusion": _confusion(centres, lg_state, truth, grid, margin_s),
    }
    return out


def _camera_truth(capture: Path) -> np.ndarray | None:
    """(time_s, present) per camera frame, relative to the capture's start."""
    cv_path = capture.with_name(f"{capture.stem}_cv.json")
    if not cv_path.is_file():
        return None
    frames = (json.loads(cv_path.read_text()).get("frames") or [])
    if not frames:
        return None
    base = frames[0].get("epoch")
    rows = []
    for f in frames:
        e = f.get("epoch")
        if e is None:
            continue
        present = bool(f.get("n", 0) > 0 and float(f.get("max_conf") or 0.0) >= 0.5)
        rows.append([float(e) - float(base), 1.0 if present else 0.0])
    return np.asarray(rows) if rows else None


def _empty_reference_pool(capture: Path, max_age_h: float, pool: int) -> list[Path]:
    """Captures the camera saw as completely empty, nearest in time, never this one.

    Both directions in time are allowed: a reference recorded an hour after a
    run describes the same room as one from an hour before, and requiring
    "before" alone discards half the evidence for no physical reason.
    """
    from datetime import datetime

    def stamp_of(q: Path) -> datetime | None:
        try:
            return datetime.strptime(q.stem[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    here = stamp_of(capture)
    if here is None:
        return []
    found: list[tuple[float, Path]] = []
    seen: set[Path] = set()
    for root in capture_roots():
        for cand in _walk_captures(root, MAX_CAPTURE_DEPTH, seen):
            if cand == capture or cand.suffix not in CAPTURE_SUFFIXES:
                continue
            when = stamp_of(cand)
            if when is None:
                continue
            age = abs((here - when).total_seconds())
            if age > max_age_h * 3600:
                continue
            cv_path = cand.with_name(f"{cand.stem}_cv.json")
            if not cv_path.is_file():
                continue
            try:
                summary = json.loads(cv_path.read_text()).get("summary") or {}
            except (OSError, json.JSONDecodeError):
                continue
            if summary.get("fraction_occupied") == 0.0:
                found.append((age, cand))
    found.sort(key=lambda pair: pair[0])
    return [q for _, q in found[:pool]]


def _nearest_reference_age_h(capture: Path, refs: list[Path]) -> float:
    from datetime import datetime

    def when(q: Path) -> datetime | None:
        try:
            return datetime.strptime(q.stem[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    here = when(capture)
    ages = [abs((here - w).total_seconds()) / 3600.0
            for w in (when(r) for r in refs) if here and w]
    return min(ages) if ages else float("nan")


def _score_ours(capture: Path, refs: list[Path], grid: float, k: float,
                centres: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    from . import presence as presence_mod

    grids, fs_ref = [], None
    for r in refs:
        g, _, f, *_ = _presence_grid(r, 0.0, 120.0, mimo=None, source_mac=None,
                                     interpolate=True)
        grids.append(g)
        fs_ref = f if fs_ref is None else fs_ref
    g, _, fs, gtimes, *_ = _presence_grid(capture, 0.0, 1e9, mimo=None,
                                          source_mac=None, interpolate=True)
    grids = [x for x in grids if x.shape[1] == g.shape[1]]
    if len(grids) < 2:
        raise ValueError(
            "the reference captures do not share this capture's subcarrier width"
        )

    # Screen the pool before trusting it. A pool spanning two different room
    # states makes dev_scale measure the gap between them, not the wander
    # within either: 20260914_131846 and 20260914_193002 are both camera-empty
    # and sit 10.6 dB apart, which produced a 63 dB threshold and a detector
    # that never fired once across a capture with 122 s of occupancy. Silently
    # returning that is worse than returning nothing.
    all_profiles = [presence_mod.amplitude_profile(g_) for g_ in grids]
    keep, spread = presence_mod.screen_reference_pool(all_profiles)
    if len(keep) < 2:
        raise ValueError(
            f"the {len(grids)} empty captures nearest this one disagree by "
            f"{spread:.2f} dB, so they describe different room states rather "
            f"than one room twice; no subset of 2 or more agrees within "
            f"{presence_mod.DEFAULT_MAX_POOL_SPREAD_DB:g} dB"
        )
    grids = [grids[i] for i in keep]
    ref = presence_mod.presence_reference(grids, fs_ref, scale_mode="loo")
    thr = k * ref["dev_scale"]
    profiles = ref["profiles"]

    n = max(1, int(round(grid * fs)))
    times, devs = [], []
    for start in range(0, g.shape[0] - n + 1, n):
        w = presence_mod.amplitude_profile(g[start:start + n])
        near = [presence_mod.baseline_deviation(w, q) for q in profiles]
        near = [v for v in near if np.isfinite(v)]
        if not near:
            continue
        times.append(float(gtimes[0]) + (start + n / 2) / fs)
        devs.append(min(near))
    if not times:
        raise ValueError("no window in this capture carries a CSI ratio")
    devs_a = np.asarray(devs)
    state = np.interp(centres, np.asarray(times), devs_a,
                      left=np.nan, right=np.nan) > thr
    return (state, float(thr), float(ref["dev_scale"]), float(spread),
            float(devs_a.min()))


def _lg_state_on_grid(events: list[dict], centres: np.ndarray) -> np.ndarray:
    """Its +/- events are a step function; sample it. It starts absent."""
    state = np.zeros(centres.shape, dtype=bool)
    current, idx = False, 0
    for ev in sorted(events, key=lambda e: e.get("t", 0.0)):
        while idx < centres.size and centres[idx] < ev.get("t", 0.0):
            state[idx] = current
            idx += 1
        current = ev.get("kind") == "+"
    state[idx:] = current
    return state


def _confusion(centres: np.ndarray, state: np.ndarray, truth: np.ndarray,
               grid: float, margin_s: float = truthmod.DEFAULT_MARGIN_S) -> dict:
    """Counted only where the camera is unambiguous across the whole cell.

    ``margin_s`` removes the empty frames next to a transition before the
    cell is judged -- see ``backend.truth`` for why those are the label's
    error, not the detector's. What was removed comes back as ``excluded``.
    """
    cells, excluded = truthmod.cell_truth(
        centres, truth[:, 0], truth[:, 1] > 0.5, grid / 2, margin_s
    )
    return truthmod.confusion(cells, np.asarray(state, dtype=bool), excluded)


@app.get("/api/lgparse")
def lgparse(
    path: str = Query(..., description="Path to the capture"),
    max_frames: int = Query(4096, ge=64, le=20000, description="Frames sampled evenly across the capture"),
) -> dict:
    """Run the vendored MT7921 parser's processing over a capture.

    Their arithmetic, this project's reader: see ``backend/lgproc``. The point
    is a second opinion on the same bytes, so every number here comes from
    calling their functions rather than reimplementing them, and the axis
    convention is theirs -- ``H`` is (packet, rx, tx, subcarrier) with rx = our
    rpi and tx = our tpi.

    The comparison worth reading is ``coherence``: their ``feature_conj``
    conjugates across rx, this project divides along tx, and the two disagree
    about which is the usable phase signal. One capture settles it.
    """
    p = resolve_capture_path(path)
    try:
        return lgproc.summarise(p, max_frames=max_frames)
    except ValueError as exc:
        # A capture with no two-stream frame has nothing for them to work on.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/labels")
def labels(
    path: str = Query(..., description="Path to the capture"),
) -> dict:
    """Ground-truth labels recorded alongside a capture, if any.

    Two sidecars, written by ``scripts/run_experiment.sh`` and the CV pass that
    follows it, sit next to the capture and share its stamp:

    ``<stamp>_cv.json``    per-frame person detections from the webcam
    ``<stamp>_meta.json``  the run's intended protocol phases

    Times come back relative to the capture's first sample, which is what every
    other endpoint here plots against. The frame filenames are stamped in local
    time while the sidecar carries a resolved UTC epoch per frame, so the
    conversion goes through ``capture_start_utc_epoch`` and never parses a
    local-time string -- a naive read of those names lands hours out.

    Phases are the operator's INTENDED protocol, not observed truth: measured
    transitions on the first labelled runs ran 2-18 s late in both directions.
    Where the two disagree the detections are the better evidence, which is why
    both are returned rather than reconciled here.

    Absent sidecars are not an error -- most captures have none.
    """
    p = resolve_capture_path(path)
    stem = p.with_suffix("")
    cv_path = stem.parent / f"{stem.name}_cv.json"
    meta_path = stem.parent / f"{stem.name}_meta.json"

    out: dict = {
        "present": None,
        "phases": None,
        "source": None,
        "captureStartUtcEpoch": None,
    }

    t0: float | None = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            m = None
        if m:
            t0 = m.get("capture_start_utc_epoch")
            out["captureStartUtcEpoch"] = t0
            out["position"] = m.get("position")
            phases = m.get("phases") or []
            out["phases"] = [
                {
                    "label": str(ph.get("label", "")),
                    "t0": float(ph.get("start_s", 0.0)),
                    "t1": float(ph.get("end_s", 0.0)),
                }
                for ph in phases
            ]

    if cv_path.exists():
        try:
            c = json.loads(cv_path.read_text())
        except (OSError, ValueError):
            c = None
        if c:
            frames = c.get("frames") or []
            # Prefer the meta anchor; fall back to the first frame's own epoch
            # so a capture with a cv sidecar but no meta still plots.
            base = t0
            if base is None and frames:
                base = frames[0].get("epoch")
            if base is not None:
                times, present, conf = [], [], []
                for f in frames:
                    e = f.get("epoch")
                    if e is None:
                        continue
                    times.append(float(e) - float(base))
                    boxes = f.get("boxes") or []
                    present.append(bool(boxes))
                    conf.append(float(f.get("max_conf") or 0.0))
                out["present"] = {
                    "timeS": times,
                    "present": present,
                    "maxConf": conf,
                    "roi": c.get("roi"),
                    "model": c.get("model"),
                }
                out["source"] = cv_path.name

    return out


@app.get("/api/presence")
def presence(
    path: str = Query(..., description="Path to capture file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    channel: str = Query("complex", description=f"One of: {', '.join(CHANNELS)}"),
    window_seconds: float = Query(30.0, gt=0, le=600, description="Analysis window length in seconds; clamped to the range if longer"),
    hop_seconds: float = Query(1.0, gt=0, le=60, description="Step between windows in seconds"),
    rpm_lo: float = Query(9.0, gt=0, le=120, description="Slowest breathing rate considered"),
    rpm_hi: float = Query(30.0, gt=0, le=120, description="Fastest breathing rate considered"),
    bandpass_lo: float = Query(0.1, gt=0, le=5, description="Bandpass low edge (Hz)"),
    bandpass_hi: float = Query(0.6, gt=0, le=5, description="Bandpass high edge (Hz)"),
    motion_frac_lo: float = Query(0.10, gt=0, le=5, description="Fractional channel change below which the motion gate is fully open"),
    motion_frac_hi: float = Query(0.25, gt=0, le=5, description="Fractional channel change above which a window counts as gross motion"),
    max_gap_fraction: float = Query(0.5, gt=0.0, le=1.0, description="Report a window as unknown once more than this fraction of it is interpolated across dropouts"),
    smooth_windows: int = Query(3, ge=1, le=51, description="Windows averaged when smoothing the score"),
    present_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Breathing score above which the rate found in a window is believed and reported; evidence only, it does not decide occupancy"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM'"),
    source_mac: str | None = Query(None, description="Source MAC filter"),
    ref_t0: list[float] | None = Query(None, description="Start of a known-empty reference range (seconds); required with ref_t1. Repeat the parameter to pool several empty stretches -- a room that changes across a capture needs all of them, or the trailing empty phase reads as occupied"),
    ref_t1: list[float] | None = Query(None, description="End of a known-empty reference range (seconds); required with ref_t0. Repeat alongside ref_t0, in the same order"),
    ref_path: str | None = Query(None, description="Capture holding the reference range; defaults to path"),
    baseline_dev_k: float = Query(3.0, gt=0, le=50, description="Channel-state deviation that counts as an occupant, in multiples of the reference room's own variability"),
    motion_ratio_hi: float = Query(2.0, gt=1, le=100, description="Gross motion, as a multiple of the reference room's fractional-motion floor"),
    interpolate: bool = Query(True, description="Fill structural subcarrier nulls before transforming"),
) -> dict:
    """Motion level and static-presence verdicts over a time range, as JSON.

    One entry per analysis window in each series, on the capture's own clock,
    so the result drops straight onto the time axis the heatmaps share.

    JSON rather than the binary framing ``/api/tile`` and ``/api/doppler`` use:
    the payload is a handful of scalar series a few hundred entries long, not
    a grid, so binary would save nothing worth the loss of being able to read
    a response.

    ``state`` is the verdict per window and is the only field a caller needs
    to draw the strip; everything else is the evidence it was built from.
    ``unknown`` marks windows assembled mostly from samples interpolated
    across a capture dropout -- those report no score and no rate, because the
    alternative is reporting invented data as an empty room.

    **Filter by source MAC.** Frames from different transmitters are different
    channels, and interleaving two of them makes consecutive samples alternate
    between unrelated propagation paths. Measured on captures/capture.dat,
    that lifts the fractional motion level from 0.37-0.47 per transmitter to
    0.53 mixed -- decorrelation read as movement.
    """
    p = resolve_capture_path(path)

    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = compute_presence(
            p, t0, t1,
            channel=channel,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            rate_band_rpm=(rpm_lo, rpm_hi),
            bandpass_hz=(bandpass_lo, bandpass_hi),
            motion_frac_lo=motion_frac_lo,
            motion_frac_hi=motion_frac_hi,
            max_gap_fraction=max_gap_fraction,
            smooth_windows=smooth_windows,
            present_threshold=present_threshold,
            ref_t0=ref_t0,
            ref_t1=ref_t1,
            ref_path=None if ref_path is None else resolve_capture_path(ref_path),
            baseline_dev_k=baseline_dev_k,
            motion_ratio_hi=motion_ratio_hi,
            mimo=mimo_filter,
            source_mac=parse_mac_filter(source_mac),
            interpolate=interpolate,
        )
    except ValueError as exc:
        # A band the capture's rate cannot reach, a window too short for the
        # rate band, an empty range: all the caller's parameters rather than
        # a server fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "time_s": [float(v) for v in result["time_s"]],
        "state": result["state"],
        "score": _nullable(result["score"]),
        "periodicity": _nullable(result["periodicity"]),
        "tonality": _nullable(result["tonality"]),
        "motion_gate": _nullable(result["motion_gate"]),
        "motion_level": _nullable(result["motion_level"]),
        "motion_ratio": _nullable(result["motion_ratio"]),
        "baseline_dev": _nullable(result["baseline_dev"]),
        "breathing": [bool(v) for v in result["breathing"]],
        "rate_rpm": _nullable(result["rate_rpm"]),
        "unknown": [bool(v) for v in result["unknown"]],
        "fs_hz": result["fs_hz"],
        "win": result["win"],
        "hop": result["hop"],
        "window_seconds": result["window_seconds"],
        "rpm_floor_eff": result["rpm_floor_eff"],
        "baseline_dev_threshold": result["baseline_dev_threshold"],
        "reference": result["reference"],
        "frames_used": result["frames_used"],
        "frames_without_ratio": result["frames_without_ratio"],
        "t_min": result["t_min"],
        "t_max": result["t_max"],
        "params": result["params"],
        "warnings": result["warnings"],
    }


def _nullable_rows(values: np.ndarray, decimals: int = 5) -> list[list[float | None]]:
    """``_nullable`` over each row of a 2-D array, rounded to keep the JSON small."""
    arr = np.round(np.asarray(values, dtype=float), decimals)
    return [_nullable(row) for row in arr]


@app.get("/api/farsense")
def farsense_detector(   # not `farsense`: that name is the module this calls
    path: str = Query(..., description="Path to capture file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    window_seconds: float = Query(farsense.WINDOW_SECONDS, gt=0, le=120, description="Projection window; the paper uses 12 s"),
    hop_seconds: float = Query(farsense.HOP_SECONDS, gt=0, le=60, description="Step between windows in seconds"),
    rpm_lo: float = Query(farsense.RATE_BAND_RPM[0], gt=0, le=120, description="Slowest breathing rate considered; the paper uses 10"),
    rpm_hi: float = Query(farsense.RATE_BAND_RPM[1], gt=0, le=120, description="Fastest breathing rate considered; the paper uses 37"),
    n_theta: int = Query(farsense.N_THETA, ge=2, le=720, description="Projection angles swept over 0..2pi; the paper uses 100"),
    keep_fraction: float = Query(farsense.BNR_KEEP_FRACTION, ge=0, le=1, description="Subcarriers with BNR below this fraction of the best are excluded; the paper uses 0.7"),
    savgol_seconds: float = Query(farsense.SAVGOL_SECONDS, ge=0, le=5, description="Savitzky-Golay window in seconds; 0 disables it"),
    savgol_order: int = Query(farsense.SAVGOL_ORDER, ge=1, le=7, description="Savitzky-Golay polynomial order"),
    highpass_hz: float = Query(0.0, ge=0, le=5, description="Zero-phase high-pass before smoothing, in Hz; 0 reproduces the paper"),
    motion_frac_hi: float = Query(farsense.MOTION_FRAC_HI, gt=0, le=5, description="Median fractional channel change above which a window is non-stationary and reports no rate"),
    min_peak: float = Query(0.0, ge=-1, le=1, description="Rates whose normalised autocorrelation peak is below this are blanked; 0 reproduces the paper"),
    max_gap_fraction: float = Query(farsense.MAX_GAP_FRACTION, gt=0, le=1, description="A window more than this fraction interpolated across dropouts reports nothing"),
    detail_t: float | None = Query(None, description="Return the window nearest this time in full: I/Q trajectory, patterns, autocorrelation"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM'"),
    source_mac: str | None = Query(None, description="Source MAC filter"),
    interpolate: bool = Query(True, description="Fill structural subcarrier nulls before transforming"),
) -> dict:
    """FarSense (Zeng et al. 2019) replayed over a capture range, as JSON.

    One entry per 12 s window: whether the target was stationary, the rate
    the paper's autocorrelation method reads, and the numbers it was read
    from. ``pattern`` is the best subcarrier's respiration pattern stitched
    across windows, on the sample grid, which is what the paper's GUI draws.
    See ``backend.farsense`` for what is copied and what is not.
    """
    p = resolve_capture_path(path)
    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = farsense.compute_farsense(
            p, t0, t1,
            mimo=mimo_filter,
            source_mac=parse_mac_filter(source_mac),
            interpolate=interpolate,
            detail_t=detail_t,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            band_rpm=(rpm_lo, rpm_hi),
            n_theta=n_theta,
            keep_fraction=keep_fraction,
            savgol_seconds=savgol_seconds,
            savgol_order=savgol_order,
            highpass_hz=highpass_hz,
            motion_frac_hi=motion_frac_hi,
            min_peak=min_peak,
            max_gap_fraction=max_gap_fraction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detail = result["detail"]
    detail_out = None
    if detail is not None:
        detail_out = {
            "index": detail["index"],
            "start_s": detail["start_s"],
            "t_s": [float(v) for v in detail["t_s"]],
            "best_sc": detail["best_sc"],
            "best_theta": detail["best_theta"],
            "iq": [[float(a), float(b)] for a, b in np.nan_to_num(detail["iq"])],
            "pattern": _nullable(detail["pattern"]),
            "acf": _nullable(detail["acf"]),
            "lag_lo": detail["lag_lo"],
            "lag_hi": detail["lag_hi"],
            "lag": float(detail["lag"]) if np.isfinite(detail["lag"]) else None,
            "sc_index": [int(v) for v in detail["sc_index"]],
            "bnr": _nullable(detail["bnr"]),
            "theta": _nullable(detail["theta"]),
            "selected": [bool(v) for v in detail["selected"]],
        }

    return {
        "time_s": [float(v) for v in result["time_s"]],
        "stationary": [bool(v) for v in result["stationary"]],
        "motion_level": _nullable(result["motion_level"]),
        "unknown": [bool(v) for v in result["unknown"]],
        "rpm": _nullable(result["rpm"]),
        "lag": _nullable(result["lag"]),
        "acf_peak": _nullable(result["acf_peak"]),
        "acf_peak_norm": _nullable(result["acf_peak_norm"]),
        "bnr_max": _nullable(result["bnr_max"]),
        "n_selected": [int(v) for v in result["n_selected"]],
        "best_sc": [int(v) for v in result["best_sc"]],
        "best_theta": _nullable(result["best_theta"]),
        "sc_index": [int(v) for v in result["sc_index"]],
        "bnr_map": _nullable_rows(result["bnr_map"]),
        # A pure tone scores win / fft_size, so this factor puts BNR on a
        # 0..1 scale that does not move with the sample rate.
        "bnr_norm_factor": float(result["params"]["fft_size"]) / float(result["win"]),
        "pattern_t": [float(v) for v in result["pattern_t"]],
        "pattern": _nullable(np.round(result["pattern"], 4)),
        "detail": detail_out,
        "win": result["win"],
        "hop": result["hop"],
        "window_seconds": result["window_seconds"],
        "fs_hz": result["fs_hz"],
        "lag_lo": result["lag_lo"],
        "lag_hi": result["lag_hi"],
        "params": result["params"],
        "frames_used": result["frames_used"],
        "frames_without_ratio": result["frames_without_ratio"],
        "t_min": result["t_min"],
        "t_max": result["t_max"],
    }


@app.get("/api/hybrid")
def hybrid_detector(   # not `hybrid`: that name is the module this calls
    path: str = Query(..., description="Path to capture file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    use_amplitude: bool = Query(False, description="Also count bursts on the raw amplitude frame-diff channel"),
    hold_s: float = Query(hybrid.HOLD_SECONDS, ge=0, le=600, description="Seconds presence is held after the last evidence"),
    burst_s: float = Query(hybrid.BURST_SECONDS, ge=1, le=60, description="Consecutive seconds above the motion threshold that make a burst"),
    motion_rel: float = Query(hybrid.MOTION_REL, ge=1, le=100, description="Ratio-motion threshold as a multiple of the range's own floor"),
    motion_abs: float = Query(hybrid.MOTION_ABS, ge=0, le=10, description="Ratio-motion threshold never below this |dr|/|r|"),
    amp_rel: float = Query(hybrid.AMP_REL, ge=1, le=100, description="Amplitude-motion threshold as a multiple of its floor"),
    amp_abs: float = Query(hybrid.AMP_ABS, ge=0, le=60, description="Amplitude-motion threshold never below this many dB"),
    floor_pct: float = Query(hybrid.FLOOR_PERCENTILE, ge=0, le=100, description="Percentile of the per-second level taken as the range's quiet floor"),
    breath_min_peak: float = Query(hybrid.BREATH_MIN_PEAK, ge=-1, le=1, description="Normalised FarSense peak a window needs to count as breathing"),
    breath_persist_s: float = Query(hybrid.BREATH_PERSIST_SECONDS, ge=1, le=120, description="Seconds of consecutive qualifying windows that must agree on the rate"),
    breath_rate_tol: float = Query(hybrid.BREATH_RATE_TOL, ge=0, le=60, description="How far the rates in that run may differ, rpm"),
    breath_window: float = Query(hybrid.BREATH_WINDOW_SECONDS, gt=0, le=120, description="FarSense window in seconds"),
    breath_highpass: float = Query(hybrid.BREATH_HIGHPASS_HZ, ge=0, le=5, description="High-pass before the FarSense sweep, Hz"),
    margin_s: float = Query(truthmod.DEFAULT_MARGIN_S, ge=0, le=60, description="Empty camera frames within this many seconds of a transition are not scored"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM'"),
    source_mac: str | None = Query(None, description="Source MAC filter"),
    interpolate: bool = Query(True, description="Fill structural subcarrier nulls before transforming"),
) -> dict:
    """The calibration-free motion/breathing detector, one verdict per second.

    See ``backend.hybrid``. Scored against the camera when the capture has a
    sidecar, through ``backend.truth`` with the same margin the other
    scorers use, and the confusion matrix travels with the series.
    """
    p = resolve_capture_path(path)
    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = hybrid.compute_hybrid(
            p, t0, t1,
            mimo=mimo_filter,
            source_mac=parse_mac_filter(source_mac),
            interpolate=interpolate,
            use_amplitude=use_amplitude,
            hold_seconds=hold_s,
            burst_seconds=burst_s,
            motion_rel=motion_rel,
            motion_abs=motion_abs,
            amp_rel=amp_rel,
            amp_abs=amp_abs,
            floor_percentile=floor_pct,
            breath_min_peak=breath_min_peak,
            breath_persist_seconds=breath_persist_s,
            breath_rate_tol=breath_rate_tol,
            breath_window_seconds=breath_window,
            breath_highpass_hz=breath_highpass,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    truth_out = None
    confusion_out = None
    cam = _camera_truth(p)
    if cam is not None and cam.size:
        cells, excluded = truthmod.cell_truth(
            result["time_s"], cam[:, 0], cam[:, 1] > 0.5, 0.5, margin_s
        )
        c = truthmod.confusion(cells, result["present"], excluded)
        scored = np.isfinite(cells)
        c["base_rate"] = float(np.mean(cells[scored] > 0.5)) if scored.any() else None
        c["margin_s"] = float(margin_s)
        confusion_out = c
        truth_out = {
            "time_s": [float(v) for v in cam[:, 0]],
            "present": [bool(v > 0.5) for v in cam[:, 1]],
        }

    def _nan(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    return {
        "time_s": [float(v) for v in result["time_s"]],
        "present": [bool(v) for v in result["present"]],
        "state": result["state"],
        "unknown": [bool(v) for v in result["unknown"]],
        "motion_ratio": _nullable(result["motion_ratio"]),
        "motion_amp": _nullable(result["motion_amp"]),
        "burst": [bool(v) for v in result["burst"]],
        "breathing": [bool(v) for v in result["breathing"]],
        "breath_peak": _nullable(result["breath_peak"]),
        "breath_rpm": _nullable(result["breath_rpm"]),
        "ratio_floor": _nan(result["ratio_floor"]),
        "ratio_threshold": _nan(result["ratio_threshold"]),
        "amp_floor": _nan(result["amp_floor"]),
        "amp_threshold": _nan(result["amp_threshold"]),
        "breath_note": result["breath_note"],
        "fs_hz": result["fs_hz"],
        "params": result["params"],
        "frames_used": result["frames_used"],
        "frames_without_ratio": result["frames_without_ratio"],
        "t_min": result["t_min"],
        "t_max": result["t_max"],
        "truth": truth_out,
        "confusion": confusion_out,
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _walk_captures(root: Path, depth: int, seen: set[Path]) -> Iterator[Path]:
    """Yield capture files under *root*, descending into subdirectories.

    Hand-rolled rather than ``rglob`` because ``rglob`` does not descend into
    symlinked directories on Python 3.12, and a symlinked directory is exactly
    how a large capture archive gets attached to ``captures/``.

    *seen* holds the real paths of directories already visited, so a symlink
    cycle terminates instead of recursing forever. Unreadable directories are
    skipped rather than failing the whole listing.
    """
    if depth < 0:
        return
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            is_dir = entry.is_dir()  # follows symlinks; False if broken
        except OSError:
            continue
        if is_dir:
            real = entry.resolve()
            if real in seen:
                continue
            seen.add(real)
            yield from _walk_captures(entry, depth - 1, seen)
        elif entry.suffix in CAPTURE_SUFFIXES and entry.is_file():
            yield entry


def _capture_conditions(capture: Path) -> dict:
    """room/configuration/scenario for a capture, as far as they are recorded.

    ``scenario`` falls back to what the camera saw, so the unattended runs sort
    into empty/partial/occupied instead of piling up as unknown. Absent rather
    than null when nothing is known, so a client can tell "not recorded" from
    a recorded blank.
    """
    out: dict = {}
    meta_path = capture.with_name(f"{capture.stem}_meta.json")
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        for key in ("room", "configuration", "scenario", "subject",
                    "activity", "facing", "distance_m"):
            if meta.get(key) not in (None, ""):
                out[key] = meta[key]
    if "scenario" not in out:
        cv_path = capture.with_name(f"{capture.stem}_cv.json")
        if cv_path.is_file():
            try:
                frac = json.loads(cv_path.read_text())["summary"]["fraction_occupied"]
            except (OSError, json.JSONDecodeError, KeyError):
                frac = None
            if frac is not None:
                out["scenario"] = ("empty" if frac == 0.0
                                   else "occupied" if frac > 0.5 else "partial")
                out["occupancy"] = float(frac)
    return out


@app.get("/api/captures")
def list_captures() -> list[dict]:
    """List capture files under the captures/ directory, recursively.

    Returns filename, path, size_bytes, and mtime for each capture, sorted by
    mtime descending (newest first). Missing dir → empty list.

    ``filename`` is the path relative to ``captures/``, so a nested capture
    reads ``2026-08/capture.dat`` and stays distinguishable from a same-named
    file in another subdirectory. A top-level capture is still a bare name.
    ``path`` remains absolute and is what the client sends back.

    ``.dat`` is FeitCSI, ``.bin`` is MediaTek. The extension only decides
    what to *list*; which parser runs is decided by sniffing the bytes in
    ``tiles.get_index``, so a misnamed file still reads correctly.
    """
    root = CAPTURES_DIR
    if not root.is_dir():
        return []

    files: list[dict] = []
    for entry in _walk_captures(root, MAX_CAPTURE_DEPTH, {root.resolve()}):
        try:
            st = entry.stat()
        except OSError:
            continue  # vanished or dangling between walk and stat
        files.append({
            "filename": entry.relative_to(root).as_posix(),
            "path": str(entry),
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
            # The conditions a capture was recorded under, for grouping the
            # picker. Read from the sidecar rather than from a directory
            # layout: scripts/build_dataset_tree.py can arrange the same
            # captures by any axes, and a capture belongs to several groupings
            # at once. Filing them on disk instead would also put a second copy
            # of every capture inside captures/, where the walk above would
            # list it twice and the reference pooling would pick it twice.
            **_capture_conditions(entry),
        })

    files.sort(key=lambda f: f["mtime"], reverse=True)
    return files


class _HashedStatic(StaticFiles):
    """Static files with cache headers that match how Vite names them.

    Everything under ``assets/`` is content-hashed, so a changed file is a
    changed URL and the old one can be cached forever. ``index.html`` is the
    one file whose URL never changes while its contents do -- it is what names
    the current bundle hash. Served without a Cache-Control, browsers apply
    heuristic caching to it and keep loading a stale bundle after a deploy,
    which shows up as a feature simply missing from the UI. Revalidate it.
    """

    def file_response(self, *args, **kwargs) -> Response:
        resp = super().file_response(*args, **kwargs)
        path = args[0] if args else kwargs.get("full_path", "")
        if "/assets/" in str(path).replace("\\", "/"):
            resp.headers["cache-control"] = "public, max-age=31536000, immutable"
        else:
            resp.headers["cache-control"] = "no-cache"
        return resp


# Serve built frontend (production). In dev, Vite runs separately on :5173.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", _HashedStatic(directory=str(_dist), html=True), name="frontend")
