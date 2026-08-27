"""FastAPI app serving FeitCSI parsed readings as JSON and binary tiles."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .presence import CHANNELS
from .stream import get_stream
from .tiles import (
    DOPPLER_METRICS,
    TILE_METRICS,
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
    ref_t0: float | None = Query(None, description="Start of a known-empty reference range (seconds); required with ref_t1"),
    ref_t1: float | None = Query(None, description="End of a known-empty reference range (seconds); required with ref_t0"),
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
