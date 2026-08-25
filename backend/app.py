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

from .stream import get_stream
from .tiles import TILE_METRICS, compute_tile, get_index, reset_tile_caches
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


# Serve built frontend (production). In dev, Vite runs separately on :5173.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
