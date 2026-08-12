"""FastAPI app serving FeitCSI parsed readings as JSON and binary tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .stream import get_stream
from .tiles import compute_tile, get_index, reset_tile_caches

DEFAULT_PATH = "captures/capture.dat"
DEFAULT_WINDOW = 200

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
        "X-Tile-VMin",
        "X-Tile-VMax",
    ],
)


def resolve_capture_path(path: str) -> Path:
    """Validate and resolve a capture file path.

    This is the single chokepoint for all filesystem access from the API.
    Currently permissive — it accepts any existing regular file, preserving
    backwards compatibility for callers passing absolute paths.  Any future
    restriction (e.g. confining to a captures/ directory, allow-listing
    extensions) should be added here, not duplicated across handlers.

    Rejects the empty string and anything that is not an existing regular
    file.
    """
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="path parameter is required")
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return p.resolve()


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
        "amp_min": float(np.nanmin(amp_finite)) if amp_finite.size else 0.0,
        "amp_max": float(np.nanmax(amp_finite)) if amp_finite.size else 1.0,
        "phase_min": float(np.nanmin(phase_finite)) if phase_finite.size else -np.pi,
        "phase_max": float(np.nanmax(phase_finite)) if phase_finite.size else np.pi,
    }


@app.get("/api/meta")
def meta(path: str = Query(..., description="Path to .dat file")) -> dict:
    """Cheap metadata endpoint — index only, never decodes payloads.

    Returns capture geometry and time range.  On a 211 MB capture this returns
    in well under a second because it only builds a FrameIndex.
    """
    p = resolve_capture_path(path)
    idx = get_index(p)

    t_min = float(idx.times[0]) if idx.count > 0 else 0.0
    t_max = float(idx.times[-1]) if idx.count > 0 else 0.0

    return {
        "filename": p.name,
        "chipset": "Intel AX2xx",
        "bandwidth": idx.bandwidth,
        "num_subcarriers": idx.num_subcarriers,
        "total_frames": idx.count,
        "t_min": t_min,
        "t_max": t_max,
        "num_rx": idx.num_rx,
        "num_tx": idx.num_tx,
    }


@app.get("/api/tile")
def tile(
    path: str = Query(..., description="Path to .dat file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    width: int = Query(1600, ge=1, description="Output columns (client plot width in pixels; capped at 4096)"),
    metric: str = Query("amplitude", description="Metric: 'amplitude' or 'phase'"),
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
    if metric not in ("amplitude", "phase"):
        raise HTTPException(status_code=400, detail="metric must be 'amplitude' or 'phase'")

    p = resolve_capture_path(path)

    grid, meta = compute_tile(p, t0, t1, width, metric)

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
            "X-Tile-VMin": str(meta["vmin"]),
            "X-Tile-VMax": str(meta["vmax"]),
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Serve built frontend (production). In dev, Vite runs separately on :5173.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
