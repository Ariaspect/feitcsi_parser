"""FastAPI app serving FeitCSI parsed readings as JSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .parser import load_capture, tail_window


DEFAULT_PATH = "captures/capture.dat"
DEFAULT_WINDOW = 200

app = FastAPI(title="FeitCSI Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/snapshot")
def snapshot(
    path: str = Query(DEFAULT_PATH, description="Path to .dat file"),
    max_packets: int = Query(DEFAULT_WINDOW, ge=1, le=10000),
) -> dict:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        capture = load_capture(p)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse error: {exc}") from exc

    window = tail_window(capture, max_packets=max_packets)

    amp_finite = window.amplitude[np.isfinite(window.amplitude)]
    phase_finite = window.phase[np.isfinite(window.phase)]

    return {
        "filename": window.filename,
        "chipset": window.chipset,
        "bandwidth": window.bandwidth,
        "num_subcarriers": window.num_subcarriers,
        "total_packets": len(capture),
        "window_packets": len(window),
        "time_seconds": window.time_seconds.tolist(),
        "amplitude": window.amplitude.tolist(),
        "phase": window.phase.tolist(),
        "amp_min": float(np.nanmin(amp_finite)) if amp_finite.size else 0.0,
        "amp_max": float(np.nanmax(amp_finite)) if amp_finite.size else 1.0,
        "phase_min": float(np.nanmin(phase_finite)) if phase_finite.size else -np.pi,
        "phase_max": float(np.nanmax(phase_finite)) if phase_finite.size else np.pi,
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Serve built frontend (production). In dev, Vite runs separately on :5173.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
