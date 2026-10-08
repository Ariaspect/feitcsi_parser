"""Hybrid 2's one-minute verdict as a standalone calculator: capture in, 1/0 out.

    python3 -m backend.hybrid2_calc CAPTURE.bin            # prints 1 or 0
    python3 -m backend.hybrid2_calc CAPTURE.bin --json     # and why

Built to run on the LG board, whose interpreter is a 32-bit ARM Python 3.12
with no scipy, no CSIKit, and NumPy 1.26.4 -- the board's own, which this
runs on (``scripts/build_board_bundle.sh`` ships code only). It imports NumPy
and the pure backend modules alone: mtk, index, presence, farsense,
framediff, hybrid, hybrid2, doppler. None of them reaches scipy or CSIKit on
this path.

It is the same computation as ``hybrid2.compute_hybrid2(...)["range_verdict"]``
with the tab's settings, not a re-implementation of it. Everything that
decides the answer is the backend's own function, called with the same
arguments: the frame selection, the frame step, the per-second median, the
FarSense sweep, the range rule. What lives here is only the glue the server
routes through ``backend.tiles`` -- which imports half the backend -- reduced
to the calls that matter for an MTK capture: decode whole 4096-frame blocks
(the structural-null fill reads every frame in a block, so decoding only the
selected frames would differ), rebuild the complex ratio, and resample it onto
the uniform grid FarSense needs. ``tests/test_hybrid2_calc.py`` holds it equal
to the server's verdict.

The rule (``hybrid2.RANGE_*``): present when the 90th percentile of the
per-second ratio-complex frame step at a 2 s lag exceeds 0.035; else present
when the FarSense peak stays at or above 0.25 for 5 consecutive windows; else
empty.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend import doppler, framediff, hybrid, hybrid2, mtk, presence

# Must equal backend.tiles.BLOCK_SIZE: the null fill is decided per block, so
# the block boundaries are part of the result (asserted in the tests).
BLOCK_SIZE = 4096

# The tab's frame set: the AP's two transmit chains as one receive chain sees
# them. framediff falls back to the capture's most common mode when there is
# no 2x1, exactly as it does on the server.
MIMO = (2, 1)

# The oldest NumPy the calculators were checked on: the board's own.
MIN_NUMPY = (1, 26)


def _ratio_rows(path: Path, index: mtk.MTKIndex, frame_ids: np.ndarray,
                blocks: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The ratio's dB and phase planes for *frame_ids*, decoded block by block.

    *blocks* caches each block's two planes so the motion and breathing paths,
    which want overlapping frame sets, decode the file once.
    """
    amp_rows, phase_rows = [], []
    for b in sorted({int(i) // BLOCK_SIZE for i in frame_ids}):
        start = b * BLOCK_SIZE
        if b not in blocks:
            n_block = min(BLOCK_SIZE, index.count - start)
            _, _, ratio_db, ratio_phase = mtk.decode_frames(
                path, index, np.arange(start, start + n_block), interpolate=True,
            )
            blocks[b] = (ratio_db, ratio_phase)
        ratio_db, ratio_phase = blocks[b]
        wanted = frame_ids[(frame_ids >= start) & (frame_ids < start + BLOCK_SIZE)]
        amp_rows.append(ratio_db[wanted - start])
        phase_rows.append(ratio_phase[wanted - start])
    return np.concatenate(amp_rows), np.concatenate(phase_rows)


def _breath_grid(times_all: np.ndarray, frame_ids: np.ndarray, ratio: np.ndarray,
                 mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """``tiles._presence_grid``'s resampling: the uniform complex grid FarSense reads.

    Frames carrying no ratio at all are dropped first, so the hole they leave
    is measured as a gap. The rate comes from every frame of the filter, the
    grid spans the surviving ones. Returns ``(grid, fabricated, fs, origin)``.
    """
    usable = np.isfinite(ratio).any(axis=1)
    times = times_all[frame_ids][usable]
    ratio = ratio[usable]
    if times.size < 2:
        raise ValueError("no frames in this capture carry a two-antenna CSI ratio")
    filtered = times_all[mask]
    _, fs = doppler.uniform_grid(filtered if filtered.size >= 2 else times)
    step = 1.0 / fs
    grid_times = times[0] + np.arange(int(np.floor((times[-1] - times[0]) / step)) + 1) * step
    gap_limit = doppler.gap_limit_for(times)
    real, fabricated = doppler.resample_uniform(times, ratio.real, grid_times, gap_limit)
    imag, _ = doppler.resample_uniform(times, ratio.imag, grid_times, gap_limit)
    return real + 1j * imag, fabricated, fs, float(grid_times[0])


def evaluate(path: str | Path, *, source_mac: str | None = None) -> dict[str, Any]:
    """Score one capture. ``label`` is 1 for present, 0 for empty.

    *source_mac* defaults to the capture's dominant transmitter -- the AP, on
    every capture so far. Raises ``ValueError`` when the capture cannot be
    scored at all (too short, no ratio, no frames from the transmitter).
    """
    t_start = time.perf_counter()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such capture: {path}")
    if not mtk.can_read(path):
        raise ValueError(f"{path.name} is not a MediaTek capture")
    index = mtk.MTKIndex(path)
    times_all = np.asarray(index.times, dtype=float)
    if times_all.size < 2:
        raise ValueError("fewer than 2 frames in the capture")
    blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Motion: framediff.capture_steps over the whole capture.
    sel = framediff._uniform_selection(index, MIMO, source_mac)
    mac = sel["source_mac"]
    ids = np.flatnonzero(sel["mask"])
    if ids.size < 2:
        raise ValueError(f"fewer than 2 frames once the set was made uniform ({sel['note']})")
    db, phase = _ratio_rows(path, index, ids, blocks)
    series = presence.complex_ratio(db, phase)
    t_sel = times_all[ids]
    lag = 1
    step_dt = float(np.median(np.diff(t_sel)))
    if step_dt > 0:
        lag = max(1, int(round(hybrid2.RANGE_LAG_SECONDS / step_dt)))
    if lag >= ids.size:
        raise ValueError(f"a {hybrid2.RANGE_LAG_SECONDS:g} s lag is {lag} frames; "
                         f"the capture holds {ids.size}")
    rssi = np.asarray(getattr(index, "rssi_1", None))
    gain_state = rssi[ids] if rssi.ndim == 1 and rssi.size > int(ids[-1]) else None
    steps = framediff.frame_steps(series, t_sel, complex_input=True, lag=lag,
                                  gain_state=gain_state, gate_gain=False)

    # Breathing: hybrid.capture_evidence -- the same frame filter, MAC fixed to
    # the one the motion path chose, so the two channels read one transmitter.
    mode = tuple(sel["mimo"]) if sel["mimo"] is not None else None
    mask = index.filter_mask(mimo=mode, source_mac=mac)
    bids = np.flatnonzero(mask)
    if bids.size < 2:
        raise ValueError("fewer than 2 frames for the breathing channel")
    db, phase = _ratio_rows(path, index, bids, blocks)
    grid, fabricated, fs, origin = _breath_grid(
        times_all, bids, presence.complex_ratio(db, phase), mask)
    ev = hybrid.evidence_series(grid, fs, fabricated=fabricated)

    # hybrid2.capture_evidence: the step folded per second on FarSense's clock,
    # blanked where the grid is mostly invented.
    seconds = origin + np.asarray(ev["time_s"], dtype=float) - 0.5
    level = hybrid.per_second(np.asarray(steps["magnitude"], dtype=float),
                              np.asarray(steps["time_s"], dtype=float), seconds)
    level[np.asarray(ev["unknown"], dtype=bool)] = np.nan
    rv = hybrid2.range_verdict(level, ev["breath_peak"])

    return {
        "file": path.name,
        "label": int(rv["present"]),
        "present": bool(rv["present"]),
        "by": rv["by"],
        "motion_p90": None if not np.isfinite(rv["motion_p90"]) else float(rv["motion_p90"]),
        "breath_run": int(rv["breath_run"]),
        "seconds": int(rv["seconds"]),
        "thresholds": rv["thresholds"],
        "source_mac": mac,
        "selection": sel["note"],
        "frames": int(ids.size),
        "fs_hz": float(fs),
        "lag_frames": int(lag),
        "breath_note": ev.get("breath_note"),
        "numpy": np.__version__,
        "elapsed_s": round(time.perf_counter() - t_start, 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hybrid2_calc",
        description="Hybrid 2 one-minute verdict: prints 1 (present) or 0 (empty).")
    ap.add_argument("capture", type=Path, help="MediaTek .bin capture, about one minute long")
    ap.add_argument("--mac", default=None, help="transmitter MAC (default: the dominant one)")
    ap.add_argument("--json", action="store_true", help="print the evidence as JSON")
    args = ap.parse_args(argv)

    # Checked on NumPy 2.5.2 (x86-64, the server) and on the board's own
    # 1.26.4 (32-bit ARM), which agree to the verdict. Older than that is
    # untested -- NumPy's integer promotion has moved between versions, and the
    # decoder does uint arithmetic -- so it stops rather than answers.
    if tuple(int(v) for v in np.__version__.split(".")[:2]) < MIN_NUMPY:
        print(f"hybrid2_calc: needs NumPy >= {MIN_NUMPY[0]}.{MIN_NUMPY[1]}, got "
              f"{np.__version__} from {Path(np.__file__).parent}", file=sys.stderr)
        return 3
    try:
        out = evaluate(args.capture, source_mac=args.mac)
    except (OSError, ValueError) as exc:
        print(f"hybrid2_calc: {args.capture}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out) if args.json else out["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
