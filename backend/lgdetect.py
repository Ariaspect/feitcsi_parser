"""Run the LG on-board detector over a capture, under the board's interpreter.

The detector is a live tool: it arms the radio, reads /proc/net/wlan/csi_data
and prints "+"/"-" as it goes. Replaying it over a file is handled by
``scripts/lg_detect_replay.py``, which imports and calls its functions rather
than reimplementing them. This module's job is only to run that script under
the RIGHT INTERPRETER and cache the answer.

The interpreter matters more than it looks. The detector's TLV length
arithmetic shifts a numpy uint8 left by 8. Under NumPy 1.x that promotes to
int and gives 512 for the CSI tags; under NumPy 2.x (NEP 50) it stays uint8,
evaluates to 0, and the TLV walk desynchronises at the first CSI field --
silently, yielding frames with zeroed imaginary parts rather than an error. The
board runs NumPy 1.26.4, so a replay in the project venv would be measuring the
NumPy version rather than the detector. Hence a second venv, and hence a
subprocess: the two NumPys cannot share a process.

Runs are cached on disk. A 300 s capture is ~22600 records and the detector
loops over all 256 subcarriers per record in pure Python, twice, so a run costs
minutes -- too slow to repeat per page load, and entirely deterministic, so
repeating it buys nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BOARD_PYTHON = Path(os.environ.get("LG_BOARD_PYTHON", REPO / ".venv-board" / "bin" / "python"))
REPLAY = REPO / "scripts" / "lg_detect_replay.py"
CACHE_DIR = Path(os.environ.get("LG_DETECT_CACHE", REPO / ".cache" / "lgdetect"))

# A capture this long would take the pure-Python detector past any sensible
# request. Measured: ~22600 records of a 300 s capture run in about a minute.
TIMEOUT_SECONDS = int(os.environ.get("LG_DETECT_TIMEOUT", "900"))


class BoardEnvMissing(RuntimeError):
    """The NumPy 1.x interpreter the detector needs is not installed."""


def _key(path: Path, threshold: float, absence: float) -> Path:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{threshold}|{absence}"
    return CACHE_DIR / (hashlib.sha256(raw.encode()).hexdigest()[:32] + ".json")


def available() -> bool:
    return BOARD_PYTHON.exists() and REPLAY.exists()


def run(path: Path, *, threshold: float = 26.0, absence: float = 10.0) -> dict[str, Any]:
    """Replay the detector over *path*, from cache when it has been run before."""
    if not available():
        raise BoardEnvMissing(
            f"{BOARD_PYTHON} not found. The detector needs NumPy 1.x -- create it with:\n"
            f"  uv venv --python 3.12 .venv-board && "
            f"VIRTUAL_ENV=.venv-board uv pip install numpy==1.26.4"
        )

    cache = _key(path, threshold, absence)
    if cache.exists():
        try:
            out = json.loads(cache.read_text())
            out["cached"] = True
            return out
        except (OSError, ValueError):
            pass  # a corrupt cache is not a reason to fail; re-run it

    proc = subprocess.run(
        [str(BOARD_PYTHON), str(REPLAY), str(path),
         "--threshold", str(threshold), "--absence", str(absence)],
        capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"replay failed: {proc.stderr.strip()[:400]}")
    out = json.loads(proc.stdout)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cache.write_text(json.dumps(out))
    except OSError:
        pass  # an unwritable cache costs time, not correctness
    out["cached"] = False
    return out


def intervals(events: list[dict], t_max: float) -> list[tuple[float, float]]:
    """Its output is a state machine: '+' says present, '-' says absent.

    An unterminated '+' runs to the end of the capture, which is what the live
    detector would have been reporting when the recording stopped.
    """
    out: list[tuple[float, float]] = []
    start: float | None = None
    for ev in events:
        if ev["kind"] == "+" and start is None:
            start = float(ev["t"])
        elif ev["kind"] == "-" and start is not None:
            out.append((start, float(ev["t"])))
            start = None
    if start is not None:
        out.append((start, t_max))
    return out
