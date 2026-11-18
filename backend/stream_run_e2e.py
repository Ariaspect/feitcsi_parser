"""Always-on capture + an e2e judgment every 5 s, on the board (user 2026-11-18: deploy the e2e model; the same form
as the 15F streaming test, ``stream_run``).

    python3 -m backend.stream_run_e2e --out /var/csi/stream_e2e [--fold env2] [--cores 3] (--install 5 | --ref REF.npz)

* ``/var/csi/csi_stream.sh`` streams CSI to this process; ``Spool`` (a copy of the 15F stream test's) copies it into
  segment files in ``--segs`` (tmpfs), byte-exact, for the laptop to pull; the oldest beyond ``--keep`` are deleted.
* Each new frame is decoded once into ``stream_e2e.Ring``; every ``HOP_S`` the last 61 s is judged
  (``stream_e2e.window_input`` + ``Model.logit``) and one JSON line is appended to ``<out>/judgments.jsonl``:
  logit, label (person if logit > 0), the window's first stamp / frame count and the stage times.
* Calibration: ``--install N`` takes the first N non-overlapping complete minutes as the room's empty reference (the
  room must be empty then), saves the per-bin scale to ``<out>/reference.npz`` and goes on judging with it; ``--ref``
  loads a saved one. Until a reference exists, lines carry no logit.
* ``--cores 3,2,1`` spreads each judgment over those cores (workers forked once, ``e2e_np.Pool``); default one core.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from backend import mtk, stream_e2e as SE


class Spool:
    """The stream's bytes in segment files: ``seg_NNNNN.part`` while written, renamed ``.bin`` once closed."""

    def __init__(self, src, directory: Path, keep: int) -> None:
        self.src = src
        self.dir = directory
        self.keep = keep
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in self.dir.glob("seg_*"):
            p.unlink()
        self.n = 0
        self.lock = threading.Lock()
        self.path = self._open()
        self.bytes = 0
        self.eof = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self) -> Path:
        p = self.dir / f"seg_{self.n:05d}.part"
        self.fh = open(p, "wb", buffering=0)
        return p

    def _run(self) -> None:
        fd = self.src.fileno()
        while True:
            b = os.read(fd, 1 << 16)
            if not b:
                break
            with self.lock:
                self.fh.write(b)
                self.bytes += len(b)
        self.eof = True

    def roll(self, cut: int) -> Path:
        """Close the current segment at byte ``cut`` (a frame-group end); the rest opens the next one."""
        with self.lock:
            old = self.path
            self.fh.close()
            with open(old, "r+b") as fh:
                fh.seek(cut)
                tail = fh.read()
                fh.truncate(cut)
            done = old.with_suffix(".bin")
            old.rename(done)
            self.n += 1
            self.path = self._open()
            self.fh.write(tail)
        closed = sorted(self.dir.glob("seg_*.bin"))
        for p in closed[:-self.keep] if len(closed) > self.keep else []:
            p.unlink(missing_ok=True)
        return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--segs", type=Path, default=Path("/tmp/csi-stream"))
    ap.add_argument("--fold", default="env2")
    ap.add_argument("--cores", default="3")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--install", type=int, default=0)
    g.add_argument("--ref", type=Path)
    ap.add_argument("--secs", type=int, default=24 * 3600)
    ap.add_argument("--ival", default="0.02")
    ap.add_argument("--seg-mb", type=float, default=1.0)
    ap.add_argument("--keep", type=int, default=120)
    ap.add_argument("--script", default="/var/csi/csi_stream.sh")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cores = [int(c) for c in a.cores.split(",") if c != ""]
    if cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cores[0]})
    model = SE.Model(a.fold, cores)
    log = open(a.out / "judgments.jsonl", "a", buffering=1)
    err = open(a.out / "stream.stderr", "a", buffering=1)

    scale = None
    if a.ref:
        z = np.load(a.ref, allow_pickle=False)
        if str(z["model"]) != SE.META["model"]:
            raise SystemExit(f"reference is for {z['model']}, not {SE.META['model']}")
        scale = np.asarray(z["scale"], np.float32)
    install: list[np.ndarray] = []
    install_first: list[int] = []
    last_install_end = None

    proc = subprocess.Popen([a.script, str(a.secs), a.ival], stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL)
    stop = [False]
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: stop.__setitem__(0, True))
    spool = Spool(proc.stdout, a.segs, a.keep)
    ring = SE.Ring()
    idx: mtk.MTKIndex | None = None
    idx_path: Path | None = None
    fed = 0
    seg_bytes = int(a.seg_mb * 1e6)
    next_t = time.monotonic() + SE.HOP_S
    print(json.dumps({"event": "start", "wall": time.time(), "model": SE.META["model"], "fold": a.fold,
                      "cores": cores, "install": a.install, "ref": str(a.ref) if a.ref else None,
                      "pid": os.getpid()}), file=log)
    try:
        while not stop[0] and not (spool.eof and proc.poll() is not None):
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t += SE.HOP_S
            tm: dict[str, float] = {}
            t0 = time.perf_counter()
            with spool.lock:
                cur = spool.path
            if cur != idx_path:
                idx, idx_path, fed = None, cur, 0
            if cur.stat().st_size == 0:
                continue
            if idx is None:
                idx = mtk.MTKIndex(cur)
            else:
                idx.extend()
            new = np.arange(fed, idx.count)
            ring.ingest(cur, idx, new)
            fed = idx.count
            arrived = time.time()
            if cur.stat().st_size >= seg_bytes and idx.count:
                spool.roll(int(idx._scan_end))
            tm["ingest"] = time.perf_counter() - t0
            out = SE.window_input(ring, None, tm)
            row = {"wall": arrived, "frames_new": int(new.size), "ring_mb": round(ring.nbytes() / 1e6, 1),
                   "stream_mb": round(spool.bytes / 1e6, 1), "seg": spool.n}
            if out is not None:
                arr, w = out
                row.update(first_stamp=w["first_stamp"], last_stamp=ring.last_stamp(), frames=w["frames"])
                if scale is None and a.install and len(install) < a.install and (
                        last_install_end is None or w["first_stamp"] >= last_install_end):
                    install.append(arr[:, SE.NEED]); install_first.append(w["first_stamp"])
                    last_install_end = w["first_stamp"] + int(SE.WINDOW_S * 1000)
                    row["install_minute"] = len(install)
                    if len(install) == a.install:
                        full = []
                        for v in install:
                            q = np.zeros((SE.T, SE.N.N_BINS, 2), np.float16); q[:, SE.NEED] = v; full.append(q)
                        scale = SE.scale_from(full)
                        np.savez(a.out / "reference.npz", model=SE.META["model"], scale=scale,
                                 first_stamps=np.array(install_first), saved=time.time())
                        row["reference_saved"] = str(a.out / "reference.npz")
                if scale is not None:
                    tj = time.perf_counter()
                    lg = model.logit(arr, scale)
                    tm["judge"] = time.perf_counter() - tj
                    row.update(logit=round(lg, 5), label="person" if lg > 0 else "empty")
            row["t"] = {k: round(v, 4) for k, v in tm.items()}
            row["t_total"] = round(time.perf_counter() - t0, 4)
            print(json.dumps(row), file=log)
            if time.monotonic() > next_t:
                next_t = time.monotonic() + SE.HOP_S
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        model.close()
        print(json.dumps({"event": "stop", "wall": time.time(), "rc": proc.poll()}), file=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
