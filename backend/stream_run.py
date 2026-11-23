"""Always-on capture + a judgment every 5 s, on the board (user 2026-11-17, full deployment test).

    python3 -m backend.stream_run --out /var/csi/stream [--model seq15pwu] [--fold scnC] (--install 5 | --ref REF.npz)

* ``/var/csi/csi_stream.sh`` (the capture script's board side: lock, stimulus, driver arming) streams CSI to this
  process; a reader thread copies it into segment files in ``--segs`` (tmpfs), so the pipe never backs up while a
  judgment runs. A segment is closed at the end of the last complete frame group once it passes ``--seg-mb``: the
  closed segments, concatenated, are the byte-exact stream (the laptop pulls them and deletes them; the oldest beyond
  ``--keep`` are deleted here regardless).
* Each new frame is decoded once into ``stream_phys.FrameRing``; every ``HOP_S`` the last ``WINDOW_S`` is judged
  (``stream_phys.series_at`` + ``judge``) and one JSON line is appended to ``<out>/judgments.jsonl``.
* Calibration: ``--install N`` takes the first N non-overlapping complete minutes as the room's empty reference
  (the room must be empty then), saves them to ``<out>/reference.npz`` and goes on judging with it; ``--ref`` loads a
  saved one. Until a reference exists, lines carry the raw series' summary but no P.
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

from backend import mtk, stream_phys as SP


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
    ap.add_argument("--model", default="seq15pwu")
    ap.add_argument("--fold", default="scnC")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--install", type=int, default=0)
    g.add_argument("--ref", type=Path)
    ap.add_argument("--secs", type=int, default=24 * 3600)
    ap.add_argument("--ival", default="0.02")
    ap.add_argument("--seg-mb", type=float, default=1.0)    # ~5.5 s: the laptop sees the stream that late
    ap.add_argument("--keep", type=int, default=120)       # ~11 min in tmpfs if the laptop stops pulling
    ap.add_argument("--script", default="/var/csi/csi_stream.sh")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "judgments.jsonl", "a", buffering=1)
    err = open(a.out / "stream.stderr", "a", buffering=1)

    ref = None
    if a.ref:
        z = np.load(a.ref, allow_pickle=False)
        if str(z["model"]) != a.model:
            raise SystemExit(f"reference is for {z['model']}, not {a.model}")
        ref = SP.Reference(a.model, list(z["minutes"]))
    install: list[np.ndarray] = []
    install_first: list[int] = []

    proc = subprocess.Popen([a.script, str(a.secs), a.ival], stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL)
    stop = [False]
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: stop.__setitem__(0, True))
    spool = Spool(proc.stdout, a.segs, a.keep)
    ring = SP.FrameRing()
    idx: mtk.MTKIndex | None = None
    idx_path: Path | None = None
    fed = 0
    seg_bytes = int(a.seg_mb * 1e6)
    next_t = time.monotonic() + SP.HOP_S
    last_install_end = None
    print(json.dumps({"event": "start", "wall": time.time(), "model": a.model, "fold": a.fold,
                      "install": a.install, "ref": str(a.ref) if a.ref else None, "pid": os.getpid()}), file=log)
    try:
        while not stop[0] and not (spool.eof and proc.poll() is not None):
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t += SP.HOP_S
            tm: dict[str, float] = {}
            t0 = time.perf_counter()
            # ingest what arrived: the index of the open segment, extended
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
            out = SP.series_at(ring, None, a.model, tm)
            row = {"wall": arrived, "frames_new": int(new.size), "ring_mb": round(ring.nbytes() / 1e6, 1),
                   "stream_mb": round(spool.bytes / 1e6, 1), "seg": spool.n}
            if out is not None:
                X, s = out
                row.update(first_stamp=s["first_stamp"], last_stamp=ring.last_stamp(), frames=s["frames"],
                           T=round(float(s["T"]), 3), w=round(float(s["w"]), 4))
                if ref is None and a.install and len(install) < a.install and (
                        last_install_end is None or s["first_stamp"] >= last_install_end):
                    install.append(X); install_first.append(s["first_stamp"])
                    last_install_end = s["first_stamp"] + int(SP.WINDOW_S * 1000)
                    row["install_minute"] = len(install)
                    if len(install) == a.install:
                        np.savez(a.out / "reference.npz", model=a.model, minutes=np.stack(install),
                                 first_stamps=np.array(install_first), saved=time.time())
                        ref = SP.Reference(a.model, install)
                        row["reference_saved"] = str(a.out / "reference.npz")
                if ref is not None:
                    tj = time.perf_counter()
                    v = SP.judge(X, ref, a.fold)
                    tm["judge"] = time.perf_counter() - tj
                    row.update(p=round(v["p"], 5), label=v["label"], p_seeds=[round(q, 5) for q in v["p_seeds"]])
                row["X"] = np.where(np.isfinite(X), X, np.nan).round(6).tolist() if a.install and ref is None else None
            row["t"] = {k: round(v, 4) for k, v in tm.items()}
            row["t_total"] = round(time.perf_counter() - t0, 4)
            print(json.dumps(row, allow_nan=True), file=log)
            if time.monotonic() > next_t:              # fell behind: skip to the next hop rather than pile up
                next_t = time.monotonic() + SP.HOP_S
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(json.dumps({"event": "stop", "wall": time.time(), "rc": proc.poll()}), file=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
