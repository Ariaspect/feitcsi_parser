"""Laptop side of the always-on board run (backend/stream_run.py, user 2026-11-17): pulls the closed CSI segments
and the judgment log from the board, keeps the raw stream as capture files, one per hour.

    python scripts/stream_pull.py --dest ~/csi_stream [--board root@192.168.50.80] [--every 10]

Each closed segment (``seg_NNNNN.bin`` in the board's ``--segs``) is appended to ``<dest>/<YYYYMMDD_HHMMSS>.bin``
and then deleted on the board; segments end at frame-group boundaries, so every file is a valid capture and their
concatenation is the byte-exact stream. A new file starts at the first segment after the hour turns. The board's
``judgments.jsonl`` is mirrored to ``<dest>/judgments.jsonl`` (appended from the last byte pulled). One ssh
connection is reused (ControlMaster).
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--board", default="root@192.168.50.80")
    ap.add_argument("--segs", default="/tmp/csi-stream")
    ap.add_argument("--out", default="/var/csi/stream/out")
    ap.add_argument("--every", type=float, default=10.0)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    sock = Path.home() / ".ssh" / "cm-stream-pull-%C"       # a socket path must stay under ~100 bytes
    ssh = ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ControlMaster=auto",
           "-o", f"ControlPath={sock}", "-o", "ControlPersist=120", a.board]
    jl = a.dest / "judgments.jsonl"
    cur: Path | None = None
    cur_hour = None
    while True:
        try:
            ls = subprocess.run(ssh + [f"ls {a.segs}/seg_*.bin 2>/dev/null; wc -c < {a.out}/judgments.jsonl 2>/dev/null"],
                                capture_output=True, text=True, timeout=30).stdout.split()
            segs = sorted(x for x in ls if x.endswith(".bin"))
            size = int(ls[-1]) if ls and ls[-1].isdigit() else 0
            for s in segs:
                now = dt.datetime.now()
                hour = now.strftime("%Y%m%d%H")
                if cur is None or hour != cur_hour:
                    cur, cur_hour = a.dest / f"{now:%Y%m%d_%H%M%S}.bin", hour
                data = subprocess.run(ssh + [f"cat {s}"], capture_output=True, timeout=120).stdout
                bsz = subprocess.run(ssh + [f"wc -c < {s}"], capture_output=True, text=True, timeout=30).stdout.strip()
                if not bsz.isdigit() or int(bsz) != len(data):
                    print(f"{now:%H:%M:%S} short read of {s} ({len(data)} of {bsz}); retry next round", flush=True)
                    break
                with cur.open("ab") as fh:
                    fh.write(data)
                subprocess.run(ssh + [f"rm -f {s}"], timeout=30)
                print(f"{now:%H:%M:%S} {s.rsplit('/', 1)[-1]} {len(data)/1e6:.1f} MB -> {cur.name}", flush=True)
            have = jl.stat().st_size if jl.exists() else 0
            if size < have:                                # the board's log restarted shorter: mirror it afresh
                jl.rename(jl.with_suffix(f".{dt.datetime.now():%H%M%S}.jsonl")); have = 0
            if size > have:
                add = subprocess.run(ssh + [f"tail -c +{have + 1} {a.out}/judgments.jsonl | head -c {size - have}"],
                                     capture_output=True, timeout=60).stdout
                with jl.open("ab") as fh:
                    fh.write(add)
        except subprocess.TimeoutExpired:
            print(f"{dt.datetime.now():%H:%M:%S} ssh timeout; retrying", flush=True)
        except Exception as e:                             # keep pulling whatever went wrong this round
            print(f"{dt.datetime.now():%H:%M:%S} {type(e).__name__}: {e}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
