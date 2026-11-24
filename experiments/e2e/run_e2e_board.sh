#!/usr/bin/env bash
# Board timing of the end-to-end picks (bench_e2e_board.py) on the 8 windows the Physics models were timed on.
#   experiments/e2e/run_e2e_board.sh BUNDLE_DIR [REPS] [BOARD]      (from the laptop; default root@192.168.50.80)
# Copies backend/ (wt-ml, ef309ba) + e2e_np.py + the bench + the bundle + the captures to the board's /tmp/cy_e2e
# (legacy scp -O), runs it pinned to core 3, fetches the result and removes /tmp/cy_e2e. Never touches the console.
set -euo pipefail
BUNDLE=$1; REPS=${2:-1}; BOARD=${3:-root@192.168.50.80}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/cyphy/feitcsi_parser-wt-ml
CAPS=/home/cyphy/feitcsi_parser/captures
OUT=${OUT:-$HERE/out/board}
STEMS="20261001_151502 20261002_040002 20261013_120426 20261021_142736 20261023_111259 20261009_112535 20261013_164146 20261021_164727"
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cy_e2e/caps"
tar -c -C "$ROOT" --exclude=__pycache__ backend | tar -x -C "$TMP/cy_e2e"
cp "$HERE/e2e_np.py" "$HERE/bench_e2e_board.py" "$TMP/cy_e2e/"
cp -r "$BUNDLE" "$TMP/cy_e2e/bundle"
for s in $STEMS; do
  f=""
  for c in "$CAPS/$s.bin" "$CAPS/lg_archive/${s:0:8}/$s.bin"; do
    [ -f "$c" ] && { f=$c; break; }
  done
  [ -n "$f" ] || { echo "capture $s not found under $CAPS" >&2; exit 1; }
  cp "$f" "$TMP/cy_e2e/caps/"
done
ssh -o ConnectTimeout=8 "$BOARD" 'rm -rf /tmp/cy_e2e; df -h /tmp | tail -1; cat /proc/loadavg'
scp -O -q -r "$TMP/cy_e2e" "$BOARD:/tmp/"
ssh "$BOARD" "cd /tmp/cy_e2e && echo \"load before: \$(cat /proc/loadavg)\" &&
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 3 python3 bench_e2e_board.py . bundle $REPS timing_board.json caps/*.bin &&
  echo \"load after: \$(cat /proc/loadavg)\"" | tee "$OUT/timing_board.log"
scp -O -q "$BOARD:/tmp/cy_e2e/timing_board.json" "$OUT/"
ssh "$BOARD" 'rm -rf /tmp/cy_e2e'
echo "fetched $OUT/timing_board.json; board /tmp/cy_e2e removed"
