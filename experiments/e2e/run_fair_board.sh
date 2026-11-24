#!/usr/bin/env bash
# Board timing, e2e candidate vs 13F / 15F in one run (bench_fair_board.py) on the 8 windows both were timed on.
#   experiments/e2e/run_fair_board.sh [REPS] [BOARD]      (from the laptop; default root@192.168.50.80, the cable)
# Copies backend/ + data/ (wt-ml) + e2e_np.py + the bench + seeds 0-2 of the two e2e bundles + the captures to the
# board's /tmp/cy_fair (legacy scp -O), runs it pinned to core 3, fetches the result, removes /tmp/cy_fair.
# Never touches the serial console.
set -euo pipefail
REPS=${1:-3}; BOARD=${2:-root@192.168.50.80}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/home/cyphy/feitcsi_parser-wt-ml
CAPS=/home/cyphy/feitcsi_parser/captures
OUT=${OUT:-$HERE/out/board}
STEMS="20261001_151502 20261002_040002 20261013_120426 20261021_142736 20261023_111259 20261009_112535 20261013_164146 20261021_164727"
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cy_fair/caps" "$TMP/cy_fair/data" "$TMP/cy_fair/bundle"
tar -c -C "$ROOT" --exclude=__pycache__ backend | tar -x -C "$TMP/cy_fair"
cp "$ROOT/data/gt_tables.csv" "$ROOT/data/ml_heads_train.csv" "$TMP/cy_fair/data/"
cp "$HERE/e2e_np.py" "$HERE/bench_fair_board.py" "$TMP/cy_fair/"
cp "$HERE/out/board_bundle2/meta.json" "$TMP/cy_fair/bundle/"
for c in sbo.B.lfhftl.st16.att sbo.B.all4.st16.att; do for s in 0 1 2; do
  cp "$HERE/out/board_bundle2/${c}__s$s.npz" "$TMP/cy_fair/bundle/"; done; done
for s in $STEMS; do
  f=""
  for c in "$CAPS/$s.bin" "$CAPS/lg_archive/${s:0:8}/$s.bin"; do
    [ -f "$c" ] && { f=$c; break; }
  done
  [ -n "$f" ] || { echo "capture $s not found under $CAPS" >&2; exit 1; }
  cp "$f" "$TMP/cy_fair/caps/"
done
ssh -o ConnectTimeout=8 "$BOARD" 'rm -rf /tmp/cy_fair; df -h /tmp | tail -1; cat /proc/loadavg'
scp -O -q -r "$TMP/cy_fair" "$BOARD:/tmp/"
ssh "$BOARD" "cd /tmp/cy_fair && echo \"load before: \$(cat /proc/loadavg)\" &&
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 3 python3 bench_fair_board.py . bundle $REPS timing_fair.json caps/*.bin &&
  echo \"load after: \$(cat /proc/loadavg)\"" | tee "$OUT/timing_fair.log"
scp -O -q "$BOARD:/tmp/cy_fair/timing_fair.json" "$OUT/"
ssh "$BOARD" 'rm -rf /tmp/cy_fair'
echo "fetched $OUT/timing_fair.json; board /tmp/cy_fair removed"
