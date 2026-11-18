#!/usr/bin/env bash
# Board timing of the e2e candidate's value-identical speed-ups (bench_opt_board.py: only the used bins, seeds forked
# onto 2 / 3 cores) on the 8 windows the models were timed on (cyphy session, 2026-11-17).
#   experiments/e2e/run_fair_board.sh [REPS] [BOARD]      (from the laptop; default root@192.168.50.80, the cable)
# Copies backend/ + data/ (wt-ml) + e2e_np.py + the bench + seeds 0-2 of the two e2e bundles + the captures to the
# board's /tmp/cy_opt (legacy scp -O), runs it (the script pins core 3, children cores 2 / 1), fetches the result, removes /tmp/cy_opt.
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
mkdir -p "$TMP/cy_opt/caps" "$TMP/cy_opt/data" "$TMP/cy_opt/bundle"
tar -c -C "$ROOT" --exclude=__pycache__ backend | tar -x -C "$TMP/cy_opt"
cp "$ROOT/data/gt_tables.csv" "$ROOT/data/ml_heads_train.csv" "$TMP/cy_opt/data/"
cp "$HERE/e2e_np.py" "$HERE/bench_opt_board.py" "$TMP/cy_opt/"
cp "$HERE/out/board_bundle2/meta.json" "$TMP/cy_opt/bundle/"
for c in sbo.B.lfhftl.st16.att; do for s in 0 1 2; do
  cp "$HERE/out/board_bundle2/${c}__s$s.npz" "$TMP/cy_opt/bundle/"; done; done
for s in $STEMS; do
  f=""
  for c in "$CAPS/$s.bin" "$CAPS/lg_archive/${s:0:8}/$s.bin"; do
    [ -f "$c" ] && { f=$c; break; }
  done
  [ -n "$f" ] || { echo "capture $s not found under $CAPS" >&2; exit 1; }
  cp "$f" "$TMP/cy_opt/caps/"
done
ssh -o ConnectTimeout=8 "$BOARD" 'rm -rf /tmp/cy_opt; df -h /tmp | tail -1; cat /proc/loadavg'
scp -O -q -r "$TMP/cy_opt" "$BOARD:/tmp/"
ssh "$BOARD" "cd /tmp/cy_opt && echo \"load before: \$(cat /proc/loadavg)\" &&
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CORE=3 CORES=3,2,1 python3 bench_opt_board.py . bundle sbo.B.lfhftl.st16.att $REPS timing_opt.json caps/*.bin &&
  echo \"load after: \$(cat /proc/loadavg)\"" | tee "$OUT/timing_opt.log"
scp -O -q "$BOARD:/tmp/cy_opt/timing_opt.json" "$OUT/"
ssh "$BOARD" 'rm -rf /tmp/cy_opt'
echo "fetched $OUT/timing_opt.json; board /tmp/cy_opt removed"
