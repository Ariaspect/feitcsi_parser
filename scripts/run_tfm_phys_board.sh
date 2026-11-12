#!/usr/bin/env bash
# Board timing of the Physics tab's models (scripts/bench_tfm_phys_board.py) on the 8 windows lg was timed on.
#   scripts/run_tfm_phys_board.sh [BOARD]        (from the laptop; default root@192.168.50.80, the cable)
# Copies backend/ + data/ + the script + the captures to the board's /tmp/cy_phys (legacy scp -O: no sftp-server),
# runs it pinned to core 3, fetches the result to $OUT and removes /tmp/cy_phys. Never touches the serial console.
set -euo pipefail
BOARD=${1:-root@192.168.50.80}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CAPS=/home/cyphy/feitcsi_parser/captures
OUT=${OUT:-$ROOT/scripts/out_tfm_phys}
STEMS="20261001_151502 20261002_040002 20261013_120426 20261021_142736 20261023_111259 20261009_112535 20261013_164146 20261021_164727"
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cy_phys/caps" "$TMP/cy_phys/data"
tar -c -C "$ROOT" --exclude=__pycache__ backend | tar -x -C "$TMP/cy_phys"
cp "$ROOT/data/gt_tables.csv" "$ROOT/data/ml_heads_train.csv" "$TMP/cy_phys/data/"
cp "$ROOT/scripts/bench_tfm_phys_board.py" "$TMP/cy_phys/"
W="{"
for s in $STEMS; do
  f=$(ls "$CAPS/$s.bin" "$CAPS/lg_archive/${s:0:8}/$s.bin" 2>/dev/null | head -1)
  cp "$f" "$TMP/cy_phys/caps/"
  W="$W\"$s\": [0.0, 61.0],"
done
echo "${W%,}}" > "$TMP/cy_phys/timing_windows.json"
ssh -o ConnectTimeout=8 "$BOARD" 'rm -rf /tmp/cy_phys; df -h /tmp | tail -1; cat /proc/loadavg'
scp -O -q -r "$TMP/cy_phys" "$BOARD:/tmp/"
ssh "$BOARD" 'cd /tmp/cy_phys && echo "load before: $(cat /proc/loadavg)" &&
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 3 python3 bench_tfm_phys_board.py . timing_windows.json 3 timing_board.json caps/*.bin &&
  echo "load after: $(cat /proc/loadavg)"' | tee "$OUT/timing_board.log"
scp -O -q "$BOARD:/tmp/cy_phys/timing_board.json" "$OUT/"
ssh "$BOARD" 'rm -rf /tmp/cy_phys'
echo "fetched $OUT/timing_board.json; board /tmp/cy_phys removed"
