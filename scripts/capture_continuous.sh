#!/usr/bin/env bash
#
# Supervisor for continuous MTK CSI capture. Runs capture_mtk_hourly.sh
# back-to-back instead of on the hour, so the only gap between segments is the
# ssh setup and driver arm (~2-5 s) rather than cron's 5 minutes.
#
#   systemctl --user start csi-capture
# or by hand:
#   ./capture_continuous.sh
#
# Why segments at all, rather than one endless stream: tag 18 packs the group
# id into 16 bits, so it wraps after 65536 groups. At the measured 19.2
# groups/s that is 56.9 min, and a file spanning the wrap contains two groups
# with the same id. 3000 s (50 min) leaves margin for a faster-than-measured
# rate; at 19.2/s it reaches 57600 groups, 88% of the ceiling.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

SEGMENT=${SEGMENT:-3000}
CAPTURE_DIR=${CAPTURE_DIR:-/home/cyphy/feitcsi_parser/captures}
RUNLOG=$CAPTURE_DIR/mtk_supervisor.log

# Backoff so a board that is off, or a laptop on the wrong network, does not
# spin the loop once a second and fill the log. Resets after any good segment.
BACKOFF_MIN=${BACKOFF_MIN:-30}
BACKOFF_MAX=${BACKOFF_MAX:-600}

mkdir -p "$CAPTURE_DIR"
say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$RUNLOG"; }

backoff=$BACKOFF_MIN
trap 'say "supervisor stopping"; exit 0' INT TERM

say "supervisor start, ${SEGMENT}s segments"
while :; do
    start=$SECONDS
    DURATION=$SEGMENT CAPTURE_DIR=$CAPTURE_DIR ./capture_mtk_hourly.sh
    rc=$?
    elapsed=$(( SECONDS - start ))

    # A segment that ends far early failed even if it reported success -- the
    # usual cause is the board dropping the ssh mid-stream, which still writes
    # a short but valid file. Treat it as a failure for backoff purposes so a
    # flapping link does not produce a tight loop of 10-second captures.
    if [ "$rc" -eq 0 ] && [ "$elapsed" -ge $(( SEGMENT / 2 )) ]; then
        backoff=$BACKOFF_MIN
        continue
    fi

    say "segment ended rc=$rc after ${elapsed}s, retrying in ${backoff}s"
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    [ "$backoff" -gt "$BACKOFF_MAX" ] && backoff=$BACKOFF_MAX
done
