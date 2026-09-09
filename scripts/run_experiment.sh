#!/usr/bin/env bash
#
# One labelled presence run: empty -> sitting -> empty, 300s total.
#
#   scripts/run_experiment.sh POSITION [NOTE...]
#   scripts/run_experiment.sh chair_facing_tv "leaning back"
#
# The protocol is 90s empty / 120s sitting / 90s empty. The two empty stretches
# bracket the sitting one deliberately: each run carries its own empty-room
# reference, so presence is judged against the room minutes earlier rather than
# against a baseline recorded on some other day with the furniture moved.
#
# A LEAD_IN countdown runs before the capture starts, because the laptop is in
# the room -- without it the first phase would record the operator walking out
# rather than an empty room.
#
# Writes <stamp>_meta.json beside the capture and uploads it with the rest, so
# the phase boundaries travel with the data instead of living in someone's
# notebook. Times in the sidecar are UTC epoch seconds: the board's clock is
# free-running with no ntp, so anything derived from board-local time is only
# as trustworthy as the last time someone checked it. The measured host/board
# offset is recorded per run for exactly that reason.
set -uo pipefail

POSITION=${1:-}
[ -n "$POSITION" ] || { echo "usage: run_experiment.sh POSITION [NOTE...]" >&2; exit 2; }
shift
NOTE="$*"

# Filenames and JSON keys both get this, so keep it boring.
SAFE_POSITION=$(printf '%s' "$POSITION" | tr -c '[:alnum:]_-' '_')

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
CAPTURE_DIR=${CAPTURE_DIR:-$REPO/captures}
LEAD_IN=${LEAD_IN:-10}
DURATION=${DURATION:-300}
PHASE1_END=${PHASE1_END:-90}
PHASE2_END=${PHASE2_END:-210}
BOARD=${BOARD:-192.168.50.80}
REMOTE=${REMOTE:-lg}
REMOTE_DIR=${REMOTE_DIR:-/home/lg_csi/lg_csi_captures}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)

WEBCAM_DEV=${WEBCAM_DEV:-/dev/video2}
if fuser "$WEBCAM_DEV" >/dev/null 2>&1; then
    echo "$WEBCAM_DEV is busy -- stop the live stream first." >&2
    exit 1
fi

# Measured before the capture, while nothing is competing for the link.
BOARD_EPOCH=$(ssh "${SSH_OPTS[@]}" "root@$BOARD" 'date -u +%s' 2>/dev/null)
HOST_EPOCH=$(date -u +%s)
if [ -n "$BOARD_EPOCH" ]; then
    OFFSET=$(( BOARD_EPOCH - HOST_EPOCH ))
else
    OFFSET=""
    echo "warning: could not read board clock" >&2
fi

cat <<BANNER

  position : $POSITION
  protocol : 0-${PHASE1_END}s EMPTY | ${PHASE1_END}-${PHASE2_END}s SITTING | ${PHASE2_END}-${DURATION}s EMPTY
  clock    : board-host offset ${OFFSET:-unknown}s

BANNER

echo "LEAVE THE ROOM NOW -- capture starts in ${LEAD_IN}s"
for i in $(seq "$LEAD_IN" -1 1); do printf '\r  %2ds ' "$i"; sleep 1; done
printf '\r      \r'

START_EPOCH=$(date -u +%s)
echo "capture running (${DURATION}s)..."
echo "  sit down at   +$(( PHASE1_END / 60 ))m$(( PHASE1_END % 60 ))s"
echo "  leave again at +$(( PHASE2_END / 60 ))m$(( PHASE2_END % 60 ))s"

WEBCAM_REQUIRED=1 RETAIN_DAYS=${RETAIN_DAYS:-30} \
    "$HERE/capture_mtk_hourly.sh" --duration "$DURATION"
RC=$?

if [ "$RC" != "0" ]; then
    echo "capture failed (rc=$RC) -- no sidecar written" >&2
    exit "$RC"
fi

# The stamp is minted inside the capture script, so recover it rather than
# guessing: newest .bin that appeared after we started.
BIN=$(find "$CAPTURE_DIR" -maxdepth 1 -name '????????_??????.bin' -newermt "@$((START_EPOCH - 5))" \
      -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$BIN" ]; then
    echo "capture reported success but no new .bin found in $CAPTURE_DIR" >&2
    exit 1
fi
STAMP=$(basename "$BIN" .bin)
META=$CAPTURE_DIR/${STAMP}_meta.json

# The archive format has changed once already: segments from 20260824 and part
# of 20260825 are _frames.tar.zst, everything since is a plain _frames.tar.
# Globbing only one of them reports zero frames without failing, which would
# put a confident "frames_archived": 0 into the sidecar of a perfectly good run.
FRAMES=0
for ext in tar tar.zst; do
    ARC=$CAPTURE_DIR/${STAMP}_frames.$ext
    [ -f "$ARC" ] || continue
    case $ext in
        tar.zst) FRAMES=$(tar --zstd -tf "$ARC" 2>/dev/null | grep -c '\.jpg$') ;;
        *)       FRAMES=$(tar -tf "$ARC" 2>/dev/null | grep -c '\.jpg$') ;;
    esac
    break
done

STAMP="$STAMP" POSITION="$POSITION" SAFE_POSITION="$SAFE_POSITION" NOTE="$NOTE" \
DURATION="$DURATION" PHASE1_END="$PHASE1_END" PHASE2_END="$PHASE2_END" \
START_EPOCH="$START_EPOCH" OFFSET="$OFFSET" FRAMES="$FRAMES" LEAD_IN="$LEAD_IN" \
python3 - "$META" <<'PY'
import json, os, sys
from datetime import datetime

def phase(label, a, b):
    return {"label": label, "start_s": a, "end_s": b}

p1, p2 = int(os.environ["PHASE1_END"]), int(os.environ["PHASE2_END"])
dur = int(os.environ["DURATION"])
off = os.environ.get("OFFSET") or None

# Phase boundaries are relative to the capture's own start, and the only thing
# naming that instant is the stamp -- which is LOCAL time (KST), while the
# collection host is UTC. A naive parse of these filenames lands 9h out and does
# so silently, so resolve it here, once, and publish the epoch alongside it.
#
# wrapper_start is NOT that instant: preflight (ping, ssh, wifi guard, board
# script deploy) runs in between, so anchoring phases to the wrapper start
# shifts every label by however long preflight happened to take on that run.
stamp = os.environ["STAMP"]
local = datetime.strptime(stamp, "%Y%m%d_%H%M%S").astimezone()

meta = {
    "stamp": stamp,
    "stamp_tz": local.tzname(),
    "capture_start_utc_epoch": int(local.timestamp()),
    "phases_relative_to": "capture_start_utc_epoch",
    "position": os.environ["POSITION"],
    "position_key": os.environ["SAFE_POSITION"],
    "note": os.environ["NOTE"] or None,
    "protocol": "empty-sitting-empty",
    "duration_s": dur,
    "phases": [phase("empty", 0, p1), phase("sitting", p1, p2), phase("empty", p2, dur)],
    "lead_in_s": int(os.environ["LEAD_IN"]),
    "wrapper_start_utc_epoch": int(os.environ["START_EPOCH"]),
    "board_minus_host_s": int(off) if off not in (None, "") else None,
    "frames_archived": int(os.environ["FRAMES"] or 0),
    "camera": os.environ.get("WEBCAM_DEV", "/dev/video2"),
}
json.dump(meta, open(sys.argv[1], "w"), indent=2)
print("sidecar:", sys.argv[1])
PY

DAY=${STAMP%%_*}
if ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR/$DAY'" 2>/dev/null \
   && scp -o BatchMode=yes -o LogLevel=ERROR -q "$META" "$REMOTE:$REMOTE_DIR/$DAY/"; then
    echo "uploaded $(basename "$META")"
else
    echo "warning: sidecar upload failed, kept locally" >&2
fi

# Labelling runs AFTER the capture, on the collection host, against the archived
# frames -- never live alongside it. Three reasons, in order of how much they'd
# cost us:
#   1. The camera is a single-consumer v4l2 device. A live detector would have
#      to take /dev/video2 from webcam_capture.sh, i.e. trade the ground truth
#      itself for a view of it.
#   2. Live inference means continuous traffic on wlo1, which is adjacent to the
#      band the board is measuring. Labelling must not perturb the measurement.
#   3. Offline is re-runnable. Model, confidence floor and ROI can all change
#      later and be re-applied to the same frames; a live verdict is a number
#      nobody can reproduce. Given the ROI already proved unstable once against
#      an occluded subject, being able to re-score is worth more than immediacy.
# Boxes are stored WITH confidences and WITHOUT an ROI filter for the same
# reason: a bystander at the desk makes any frame-wide "occupied" boolean wrong,
# and the fix is to score a region later, not to discard the evidence now.
# Labelling is done by capture_mtk_hourly.sh for every capture, labelled run
# or not, so there is nothing to do here. It used to be duplicated: two copies
# of the same ssh-and-scp drifted the moment one of them learned that the
# archive may be .tar or .tar.zst.

echo
echo "  $STAMP  position=$POSITION  frames=$FRAMES"
echo "  review, then run the next position."
