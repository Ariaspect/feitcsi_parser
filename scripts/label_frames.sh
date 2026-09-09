#!/usr/bin/env bash
#
# Label a capture's webcam frames with person detections.
#
#   scripts/label_frames.sh <capture.bin|stamp>
#   scripts/label_frames.sh 20260909_195450
#
# Produces <stamp>_cv.json beside the capture on the collection host and pulls
# a copy back. Every capture with frames gets one: a capture whose labels were
# never generated is a capture nobody can score later, and generating them
# after the fact means finding the frames again.
#
# Runs on the COLLECTION HOST, not here. The detector wants a GPU (55 fps
# there against a CPU crawl here), the frames are already on that side, and
# labelling must never compete with a capture for this laptop's radio or its
# camera.
#
# Detections are stored as boxes WITH confidences and WITHOUT an ROI filter,
# deliberately. A frame-wide "occupied" boolean is wrong the moment a second
# person is in shot -- 20260903_195337 has a bystander at the desk for its
# whole length and scores 100% occupied frame-wide, 91.8% against a chair ROI
# -- and the fix is to score a region in analysis, where it can be changed,
# not to discard the evidence here.
set -uo pipefail

TARGET=${1:-}
[ -n "$TARGET" ] || { echo "usage: label_frames.sh <capture.bin|stamp>" >&2; exit 2; }

STAMP=$(basename "$TARGET"); STAMP=${STAMP%.bin}
DAY=${STAMP%%_*}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
CAPTURE_DIR=${CAPTURE_DIR:-$REPO/captures}
REMOTE=${REMOTE:-lg}
REMOTE_DIR=${REMOTE_DIR:-/home/lg_csi/lg_csi_captures}
CV_PY=${CV_PY:-/home/lg_csi/feitcsi_parser/.venv/bin/python}
CV_SCRIPT=${CV_SCRIPT:-/home/lg_csi/feitcsi_parser/scripts/cv_presence.py}
SAMPLES=${CV_SAMPLES:-6}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)

# The archive format changed mid-dataset: 20260824 and part of 20260825 are
# .tar.zst, everything since is a plain .tar. Try both rather than reporting
# "no frames" for the compressed ones.
remote_tar=""
for ext in tar tar.zst; do
    candidate=$REMOTE_DIR/$DAY/${STAMP}_frames.$ext
    if ssh "${SSH_OPTS[@]}" "$REMOTE" "[ -f '$candidate' ]" 2>/dev/null; then
        remote_tar=$candidate
        break
    fi
done
[ -n "$remote_tar" ] || { echo "no frame archive for $STAMP on $REMOTE" >&2; exit 1; }

# An existing sidecar is fetched rather than regenerated. The pass is
# deterministic, so re-running it produces the same file for 11 s of GPU --
# and cv_presence.py refuses to overwrite without -f anyway. CV_FORCE=1 for a
# new model, a new confidence floor, or a changed ROI.
remote_json=$REMOTE_DIR/$DAY/${STAMP}_cv.json
if [ "${CV_FORCE:-0}" = "1" ] \
   || ! ssh "${SSH_OPTS[@]}" "$REMOTE" "[ -f '$remote_json' ]" 2>/dev/null; then
    force=""
    [ "${CV_FORCE:-0}" = "1" ] && force="-f"
    if ! ssh "${SSH_OPTS[@]}" "$REMOTE" \
            "$CV_PY $CV_SCRIPT '$remote_tar' --save-annotated $SAMPLES $force" \
            >/dev/null 2>&1; then
        echo "labelling failed for $STAMP -- frames are archived, so it can be re-run" >&2
        exit 1
    fi
fi

out=$CAPTURE_DIR/${STAMP}_cv.json
if scp -o BatchMode=yes -o LogLevel=ERROR -q \
        "$REMOTE:$REMOTE_DIR/$DAY/${STAMP}_cv.json" "$out" 2>/dev/null; then
    python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
frames = d.get("frames", [])
withp = sum(1 for f in frames if f.get("boxes"))
most = max((f.get("n", 0) for f in frames), default=0)
if frames:
    print("  cv: %d/%d frames with a person (%.1f%%), max %d in frame"
          % (withp, len(frames), 100.0 * withp / len(frames), most))
    if most > 1:
        # Worth saying out loud: frame-wide occupancy stops meaning "the
        # subject" the moment a second person is in shot.
        print("  cv: MORE THAN ONE PERSON in some frames -- score a chair ROI,"
              " not the whole frame")
PY
    echo "  cv sidecar: $(basename "$out")"
else
    echo "labelling ran but the sidecar did not come back" >&2
    exit 1
fi
