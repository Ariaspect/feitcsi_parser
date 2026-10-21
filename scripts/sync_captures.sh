#!/usr/bin/env bash
#
# Push captures that never reached the collection host -- recorded while lg was
# unreachable -- and label the ones that arrived without a camera sidecar.
#
#   scripts/sync_captures.sh              # upload what is missing, then label
#   scripts/sync_captures.sh --check      # only count what is missing, print "pending N"
#   scripts/sync_captures.sh --no-label   # upload only
#   DAYS=7 scripts/sync_captures.sh       # look back 7 days instead of 30
#
# The capture script uploads each capture as it ends and labels it on lg; when
# lg is down that step just warns, and nothing came back for the file later
# (2026-10-15: two captures from 18:15 sat on the laptop all evening). This is
# the later. It is idempotent: a file is sent only when lg does not have it,
# and labelling runs only where lg has the frames but no _cv.json.
#
# Per stem it moves <stem>.bin, <stem>_frames.tar (or .tar.zst) and
# <stem>_meta.json into lg:<REMOTE_DIR>/<day>/, the layout the capture script
# uses, so the web on lg lists them like any other capture.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CAPTURE_DIR=${CAPTURE_DIR:-/home/cyphy/feitcsi_parser/captures}
REMOTE=${REMOTE:-lg}
REMOTE_DIR=${REMOTE_DIR:-/home/lg_csi/lg_csi_captures}
DAYS=${DAYS:-30}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)

check=0; label=1
for a in "$@"; do
    case $a in
        --check)    check=1 ;;
        --no-label) label=0 ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

# A capture's own upload and labelling may still be running in the background
# (capture_mtk_hourly.sh POST_ASYNC=1, holding POST_LOCK shared). Wait for them,
# or this would send the same files a second time and label twice.
if [ "$check" = 0 ]; then
    exec 8>"${POST_LOCK:-/tmp/mtk-post.lock}"
    flock -w 900 8 || echo "background uploads still running after 900 s, syncing anyway" >&2
fi

# Local stems of the last DAYS days: the flat layout the capture script writes.
stems=$(find "$CAPTURE_DIR" -maxdepth 1 -name '????????_??????.bin' -mtime -"$DAYS" -printf '%f\n' 2>/dev/null \
        | sed 's/\.bin$//' | sort)
[ -n "$stems" ] || { echo "pending 0"; exit 0; }
days=$(printf '%s\n' "$stems" | cut -c1-8 | sort -u)

# One round trip: everything lg has for those days.
remote=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "for d in $(printf '%s ' $days); do ls '$REMOTE_DIR'/\$d 2>/dev/null; done; true") \
    || { echo "lg not reachable" >&2; exit 1; }
# A here-string, not a pipe: under pipefail, grep -q closing the pipe early
# would fail the pipeline and report every file as missing.
have() { grep -qxF -- "$1" <<<"$remote"; }

pending=0; uploaded=0; failed=0; to_label=()
for s in $stems; do
    day=${s%%_*}
    for f in "$s.bin" "${s}_frames.tar" "${s}_frames.tar.zst" "${s}_meta.json"; do
        [ -f "$CAPTURE_DIR/$f" ] || continue
        have "$f" && continue
        pending=$(( pending + 1 ))
        [ "$check" = 1 ] && continue
        if ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR/$day'" \
           && scp -o BatchMode=yes -o LogLevel=ERROR -q "$CAPTURE_DIR/$f" "$REMOTE:$REMOTE_DIR/$day/"; then
            echo "uploaded $f"
            uploaded=$(( uploaded + 1 ))
        else
            echo "FAILED $f" >&2
            failed=$(( failed + 1 ))
        fi
    done
    # Frames there (now or before) but no camera verdict yet.
    if [ "$label" = 1 ] && [ "$check" = 0 ] && ! have "${s}_cv.json" \
       && { [ -f "$CAPTURE_DIR/${s}_frames.tar" ] || [ -f "$CAPTURE_DIR/${s}_frames.tar.zst" ] \
            || have "${s}_frames.tar" || have "${s}_frames.tar.zst"; }; then
        to_label+=("$s")
    fi
done

if [ "$check" = 1 ]; then
    echo "pending $pending"
    exit 0
fi

labelled=0
for s in "${to_label[@]}"; do
    if out=$("$HERE/label_frames.sh" "$s" 2>&1); then
        echo "labelled $s"
        labelled=$(( labelled + 1 ))
    else
        echo "label skipped $s: $(printf '%s' "$out" | tail -1)" >&2
    fi
done
echo "uploaded $uploaded, failed $failed, labelled $labelled, pending-before $pending"
[ "$failed" = 0 ]
