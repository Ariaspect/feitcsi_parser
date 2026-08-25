#!/usr/bin/env bash
#
# Grab one JPEG per second from the webcam for $1 seconds into $2.
#
#   scripts/webcam_capture.sh 60 /tmp/frames
#
# Runs alongside a CSI capture so the frames can label it. MJPG is the
# camera's native format, so a grab is a straight copy out of the device with
# no transcode -- which is why this needs no ffmpeg.

set -uo pipefail

SECS=${1:?usage: webcam_capture.sh SECONDS OUTDIR}
OUTDIR=${2:?usage: webcam_capture.sh SECONDS OUTDIR}

DEV=${WEBCAM_DEV:-/dev/video2}
W=${WEBCAM_W:-640}
H=${WEBCAM_H:-480}

# The first frames after opening a UVC device are exposed for whatever the
# sensor was doing before, and come out black or blown out. Dropping four
# costs ~0.15s and is the difference between a usable label and a dark frame.
SKIP=${WEBCAM_SKIP:-4}

mkdir -p "$OUTDIR"

start=$(date +%s.%N)
i=0
n=0
fail=0

while :; do
    now=$(date +%s.%N)
    [ "$(echo "$now - $start >= $SECS" | bc)" = "1" ] && break

    # Millisecond resolution in the name, not seconds: it is what lets a frame
    # be matched to a CSI group later, and two grabs can land in one second.
    #
    # Truncated by hand rather than with %3N, which this system's date ignores
    # -- it is uutils coreutils, not GNU, and emits all nine nanosecond digits.
    # One date call, then a slice, so the seconds and the fraction cannot come
    # from opposite sides of a second boundary.
    stamp=$(date +%Y%m%d_%H%M%S_%N)
    ts=${stamp:0:19}
    if v4l2-ctl -d "$DEV" \
            --set-fmt-video=width="$W",height="$H",pixelformat=MJPG \
            --stream-mmap --stream-skip="$SKIP" --stream-count=1 \
            --stream-to="$OUTDIR/$ts.jpg" >/dev/null 2>&1; then
        n=$(( n + 1 ))
    else
        fail=$(( fail + 1 ))
        rm -f "$OUTDIR/$ts.jpg"
    fi

    # Absolute deadlines, not "sleep 1". A grab costs ~0.6s, so sleeping a flat
    # second would drift by that much every frame, and within an hour the frame
    # timestamps would have wandered far from the CSI they are meant to label.
    i=$(( i + 1 ))
    target=$(echo "$start + $i" | bc)
    delay=$(echo "$target - $(date +%s.%N)" | bc)
    [ "$(echo "$delay > 0" | bc)" = "1" ] && sleep "$delay"
done

echo "$n frames, $fail failures"
