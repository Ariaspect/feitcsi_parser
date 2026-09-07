#!/usr/bin/env bash
#
# Live webcam view on the collection host, reachable only through an ssh tunnel.
#
#   scripts/webcam_stream.sh [SECONDS]        # default: run until interrupted
#
# Then, from wherever you want to watch:
#   ssh -N -L 8001:127.0.0.1:8001 lg
#   open http://localhost:8001/
#
# Why H.264 on the wire and MJPEG only at the far end: the laptop's uplink
# (wlo1, 5250-5290 MHz) sits directly against the 5170-5250 MHz band the board
# measures, with no guard. Raw MJPEG would put ~9.4 Mbit/s of continuous TX
# beside the measurement; H.264 carries the same picture at ~1.5 Mbit/s. The
# expensive-to-transmit format is reconstructed on lg, where bandwidth is free.
#
# The relay binds loopback on lg, so nothing is exposed to the network -- the
# ssh tunnel is the only way in, which is also why no auth is built in.
set -uo pipefail

DUR=${1:-}
DEV=${DEV:-/dev/video2}
W=${W:-1280}
H=${H:-720}
FPS=${FPS:-15}
BITRATE=${BITRATE:-1500k}
REMOTE=${REMOTE:-lg}
RELAY=${RELAY:-/home/lg_csi/bin/mjpeg_relay.py}
FFMPEG_REMOTE=${FFMPEG_REMOTE:-/home/lg_csi/bin/ffmpeg}
JPEG_Q=${JPEG_Q:-5}

[ -e "$DEV" ] || { echo "no such camera: $DEV" >&2; exit 1; }

# fuser rather than a lock file: the capture pipeline grabs the same camera, and
# what matters is whether the device is actually held, not whether we think so.
if fuser "$DEV" >/dev/null 2>&1; then
    echo "$DEV is busy -- a capture is probably running. Refusing to fight for it." >&2
    exit 1
fi

TLIMIT=()
[ -n "$DUR" ] && TLIMIT=(-t "$DUR")

echo "streaming $W x $H @ ${FPS}fps, H.264 ${BITRATE} -> $REMOTE:8001 (loopback)"
echo "tunnel with:  ssh -N -L 8001:127.0.0.1:8001 $REMOTE"

ffmpeg -hide_banner -loglevel error \
    -f v4l2 -input_format mjpeg -video_size "${W}x${H}" -framerate "$FPS" \
    -i "$DEV" -an "${TLIMIT[@]}" \
    -c:v libx264 -preset veryfast -tune zerolatency \
    -b:v "$BITRATE" -maxrate "$BITRATE" -bufsize "$((${BITRATE%k} * 2))k" \
    -g "$((FPS * 2))" -pix_fmt yuv420p \
    -f matroska - \
| ssh -o BatchMode=yes -o ServerAliveInterval=15 -o LogLevel=ERROR "$REMOTE" \
    "$FFMPEG_REMOTE -hide_banner -loglevel error -i - \
        -f image2pipe -c:v mjpeg -q:v $JPEG_Q - | python3 $RELAY"
