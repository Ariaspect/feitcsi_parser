#!/usr/bin/env bash
#
# MediaTek CSI capture from the LG webOS board, streamed to this host.
#
# Two cron modes, each pairing a capture length with the schedule it belongs to:
#
#   0 * * * *    /home/cyphy/feitcsi_parser/scripts/capture_mtk_hourly.sh --mode hourly
#   0,30 * * * * /home/cyphy/feitcsi_parser/scripts/capture_mtk_hourly.sh --mode 30min
#   */10 * * * * /home/cyphy/feitcsi_parser/scripts/capture_mtk_hourly.sh --mode 10min
#
# Rather than writing those by hand, have the script emit the stanza:
#   scripts/capture_mtk_hourly.sh --mode 10min --show-cron | crontab -
#
# By hand, with a short run to check the plumbing:
#   scripts/capture_mtk_hourly.sh --duration 60
#
# Why stream rather than capture-then-fetch: /proc/net/wlan/csi_data is a
# blocking stream backed by a ring of only ~1024 records (~12.8 s at 20
# groups/s). With no reader attached the driver stops generating once that
# fills, so anything longer than ~13 s MUST have a reader running throughout.
#
# Why ssh and not the serial console: 55 minutes of CSI is ~275 MB at ~83 kB/s.
# A 115200-baud UART tops out near 11 kB/s, so serial is off by an order of
# magnitude before base64 overhead. Serial stays the recovery channel; see the
# shared tmux session in README-CAPTURE.md.

set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

BOARD=${BOARD:-192.168.50.80}
CAPTURE_DIR=${CAPTURE_DIR:-/home/cyphy/feitcsi_parser/captures}

# ------------------------------------------------------------------- modes --
# A mode fixes the capture length and the cron schedule together, so the two
# cannot drift apart. That pairing is the whole point: a duration equal to or
# longer than its own cron interval would leave the previous run still holding
# the lock when the next fires, and the flock below would skip every second
# capture rather than overlap them. Each mode therefore stops short of its
# interval, leaving room for preflight (wake, wifi check, ssh) and for the
# post-run parse.
#
# The hard ceiling is elsewhere and no mode comes near it: tag 18 packs the
# group id into 16 bits, so it wraps after 65536 groups -- 59 minutes at the
# measured 18.5 groups/s. A capture past that contains two groups sharing an id.
MODE=${MODE:-hourly}
DURATION_FLAG=""
SHOW_CRON=0

usage() {
    cat <<'USAGE'
Usage: capture_mtk_hourly.sh [OPTIONS]

  -m, --mode MODE       hourly  55 min every hour      (0 * * * *)
                        30min   29.5 min every half hour (0,30 * * * *)
                        10min   9.5 min every ten min  (*/10 * * * *)
  -d, --duration SECS   override the mode's capture length
      --show-cron       print the crontab stanza for the mode and exit
  -h, --help            this message

Environment variables still work and sit between the two: an explicit
--duration beats DURATION=, which beats the mode's own default.
USAGE
}

while [ $# -gt 0 ]; do
    case $1 in
        -m|--mode)     MODE=${2:?--mode needs a value};     shift 2 ;;
        --mode=*)      MODE=${1#*=};                        shift ;;
        -d|--duration) DURATION_FLAG=${2:?--duration needs a value}; shift 2 ;;
        --duration=*)  DURATION_FLAG=${1#*=};               shift ;;
        --show-cron)   SHOW_CRON=1;                         shift ;;
        -h|--help)     usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case $MODE in
    hourly) MODE_DURATION=3300; MODE_SCHEDULE='0 * * * *' ;;
    30min)  MODE_DURATION=1770; MODE_SCHEDULE='0,30 * * * *' ;;
    10min)  MODE_DURATION=570;  MODE_SCHEDULE='*/10 * * * *' ;;
    *) printf 'unknown mode: %s (expected hourly, 30min or 10min)\n' "$MODE" >&2; exit 2 ;;
esac

if [ -n "$DURATION_FLAG" ]; then
    DURATION=$DURATION_FLAG
else
    DURATION=${DURATION:-$MODE_DURATION}
fi

# Emitted before the lock and the capture dir are touched, so this stays a pure
# query that can be run while a capture is in progress.
if [ "$SHOW_CRON" = "1" ]; then
    printf '# %s mode: %ss capture on %s\n' "$MODE" "$DURATION" "$MODE_SCHEDULE"
    printf 'MAILTO=""\n'
    printf 'RETAIN_DAYS=%s\n' "${RETAIN_DAYS:-7}"
    printf '%s %s --mode %s\n' "$MODE_SCHEDULE" "$(readlink -f "$0")" "$MODE"
    exit 0
fi

# Stimulus. The board pings its own gateway over wlan0; CSI comes from received
# frames, so without downlink traffic there is nothing to measure. 20/s yields
# ~19.2 groups/s.
PING_INTERVAL=${PING_INTERVAL:-0.05}

# Measured per 10min segment: 43.5 MB of CSI at 4.8 MB/min, plus a 41 MB frame
# archive (JPEG barely compresses, so the archive is near the frames' raw size).
# That is ~12 GB/day, of which the webcam is very nearly half. Hourly mode lands
# in the same place -- the two modes spend almost the same fraction of the clock
# capturing, so retention costs the same either way. Seven days is ~85 GB
# against the 307 GB free here. Setting WEBCAM=0 halves it.
RETAIN_DAYS=${RETAIN_DAYS:-7}
MIN_FREE_MB=${MIN_FREE_MB:-3000}

# ------------------------------------------------------------------ remote --
# Each segment is pushed to the collection host as soon as it is complete.
# Measured 19.6 MB/s, so a 10min segment (~46 MB) plus its frame archive costs
# about 5 s -- small enough to do inline. Inline also matters for correctness:
# backgrounding it would put the upload's radio traffic on top of the next
# capture, and this laptop's wifi shares the band the board is measuring.
REMOTE=${REMOTE:-lg}
REMOTE_DIR=${REMOTE_DIR:-/home/lg_csi/lg_csi_captures}
UPLOAD=${UPLOAD:-1}

# ------------------------------------------------------------------- live ---
# Stream the capture to the collection host while it is still being written, so
# a viewer on another machine can watch it grow instead of waiting out the
# segment. The batch upload below stays the authority; this is additive, and a
# failed live copy costs a live view rather than data.
#
# Safe on the radio: this goes over the laptop's uplink (wlo1, 5620 MHz), 320
# MHz clear of the 5240 MHz channel the board measures. REQUIRE_WIRED still
# governs the board->laptop hop, which is what must never ride the measured
# link.
LIVE=${LIVE:-1}
LIVE_DIR=${LIVE_DIR:-$REMOTE_DIR/live}

# ------------------------------------------------------------------ webcam --
# One JPEG per second for the length of the capture, to label human presence
# and movement against the CSI. Frames land on scratch, are archived when the
# segment ends, and only the archive is kept and uploaded -- 570 loose files
# per segment is a burden on the filesystem and a very slow scp.
WEBCAM=${WEBCAM:-1}
WEBCAM_DEV=${WEBCAM_DEV:-/dev/video2}
WEBCAM_W=${WEBCAM_W:-640}
WEBCAM_H=${WEBCAM_H:-480}
FRAME_TMP=${FRAME_TMP:-/tmp/csi-frames}

# Bounds on a camera that stops answering. WEBCAM_GRAB_TIMEOUT caps one grab
# inside the grabber; WEBCAM_GRACE caps how long this script will wait for the
# grabber itself once the CSI stream has closed.
WEBCAM_GRAB_TIMEOUT=${WEBCAM_GRAB_TIMEOUT:-5}
WEBCAM_GRACE=${WEBCAM_GRACE:-30}
export WEBCAM_DEV WEBCAM_W WEBCAM_H WEBCAM_GRAB_TIMEOUT

# The wired path is CSI-safe; WiFi is not (see check_transport below).
REQUIRE_WIRED=${REQUIRE_WIRED:-1}
WIRED_IFACE=${WIRED_IFACE:-enp1s0}

# LogLevel=ERROR suppresses the client's post-quantum key-exchange warning,
# which this board's older sshd triggers on every single connection. Without
# it three lines of boilerplate land in the capture log per segment, which over
# a multi-day run buries the board diagnostics that share the same stderr.
# ERROR still lets real ssh failures through.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30
          -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new
          -o LogLevel=ERROR)

BOARD_DIR=/var/csi
BOARD_SCRIPT=$BOARD_DIR/csi_stream.sh

LOCK=/tmp/mtk-capture.lock
LOG=$CAPTURE_DIR/mtk_capture.log
VENV=${VENV:-/home/cyphy/feitcsi_parser/.venv/bin/python}

mkdir -p "$CAPTURE_DIR"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }
die() { log "ABORT $*"; exit 1; }

# Serialize. -n bails rather than queueing: a run that overran must not have a
# second capture stacked behind it, since both would fight for the one stream.
exec 9>"$LOCK"
flock -n 9 || { log "SKIP  previous capture still running"; exit 0; }

# ---------------------------------------------------------------- retention --
if [ "$RETAIN_DAYS" -gt 0 ]; then
    pruned=$(find "$CAPTURE_DIR" -maxdepth 1 -type f \
                  \( -name '????????_??????.bin' \
                     -o -name '????????_??????_frames.tar' \
                     -o -name '????????_??????_frames.tar.zst' \) \
                  -mtime +"$RETAIN_DAYS" -print -delete 2>/dev/null | wc -l)
    [ "$pruned" -gt 0 ] && log "PRUNE $pruned file(s) older than ${RETAIN_DAYS}d"
fi

free_mb=$(df -Pm "$CAPTURE_DIR" | awk 'NR==2 {print $4}')
[ "${free_mb:-0}" -ge "$MIN_FREE_MB" ] \
    || die "only ${free_mb}MB free, need ${MIN_FREE_MB}MB"

# --------------------------------------------------------------- serial wake --
# The board sleeps on its own, and once asleep it answers neither ping nor ssh.
# The RS-232C control port is the only channel still live: "ka 01 01" is LG's
# power-on command (ka = power, 01 = set id, 01 = on).
#
# It goes through the shared tmux session rather than to /dev/ttyUSB0 directly,
# because tio holds the port open -- a second writer would interleave with it
# and neither side would parse cleanly.
TIO_SESSION=${TIO_SESSION:-tio}
WAKE_CMD=${WAKE_CMD:-ka 01 01}
SHELL_KEY=${SHELL_KEY:-s}
BOARD_CIDR=${BOARD_CIDR:-24}
BOOT_WAIT=${BOOT_WAIT:-75}
WAKE_WAIT=${WAKE_WAIT:-90}

# Waking is three steps, not one, and skipping any of them leaves the board
# reachable-looking but unusable:
#
#   1. "ka 01 01" -- LG's RS-232C power-on (ka = power, 01 = set id, 01 = on).
#   2. "s" -- after boot the console is in the TV's control mode, not a shell.
#   3. re-apply eth0's address -- it is set with "ip addr add" and so does NOT
#      survive the reboot. Without this the board boots to a 169.254 link-local
#      address only, and every ssh in this script fails with "no route to host".
#
# All of it goes through the shared tmux session rather than to /dev/ttyUSB0
# directly, because tio holds the port open and a second writer would
# interleave with it.
wake_board() {
    tmux has-session -t "$TIO_SESSION" 2>/dev/null \
        || { log "WAKE  no tmux session '$TIO_SESSION', cannot wake"; return 1; }

    # If tio is not the pane's foreground process the pane is a plain shell,
    # and every string below would be run as a local shell command instead of
    # being sent down the wire -- a silent no-op, the worst failure mode here.
    pane_cmd=$(tmux list-panes -t "$TIO_SESSION" -F '#{pane_current_command}' \
                   2>/dev/null | head -1)
    [ "$pane_cmd" = "tio" ] \
        || { log "WAKE  pane runs '$pane_cmd', not tio -- refusing to send"; return 1; }

    log "WAKE  power on via '$WAKE_CMD'"
    tmux send-keys -t "$TIO_SESSION" "$WAKE_CMD" Enter
    sleep "$BOOT_WAIT"

    log "WAKE  entering shell mode"
    tmux send-keys -t "$TIO_SESSION" "$SHELL_KEY" Enter
    sleep 5

    # Idempotent: "ip addr add" on an address already present just reports
    # "File exists", which is fine -- it means the board never actually slept.
    log "WAKE  re-applying $BOARD/$BOARD_CIDR to eth0"
    tmux send-keys -t "$TIO_SESSION" \
        "ip addr add $BOARD/$BOARD_CIDR dev eth0; ip link set eth0 up" Enter

    waited=0
    while [ "$waited" -lt "$WAKE_WAIT" ]; do
        sleep 5
        waited=$(( waited + 5 ))
        if ping -c1 -W2 "$BOARD" >/dev/null 2>&1; then
            log "WAKE  board answered after ${waited}s"
            return 0
        fi
    done
    log "WAKE  board still silent ${WAKE_WAIT}s after shell+ifconfig"
    return 1
}

# ------------------------------------------------------------------- wifi --
# The capture measures wlan0. If the board is up and ssh-reachable over the
# cable but wlan0 has dropped the AP, every segment still "succeeds" -- it just
# contains no CSI, and that is only visible later in the frame count. So the
# association is checked before committing to a run, not after.
WIFI_SSID=${WIFI_SSID:-ASUS_00_5G}
WIFI_PROFILE_ID=${WIFI_PROFILE_ID:-777}
WIFI_WAIT=${WIFI_WAIT:-120}

# Association is read with iw/ip rather than the wifi service's own getstatus.
# luna-send delivers its reply asynchronously: it exits 0 before the response
# has been written, so a scripted read races it and comes back empty -- which
# is indistinguishable from "not associated" and would abort runs whose link
# was perfectly fine. The -w timeout flag does not close the race. Measured
# back to back on an idle, associated board: luna getstatus 0 for 5, iw/ip 5
# for 5.
wifi_up() {
    state=$(ssh "${SSH_OPTS[@]}" "root@$BOARD" \
        'iw dev wlan0 link 2>/dev/null | head -2; ip -4 addr show wlan0 2>/dev/null | grep inet' \
        2>>"$LOG")
    case $state in
        *"SSID: $WIFI_SSID"*) ;;
        *) return 1 ;;
    esac
    # Associated but addressless is not usable: the stimulus ping needs a
    # source address on wlan0, and DHCP can lag association by several seconds.
    case $state in
        *inet*) return 0 ;;
        *)      return 1 ;;
    esac
}

ensure_wifi() {
    wifi_up && return 0

    log "WIFI  wlan0 not on $WIFI_SSID, connecting to profileId $WIFI_PROFILE_ID"
    # Fire and forget. The reply cannot be read reliably (see above), so the
    # outcome is judged by polling the link rather than by parsing a response.
    ssh "${SSH_OPTS[@]}" "root@$BOARD" \
        "luna-send -n 1 luna://com.webos.service.wifi/connect '{\"profileId\":$WIFI_PROFILE_ID}' >/dev/null 2>&1" \
        >/dev/null 2>>"$LOG"

    waited=0
    while [ "$waited" -lt "$WIFI_WAIT" ]; do
        sleep 5
        waited=$(( waited + 5 ))
        if wifi_up; then log "WIFI  associated after ${waited}s"; return 0; fi
    done
    return 1
}

# ---------------------------------------------------------------- preflight --
# The transport check comes first deliberately. If the route to the board does
# not leave via the cable, a reply proves nothing -- some unrelated host
# upstream can answer for $BOARD, and a wake would be sent at a board that was
# never asleep. Establish the path is right, then trust what comes back on it.

# Which link carries the transfer decides whether the capture is valid, not
# just whether it is fast. CSI is computed from received frames, so exporting
# over the measured WiFi link feeds TCP ACKs back as fresh stimulus -- a loop
# whose gain exceeds 1, ending with most of the CSI self-generated. Over the
# cable the transfer is invisible: a 400 MB copy during a capture left the
# group rate at 19.2/s with zero missing groups, identical to an idle run.
iface=$(ip route get "$BOARD" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
if [ "$iface" != "$WIRED_IFACE" ]; then
    if [ "$REQUIRE_WIRED" = "1" ]; then
        die "transfer would use '$iface', not wired '$WIRED_IFACE' -- CSI would be self-generated (set REQUIRE_WIRED=0 to override)"
    fi
    log "WARN  transfer over '$iface', not wired '$WIRED_IFACE' -- CSI may be contaminated"
fi

if ! ping -c2 -W2 "$BOARD" >/dev/null 2>&1; then
    log "board $BOARD silent on $WIRED_IFACE, trying serial wake"
    wake_board || die "board $BOARD unreachable and serial wake failed"
fi

# Reachable is not the same as ready: a board that has just woken answers ping
# from the kernel while sshd is still coming up.
ssh_ok=0
for attempt in 1 2 3 4 5 6; do
    if ssh "${SSH_OPTS[@]}" "root@$BOARD" true 2>>"$LOG"; then ssh_ok=1; break; fi
    log "ssh attempt $attempt failed, retrying in 10s"
    sleep 10
done
[ "$ssh_ok" = "1" ] || die "ssh to $BOARD failed after 6 attempts"

ensure_wifi || die "wlan0 not associated -- capture would contain no CSI"

# ------------------------------------------------------- deploy board script --
# Kept in sync by checksum so a board reboot, or an edit here, self-heals.
read -r -d '' BOARD_SRC <<'BOARDEOF'
#!/bin/sh
# Stream MTK CSI to stdout for $1 seconds. Deployed by capture_mtk_hourly.sh.
#
# stdout is the binary capture and carries nothing else -- every diagnostic
# goes to stderr, which the host appends to its log.
SECS=${1:-3300}
IVAL=${2:-0.05}
IW=/var/iwtools/iw-priv
LOCK=/var/csi/capture.lock
TOKEN=/var/csi/run_token

say() { printf '%s board: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

PING_PID=""
WATCHDOG_PID=""
cleanup() {
    [ -n "$PING_PID" ]     && kill "$PING_PID" 2>/dev/null
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
    pkill -x ping 2>/dev/null
    $IW wlan0 driver "set_csi 0" >/dev/null 2>&1
    say "disarmed"
}
# HUP matters: if the host's ssh dies the script is hung up on, and without
# this the driver would stay armed and the ping would run forever.
trap cleanup EXIT INT TERM HUP

exec 9>"$LOCK"
flock -n 9 || { say "FATAL another capture already running"; exit 3; }

[ -x "$IW" ] || { say "FATAL $IW missing"; exit 4; }

# Holding the lock proves no other capture is live, so anything still armed is
# debris from a run that was killed outright. Clearing it here rather than on
# the host side is what makes it safe: the host cannot tell "leftover" from
# "still running", and resetting the latter would quietly ruin it.
pkill -x ping 2>/dev/null
$IW wlan0 driver "set_csi 0" >/dev/null 2>&1

# The stimulus target must sit on wlan0: CSI is measured there, and only
# frames the board *receives* on that link produce records. The default route
# is not a safe source for it -- with the LAN cable plugged in, connman prefers
# ethernet and installs "default dev eth0 scope link", whose third field is an
# interface name rather than an address. So ask wlan0 directly, then fall back
# to the subnet's first host, which is where consumer APs live.
GW=${3:-}
[ -n "$GW" ] || GW=$(ip route show dev wlan0 2>/dev/null | awk '/^default .* via /{print $3; exit}')
[ -n "$GW" ] || GW=$(ip route 2>/dev/null | awk '/^default .* via .* dev wlan0/{print $3; exit}')
[ -n "$GW" ] || GW=$(ip -4 addr show wlan0 2>/dev/null \
                     | awk '/inet /{split($2,a,"/"); split(a[1],o,"."); print o[1]"."o[2]"."o[3]".1"; exit}')
case $GW in
    [0-9]*.[0-9]*.[0-9]*.[0-9]*) ;;
    *) say "FATAL no usable stimulus target on wlan0 (got '${GW:-none}')"; exit 5 ;;
esac

# Verify it answers before committing to a 55-minute run. A silent stimulus
# yields a nearly empty capture that only shows up an hour later.
if ! ping -I wlan0 -c 2 -W 2 "$GW" >/dev/null 2>&1; then
    say "FATAL stimulus target $GW does not answer on wlan0"
    exit 6
fi

say "AP $(iw dev wlan0 link 2>/dev/null \
        | awk '/SSID/{ssid=$2} /freq/{f=$2} /signal/{sig=$2} END{printf "%s %s MHz %s dBm", ssid, f, sig}')"
# "flood" is -f: ping sends as fast as replies come back, with no fixed
# period. Anything else is a period in seconds. Unquoted on purpose below --
# it has to split into two words for the -i form.
if [ "$IVAL" = "flood" ]; then
    PING_MODE="-f"
    say "stimulus $GW flooded, streaming ${SECS}s"
else
    PING_MODE="-i $IVAL"
    say "stimulus $GW every ${IVAL}s, streaming ${SECS}s"
fi
$IW wlan0 driver "set_csi 2 0 1"    >/dev/null 2>&1   # 5 GHz
$IW wlan0 driver "set_csi 2 3 0 34" >/dev/null 2>&1   # QoS data (32 = beacon)
$IW wlan0 driver "set_csi 2 5 2"    >/dev/null 2>&1   # VHT80
$IW wlan0 driver "set_csi 1"        >/dev/null 2>&1

# Both of the next two survive this script being SIGKILLed, which a trap
# cannot cover. If the host dies mid-capture the session is torn down without
# a signal we can catch, and a stimulus running forever plus an armed driver
# would burn airtime and corrupt the following hour's capture. timeout and the
# watchdog are separate processes, so they still fire once orphaned.
# Kept rather than discarded so the achieved stimulus rate can be reported.
# -q still prints the summary; it only suppresses the per-packet lines.
PING_OUT=/tmp/csi_ping_stats.$$
rm -f "$PING_OUT"
timeout $(( SECS + 30 )) ping -I wlan0 $PING_MODE -q "$GW" >"$PING_OUT" 2>&1 &
PING_PID=$!

# A watchdog outliving its own run must not disarm a later one. Surviving
# SIGKILL is the whole point of it, but a run killed outright leaves it
# ticking, and 45s later it would turn the driver off underneath whatever
# capture started meanwhile -- which is how a 120s capture came back holding
# 24s of data. The token says whose run it is; a stale watchdog sees someone
# else's and exits without touching anything.
RUN_TOKEN="$$-$(date +%s)"
echo "$RUN_TOKEN" > "$TOKEN"
setsid sh -c "sleep $(( SECS + 45 ));
    [ \"\$(cat $TOKEN 2>/dev/null)\" = '$RUN_TOKEN' ] || exit 0;
    pkill -x ping; $IW wlan0 driver 'set_csi 0'" \
    >/dev/null 2>&1 </dev/null 9>&- &
WATCHDOG_PID=$!

# The reader must start after arming and stay attached for the whole run.
{ timeout "$SECS" cat /proc/net/wlan/csi_data; } 2>/dev/null
rc=$?
# 143 (SIGTERM from timeout) is how a completed capture ends here; 124 is the
# coreutils spelling. Neither is a failure.
case $rc in
    143|124) say "stream closed normally (rc=$rc)" ;;
    *)       say "stream ended unexpectedly rc=$rc" ;;
esac

# INT makes ping print its summary and exit, which is the only way to learn how
# much stimulus actually went out -- the requested rate and the achieved one
# differ, and for flood there is no requested rate at all.
# INT makes ping print its summary and exit. Scoped to this run's own ping via
# the timeout wrapper's child, not pkill -x, which would also cut short a
# concurrent run's stimulus.
pkill -INT -P "$PING_PID" 2>/dev/null || kill -INT "$PING_PID" 2>/dev/null
# The summary is written only as ping exits, so it is not there immediately.
n=0
while [ "$n" -lt 15 ]; do
    grep -q 'packets transmitted' "$PING_OUT" 2>/dev/null && break
    sleep 0.2
    n=$(( n + 1 ))
done
if grep -q 'packets transmitted' "$PING_OUT" 2>/dev/null; then
    say "stimulus sent: $(sed -n '/packets transmitted/,$p' "$PING_OUT" | tr '\n' ' ')"
else
    # Saying so beats echoing ping's opening banner as though it were the
    # summary, which is what the unguarded sed did.
    say "stimulus stats unavailable"
fi
rm -f "$PING_OUT"
exit 0
BOARDEOF

want=$(printf '%s\n' "$BOARD_SRC" | md5sum | cut -d' ' -f1)
have=$(ssh "${SSH_OPTS[@]}" "root@$BOARD" "md5sum $BOARD_SCRIPT 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
if [ "$want" != "$have" ]; then
    printf '%s\n' "$BOARD_SRC" \
        | ssh "${SSH_OPTS[@]}" "root@$BOARD" \
              "mkdir -p $BOARD_DIR && cat > $BOARD_SCRIPT && chmod +x $BOARD_SCRIPT" \
        2>>"$LOG" || die "could not deploy $BOARD_SCRIPT"
    log "DEPLOY $BOARD_SCRIPT updated ($want)"
fi

# ----------------------------------------------------------------- capture ---
# One stamp for both artefacts, so a segment's CSI and its frames share a name
# and sort together on the collection host.
STAMP=$(date '+%Y%m%d_%H%M%S')
OUT=$CAPTURE_DIR/$STAMP.bin
FRAMES=$FRAME_TMP/$STAMP
ARCHIVE=$CAPTURE_DIR/${STAMP}_frames.tar
LIVE_PATH=$LIVE_DIR/$STAMP.bin

log "START $(basename "$OUT")  ${DURATION}s via $iface  (${free_mb}MB free)"
start=$SECONDS

# Started before the stream rather than after, so the first frame covers the
# beginning of the CSI instead of trailing it by a second. Both are given the
# same DURATION and both are wall-clock paced, so they end together.
WEBCAM_PID=""
if [ "$WEBCAM" = "1" ]; then
    if [ -e "$WEBCAM_DEV" ]; then
        "$(dirname "$0")/webcam_capture.sh" "$DURATION" "$FRAMES" >>"$LOG" 2>&1 &
        WEBCAM_PID=$!
    else
        log "WARN  $WEBCAM_DEV absent, capturing CSI without frames"
    fi
fi

# Backgrounded so the live shipper has a PID to guard on -- it needs to know
# exactly when the writer stops, and `wait` returns the same status the
# foreground form did.
ssh "${SSH_OPTS[@]}" "root@$BOARD" \
    "$BOARD_SCRIPT $DURATION $PING_INTERVAL" >"$OUT" 2>>"$LOG" &
CAPTURE_PID=$!

SHIP_PID=""
if [ "$LIVE" = "1" ]; then
    LIVE_LOG=$LOG "$(dirname "$0")/csi_live_ship.sh" \
        "$OUT" "$LIVE_PATH" "$CAPTURE_PID" >>"$LOG" 2>&1 &
    SHIP_PID=$!
fi

wait "$CAPTURE_PID"
rc=$?
elapsed=$(( SECONDS - start ))

# Let the shipper flush the tail of the file before anything moves it.
live_ok=0
if [ -n "$SHIP_PID" ]; then
    wait "$SHIP_PID" 2>/dev/null && live_ok=1
    [ "$live_ok" = 1 ] || log "WARN  live ship did not complete, batch upload will cover it"
fi

# The grabber paces itself to the same DURATION, so this normally returns at
# once. It is a real wait rather than a kill because a half-written final JPEG
# would land in the archive as a corrupt file.
if [ -n "$WEBCAM_PID" ]; then
    # Bounded, not an open-ended wait. A UVC device that drops off the bus
    # leaves its /dev node behind, so the -e check above still passes and the
    # grab blocks on a device that will never answer. That used to hold the
    # segment open indefinitely: it never archived, never uploaded, and the
    # next cron fire was skipped on the lock. CSI is the primary signal and
    # must not be held hostage by the camera that only labels it.
    grace_end=$(( SECONDS + WEBCAM_GRACE ))
    while kill -0 "$WEBCAM_PID" 2>/dev/null && [ "$SECONDS" -lt "$grace_end" ]; do
        sleep 1
    done
    if kill -0 "$WEBCAM_PID" 2>/dev/null; then
        log "WARN  webcam grabber overran ${WEBCAM_GRACE}s past the capture, killing it"
        pkill -P "$WEBCAM_PID" 2>/dev/null
        kill -TERM "$WEBCAM_PID" 2>/dev/null
        sleep 2
        pkill -9 -P "$WEBCAM_PID" 2>/dev/null
        kill -9 "$WEBCAM_PID" 2>/dev/null
    fi
    wait "$WEBCAM_PID" 2>/dev/null

    # A grab killed mid-write leaves a truncated JPEG, and only the newest file
    # can be that one. Every valid JPEG ends with the EOI marker FFD9, so an
    # ending that is not FFD9 is a partial file rather than a dark frame.
    newest=$(ls -t "$FRAMES"/*.jpg 2>/dev/null | head -1)
    if [ -n "$newest" ] \
       && [ "$(tail -c2 "$newest" | od -An -tx1 | tr -d ' \n')" != "ffd9" ]; then
        log "WARN  dropped truncated final frame $(basename "$newest")"
        rm -f "$newest"
    fi

    frame_count=$(find "$FRAMES" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
    log "FRAMES $frame_count jpg in $(basename "$FRAMES")"
fi

# Belt and braces: if ssh was killed the remote trap may not have fired, and a
# still-armed driver would poison the next hour's capture.
ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "root@$BOARD" \
    '/var/iwtools/iw-priv wlan0 driver "set_csi 0" >/dev/null 2>&1; pkill -x ping 2>/dev/null; true' \
    >/dev/null 2>&1

# ---------------------------------------------------------------- validate ---
size=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
human=$(du -h "$OUT" 2>/dev/null | cut -f1)

if [ "$size" -eq 0 ]; then
    rm -f "$OUT"
    # Frames whose CSI never arrived label nothing, and left behind they would
    # accumulate in scratch across every failed run until the disk filled.
    rm -rf "$FRAMES"
    # rc=3 is the board's own lock: a previous capture is still streaming, which
    # on the normal 55-in-60 schedule cannot happen, since a run always ends
    # before the next cron fires.
    if [ "$rc" -eq 3 ]; then
        log "SKIP  board still capturing from an earlier run"
        exit 0
    fi
    log "FAIL  $(basename "$OUT")  0 bytes after ${elapsed}s (rc=$rc)"
    exit 1
fi

# 0xAC is the record magic; a capture not starting with it is not a capture.
magic=$(head -c1 "$OUT" | od -An -tx1 | tr -d ' \n')
[ "$magic" = "ac" ] || log "WARN  $(basename "$OUT") starts with 0x$magic, expected 0xac"

frames=""
if [ -x "$VENV" ]; then
    frames=$("$VENV" - "$OUT" <<'PYEOF' 2>/dev/null
import sys
import numpy as np
sys.path.insert(0, "/home/cyphy/feitcsi_parser")
from backend.mtk import MTKIndex

i = MTKIndex(sys.argv[1])
if not i.count:
    print("0 frames")
    raise SystemExit
span = float(i.times[-1] - i.times[0])
gaps = int((np.diff(i.times) > 1.0).sum())
# num_rx is how many tpi slots a group filled. Two is the normal 2x2 grid; a
# capture of all-ones has no tx pair, so no ratio can be formed from it -- the
# whole ratio pipeline goes dark. That happened on 2026-08-21 by associating to
# the wrong AP, and cost a capture, so it is reported rather than left to be
# noticed later.
two = int((i.num_rx_arr >= 2).sum())
pct = 100.0 * two / i.count
out = (f"{i.count} frames, {span:.0f}s span, {i.count / span:.1f}/s, "
       f"{gaps} gaps>1s, {pct:.0f}% two-stream")
if pct < 50:
    out += "  <-- WARN no tx pair, ratio unavailable"
print(out)
PYEOF
)
fi

log "DONE  $(basename "$OUT")  ${human}  ${elapsed}s  rc=$rc  ${frames:-parse skipped}"

# ----------------------------------------------------------------- archive ---
# Plain tar, no compression. JPEG is already entropy-coded: measured 4.0% saved
# on a normal segment, which does not pay for the CPU. The archive exists to
# replace ~570 scp round trips with one, and that saving is unaffected.
#
# The exception is a dark or static scene, where near-uniform frames compressed
# 92.7%. If the camera is ever run blind for long stretches, revisit this.
if [ -d "$FRAMES" ]; then
    if tar -C "$FRAME_TMP" -cf "$ARCHIVE" "$STAMP" 2>>"$LOG"; then
        log "ARCHIVE $(basename "$ARCHIVE")  $(du -h "$ARCHIVE" | cut -f1)"
        # Only after the archive exists, so a failed tar leaves the frames on
        # disk to be recovered rather than deleting the sole copy.
        rm -rf "$FRAMES"
    else
        log "WARN  archiving $FRAMES failed, frames left in place"
    fi
fi

# ------------------------------------------------------------------ upload ---
# Date directory from the segment's own stamp, not from today's date: a segment
# that starts at 23:59 and is uploaded after midnight belongs with the day it
# recorded, not the day it finished.
if [ "$UPLOAD" = "1" ]; then
    day=${STAMP%%_*}
    dest=$REMOTE_DIR/$day
    if ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$dest'" 2>>"$LOG"; then
        # The live copy is already the whole file. Promote it with a remote
        # mv rather than sending the same 45 MB a second time; fall through to
        # scp if it is short or missing.
        if [ "$live_ok" = 1 ]; then
            lsz=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
            if ssh "${SSH_OPTS[@]}" "$REMOTE" \
                 "[ \"\$(stat -c %s '$LIVE_PATH' 2>/dev/null || echo 0)\" = '$lsz' ] \
                  && mv -f '$LIVE_PATH' '$dest/'" 2>>"$LOG"; then
                log "UPLOAD $(basename "$OUT") -> $REMOTE:$dest/ (promoted live copy)"
                promoted=1
            else
                log "WARN  live copy incomplete, falling back to scp"
                promoted=0
            fi
        else
            promoted=0
        fi

        for f in "$OUT" "$ARCHIVE"; do
            [ -f "$f" ] || continue
            [ "$f" = "$OUT" ] && [ "$promoted" = 1 ] && continue
            if scp -o BatchMode=yes -o LogLevel=ERROR -q "$f" "$REMOTE:$dest/" 2>>"$LOG"; then
                log "UPLOAD $(basename "$f") -> $REMOTE:$dest/"
            else
                # Left on disk deliberately. Retention prunes by age, so a
                # failed upload survives days of retries rather than being
                # lost at the end of this run.
                log "WARN  upload of $(basename "$f") failed, kept locally"
            fi
        done
    else
        log "WARN  cannot mkdir $dest on $REMOTE, skipping upload"
    fi
fi

exit 0
