#!/bin/bash
# Ship a growing capture to the collection host while it is still being written.
#
# The batch path (scp after the segment ends) is unchanged and remains the
# authority: this is an additive live copy so a viewer elsewhere can watch the
# capture grow. Losing the live copy costs a live view, never the dataset.
#
#   csi_live_ship.sh LOCAL_FILE REMOTE_PATH [GUARD_PID]
#
# Resumes from the REMOTE file's own size rather than a local bookmark, so a
# dropped link cannot duplicate or skip bytes: whatever the server actually has
# is the offset we continue from. Exits when GUARD_PID is gone and local and
# remote agree.
#
# Radio note: this streams over the laptop's uplink (wlo1, 5620 MHz), which is
# 320 MHz clear of the 5240 MHz channel the board measures. It does NOT go over
# the board's own link -- that would feed ACKs back as stimulus, the feedback
# loop REQUIRE_WIRED exists to prevent.
set -u

LOCAL=${1:?usage: csi_live_ship.sh LOCAL_FILE REMOTE_PATH [GUARD_PID]}
REMOTE_PATH=${2:?missing REMOTE_PATH}
GUARD_PID=${3:-}

REMOTE=${REMOTE:-lg}
POLL=${POLL:-1}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15
          -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new
          -o LogLevel=ERROR)
LOG=${LIVE_LOG:-/dev/stderr}

log() { printf '%s  ship: %s\n' "$(date '+%H:%M:%S')" "$*" >>"$LOG"; }

remote_size() {
    ssh "${SSH_OPTS[@]}" "$REMOTE" "stat -c %s '$REMOTE_PATH' 2>/dev/null || echo 0" 2>/dev/null
}
local_size() { stat -c %s "$LOCAL" 2>/dev/null || echo 0; }
guard_alive() { [ -n "$GUARD_PID" ] && kill -0 "$GUARD_PID" 2>/dev/null; }

ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$(dirname "$REMOTE_PATH")'" 2>>"$LOG" \
    || { log "FATAL cannot create remote dir"; exit 1; }

backoff=1
while :; do
    off=$(remote_size)
    case "$off" in ''|*[!0-9]*) off=0 ;; esac

    lsz=$(local_size)
    if [ "$off" -gt "$lsz" ]; then
        # Remote longer than local: a stale file from an earlier run under the
        # same name. Refuse rather than append corruption to it.
        log "FATAL remote $REMOTE_PATH is $off bytes, local only $lsz -- stale remote, not appending"
        exit 1
    fi

    if ! guard_alive && [ "$off" -eq "$lsz" ] && [ "$lsz" -gt 0 ]; then
        log "done, $off bytes shipped"
        exit 0
    fi

    # -F reopens on rotation; --pid makes tail exit on its own once the writer
    # is gone, so the pipeline drains and closes instead of hanging forever.
    if [ -n "$GUARD_PID" ]; then
        tail -c "+$((off + 1))" -F --pid="$GUARD_PID" "$LOCAL" 2>/dev/null \
            | ssh "${SSH_OPTS[@]}" "$REMOTE" "cat >> '$REMOTE_PATH'" 2>>"$LOG"
    else
        tail -c "+$((off + 1))" -F "$LOCAL" 2>/dev/null \
            | ssh "${SSH_OPTS[@]}" "$REMOTE" "cat >> '$REMOTE_PATH'" 2>>"$LOG"
    fi
    rc=$?

    if guard_alive; then
        log "link dropped (rc=$rc), resuming in ${backoff}s"
        sleep "$backoff"
        backoff=$(( backoff < 8 ? backoff * 2 : 8 ))
    else
        # Writer gone: one more pass to flush the tail, then the loop's
        # equality check above ends it.
        sleep 1
        backoff=1
    fi
done
