# Capture controls: collecting from the laptop at the board

The experiment runs from the laptop wired to the LG board (cable on `enp1s0`,
serial console in the tmux session `tio`). The capture controls sit in the web
app's toolbar, next to the MIMO picker, always visible: three numbers (length,
delay, count), two buttons (Start, Stop), a status chip, four preflight
dots, and a ▾ that unfolds the details panel (scenario fields, per-capture
results, the script's log) under the header. Served on that laptop they work
with neither a shell nor an online connection; served on lg they forward to
the laptop (see "Using the controls from lg's web").

## What a button does

| control | what happens |
|---|---|
| **Start** | after *Delay* seconds, *Count* runs of `scripts/capture_mtk_hourly.sh --duration <Len>` back to back, *Gap between captures* (details panel) seconds apart; the batch stops at the first failure. Count 1 is one capture |
| **Stop** | ends the current capture and the rest of the batch (see below) |
| Delay s | the countdown while the operator leaves the room; the chip counts it down |
| (stimulus) | not a control: fixed at one ping every **0.02 s (≈43 Hz of CSI)**, the dataset's rate; the script's own default would be 0.05 s (≈19 Hz) |
| Room / Scenario / Note (details panel) | written to `<stamp>_meta.json` beside the capture, uploaded with it; blank fields are not written |
| upload to lg (details panel) | off keeps everything local (`UPLOAD=0 LIVE=0`); on, an unreachable lg only produces `WARN` lines and the files wait locally |
| label frames on lg (details panel) | on runs the YOLO labelling on lg after each capture (`CV=1`); it adds up to a minute between captures |

The capture itself is unchanged: `capture_mtk_hourly.sh` does what it does for
cron and for the shell. The web only runs it, with three differences:

* `PING_INTERVAL=0.02`, fixed (the script's own default is 0.05);
* `RETAIN_DAYS=0`: no pruning of local captures. The script's default removes
  local files older than seven days, which is safe only when every capture has
  been uploaded, and offline sessions are exactly when it has not.
* each run is its own process group, so **Stop** kills the wrapper, the webcam
  grabber, the live shipper and both ssh connections with one signal;
* after **Stop**, the board is cleaned explicitly (the PIDs holding
  `/var/csi/capture.lock` are killed and the driver disarmed), and the partial
  `.bin`, its frames and lg's `live/` copy are removed. A capture that already
  finished is kept.

## Where the files go

Exactly where they went before: `/home/cyphy/feitcsi_parser/captures/<stamp>.bin`,
`<stamp>_frames.tar`, `<stamp>_cv.json` (when labelled), `<stamp>_meta.json`
(when a scenario was given), then `lg:/home/lg_csi/lg_csi_captures/<day>/`
when lg is reachable. The server reports the directory it uses in the status.

## Status and preflight

The page polls `/api/capture/status` once a second; the details panel shows the
script's own log (`mtk_capture.log`) from the moment the job started. Four
checks run in the background and are shown as dots in the toolbar (green good,
red not, grey unknown) and as words in the details panel:

* **board** answers a ping on the cable. A silent board is woken over the
  serial console by the script itself (about three minutes); the state shows
  *waking*.
* **webcam** `/dev/video2` is free. A busy camera (the live stream) means the
  capture records CSI without frames, which the log says.
* **lg** reachable over ssh. Offline is allowed; uploads then fail softly.
* **capture lock** `/tmp/mtk-capture.lock` free. Held means a capture is
  running from cron or a shell; the server refuses to start a second one.

## Running the web app on the laptop

```
# once
git worktree add /home/cyphy/feitcsi_parser-serve main      # or any checkout of main
cd /home/cyphy/feitcsi_parser-serve && ln -s /home/cyphy/feitcsi_parser/captures captures
ln -s /home/cyphy/feitcsi_parser/frontend/node_modules frontend/node_modules
npm --prefix frontend run build
cp scripts/csi-web.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now csi-web && loginctl enable-linger $USER

# then
xdg-open http://localhost:8002
```

Without systemd: `HOST=127.0.0.1 PORT=8002 pnpm serve` (or the uvicorn line in
the unit file) from the checkout. The server binds to localhost only; nothing
else should be able to start a capture.

After pulling a newer main: `npm --prefix frontend run build && systemctl --user restart csi-web`.
A restart while a capture is running kills that capture's controller, not the
capture script, so wait for the job to finish first.

## Using the controls from lg's web (online)

lg cannot reach the laptop, so the laptop keeps a reverse tunnel open
(`scripts/csi-web-tunnel.service`: lg's `127.0.0.1:18002` → the laptop's
`:8002`). A server on lg started with `CAPTURE_PROXY_URL=http://127.0.0.1:18002`
forwards `/api/capture/*` to the laptop, and its toolbar controls the same
captures; the status says which host runs them. When the tunnel is down the
chip reads "controller unreachable" (503 "capture laptop not reachable"). Offline, use the laptop's own
`http://localhost:8002`.

## API

```
GET  /api/capture/status
POST /api/capture/start   {"duration_s": 60, "delay_s": 10, "count": 10, "gap_s": 0,
                           "room": "env2", "scenario": "static sitting", "note": "...",
                           "upload": true, "label": true}
POST /api/capture/stop
```

`start` answers 400 for a value out of range (duration 10–3300 s, delay and gap
0–3600 s, count 1–200) and 409 when a job is already running or the capture
lock is held elsewhere. Environment knobs for the server: `CAPTURE_SCRIPT`,
`CAPTURE_DIR`, `CAPTURE_BOARD` (empty skips the board cleanup), `CAPTURE_REMOTE`
(empty disables lg), `CAPTURE_REMOTE_DIR`, `CAPTURE_LOCK`.
