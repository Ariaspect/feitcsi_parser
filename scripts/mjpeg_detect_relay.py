#!/usr/bin/env python3
"""Serve a live MJPEG stream with YOLO person boxes drawn on it.

Same stdin contract as ``mjpeg_relay.py`` -- concatenated JPEGs in, HTTP
multipart out on loopback, viewable through an ssh tunnel -- with a detector
spliced into the middle. The webcam is on the laptop and the GPU is here, so
the split that makes sense is: laptop encodes and ships raw MJPEG over ssh,
this host decodes, detects, annotates, and serves.

    # on the laptop
    ffmpeg -f v4l2 -i /dev/video2 -f image2pipe -c:v mjpeg - \\
      | ssh lg '/home/lg_csi/feitcsi_parser/.venv/bin/python \\
                /home/lg_csi/feitcsi_parser/scripts/mjpeg_detect_relay.py --roi 200,280,400,480'

    # on the laptop, in another shell
    ssh -N -L 8001:127.0.0.1:8001 lg    # then open http://127.0.0.1:8001

THE FRAME-DROPPING IS THE DESIGN. Three stages run concurrently and each keeps
only the newest frame it has been handed: the reader overwrites the raw slot,
the detector overwrites the annotated slot, the server sends whatever is in
that slot when a client is ready for it. Nothing queues. A detector slower
than the camera therefore degrades into a lower frame rate on an
always-current picture, which is what you want when the picture is being used
to watch a room -- rather than into a growing backlog that drifts further from
live the longer it runs. Inference on a 640x480 frame costs ~18 ms on the
5090 against ~33 ms between frames at 30 fps, so in practice it keeps up and
the dropping is insurance, not the normal path.

A frame the detector cannot handle is served raw rather than dropped: losing
the boxes on one frame is a worse outcome than losing the picture, and a
detector that dies takes the whole feed with it.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# See cv_presence.py: absolute so an ssh invocation resolves it, CV_WEIGHTS to
# override it anywhere the models do not live there.
DEFAULT_WEIGHTS = os.environ.get("CV_WEIGHTS", "/home/lg_csi/models/yolo11n.pt")
DEFAULT_CONFIG_DIR = "/home/lg_csi/models/.ultralytics"
PERSON_CLASS = 0
BOUNDARY = "frameboundary"

# Slot for the newest raw frame off stdin, and for the newest annotated one.
# Each is guarded by its own condition so a slow client blocking on the
# annotated slot cannot stall the reader filling the raw one.
_raw = {"jpeg": None, "seq": 0}
_raw_cv = threading.Condition()
_out = {"jpeg": None, "seq": 0, "n": 0, "n_roi": 0, "ms": 0.0, "annotated": False}
_out_cv = threading.Condition()
_eof = threading.Event()
# Until the detector has loaded its weights and warmed the graph -- ten-odd
# seconds -- the reader publishes raw frames straight through, so the feed is
# live from the first frame and merely gains boxes once inference starts. The
# alternative is a blank page for the whole warmup, which reads as a broken
# stream at exactly the moment someone is checking whether it works.
_ready = threading.Event()

# The ROI is mutable at runtime. Re-fitting it is the single most common thing
# anyone needs to do -- it is a rectangle in a particular room, so it is wrong
# the moment the camera moves -- and restarting to change it would tear down
# the laptop's ffmpeg and the ssh pipe with it. GET /roi retunes it in place.
_roi_lock = threading.Lock()
_roi: tuple[float, float, float, float] | None = None
_stats = {"in": 0, "detected": 0, "dropped": 0, "failed": 0}


def reader() -> None:
    """Split the stdin byte stream on JPEG SOI/EOI markers into the raw slot."""
    buf = b""
    stdin = sys.stdin.buffer
    while True:
        chunk = stdin.read(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            i = buf.find(b"\xff\xd8")
            if i < 0:
                if len(buf) > (1 << 20):
                    buf = buf[-2:]
                break
            j = buf.find(b"\xff\xd9", i + 2)
            if j < 0:
                buf = buf[i:]
                break
            with _raw_cv:
                if _raw["jpeg"] is not None:
                    # The previous frame was never picked up by the detector.
                    _stats["dropped"] += 1
                frame = buf[i:j + 2]
                _raw["jpeg"] = frame
                _raw["seq"] += 1
                _stats["in"] += 1
                seq = _raw["seq"]
                _raw_cv.notify_all()
            if not _ready.is_set():
                publish(frame, seq, 0, 0, 0.0, False)
            buf = buf[j + 2:]
    _eof.set()
    with _raw_cv:
        _raw_cv.notify_all()
    with _out_cv:
        _out_cv.notify_all()


def anchor_of(box) -> tuple[float, float]:
    """The point that decides whether a box is 'at' the ROI.

    Bottom-centre: for a person standing or sitting, that is roughly where
    they meet the floor, and a region of interest is a place in the room
    rather than a shape on the screen. The alternative -- how much of the box
    area overlaps the rectangle -- sounds more robust and is worse, because a
    tall standing body only ever has a sliver of its area inside a
    floor-level region, so a threshold that admits a seated subject also
    admits anyone in the background.

    The caveat that bit us: when the subject is occluded, the box bottom is
    where the OCCLUDER cuts them off, not their feet. Behind a desk that can
    be 150 px high, and it moves as the silhouette changes -- which is why
    the verdict flipped when someone raised their hands. Hysteresis absorbs
    the jitter; the real fix is an ROI fitted to where the box bottom
    actually lands in that scene, which is what drawing the anchor is for.
    """
    x1, _, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, y2


def publish(jpeg: bytes, seq: int, n: int, n_roi: int, ms: float, annotated: bool) -> None:
    with _out_cv:
        _out.update(jpeg=jpeg, seq=seq, n=n, n_roi=n_roi, ms=ms, annotated=annotated)
        _out_cv.notify_all()


def detector(args) -> None:
    """Take the newest raw frame, annotate it, publish it. Repeat."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(args.model)
    names = getattr(model, "names", {}) or {}
    if names.get(PERSON_CLASS) != "person":
        print(f"error: class {PERSON_CLASS} is {names.get(PERSON_CLASS)!r}, not 'person'",
              file=sys.stderr)
        _eof.set()
        return

    # Warm the graph on a synthetic frame so the first real one is not the
    # slowest one -- cuDNN autotuning on first call costs a second or more,
    # which is otherwise paid while someone is watching.
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False,
                  device=args.device, imgsz=args.imgsz)
    _ready.set()
    with _roi_lock:
        r = _roi
    print(f"detector ready ({args.model}, conf {args.conf}, "
          f"{'roi ' + ','.join(f'{v:.0f}' for v in r) if r else 'whole frame'})",
          file=sys.stderr, flush=True)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    nonlocal_state = {"verdict": False, "want": False, "run": 0}
    last_seq = 0
    while not _eof.is_set():
        with _raw_cv:
            while _raw["jpeg"] is None and not _eof.is_set():
                _raw_cv.wait(0.5)
            if _eof.is_set() and _raw["jpeg"] is None:
                break
            jpeg, seq = _raw["jpeg"], _raw["seq"]
            _raw["jpeg"] = None          # claim it; reader may overwrite freely
        if seq == last_seq:
            continue
        last_seq = seq

        t0 = time.perf_counter()
        try:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("undecodable frame")
            res = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                                classes=[PERSON_CLASS], device=args.device,
                                verbose=False)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                boxes = [(float(a), float(b), float(c), float(d), float(e))
                         for (a, b, c, d), e in zip(xyxy, confs)]

            with _roi_lock:
                roi = _roi

            # Every annotation size below is expressed against the 480-line
            # frame these were originally tuned on. Without this a 1080p feed
            # gets 2 px boxes and unreadable labels -- correct, and useless to
            # look at, which is the whole point of this view.
            k = img.shape[0] / 480.0
            th = max(1, round(2 * k))
            fs = 0.5 * k

            n_roi = 0
            for x1, y1, x2, y2, conf in boxes:
                ax, ay = anchor_of((x1, y1, x2, y2))
                inside = roi is None or (
                    roi[0] <= ax <= roi[2] and roi[1] <= ay <= roi[3])
                n_roi += inside
                # Green inside the region of interest, grey outside it. A
                # bystander is a person and should still be boxed; what the
                # colour says is whether this box counts as the subject,
                # which is the only question the label cares about.
                colour = (0, 235, 0) if inside else (150, 150, 150)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, th)
                cv2.putText(img, f"{conf:.2f}",
                            (int(x1), max(int(14 * k), int(y1) - int(6 * k))),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, colour, th)
                # Draw the anchor, because it is the thing actually being
                # tested and it is not where people assume. Watching this dot
                # sit 40 px from the ROI edge explains a flapping verdict
                # instantly; without it the box looks fine and the answer
                # looks arbitrary.
                if roi is not None:
                    cv2.circle(img, (int(ax), int(ay)), max(3, round(5 * k)), colour, -1)
                    cv2.circle(img, (int(ax), int(ay)), max(3, round(5 * k)), (0, 0, 0), 1)

            if roi is not None:
                cv2.rectangle(img, (int(roi[0]), int(roi[1])),
                              (int(roi[2]), int(roi[3])), (255, 190, 0), max(1, th - 1))

            ms = (time.perf_counter() - t0) * 1000.0

            # Hysteresis. A single frame must not flip the verdict: the anchor
            # jitters by tens of pixels as a silhouette changes, and near an
            # ROI edge that reads as the room emptying and refilling several
            # times a second. Require the same answer args.hold times running
            # before believing it -- at 15 fps the default costs ~200 ms of
            # lag on a real transition and removes essentially all flapping.
            nonlocal_state["want"] = bool(n_roi)
            if nonlocal_state["want"] == nonlocal_state["verdict"]:
                nonlocal_state["run"] = 0
            else:
                nonlocal_state["run"] += 1
                if nonlocal_state["run"] >= args.hold:
                    nonlocal_state["verdict"] = nonlocal_state["want"]
                    nonlocal_state["run"] = 0
            held = nonlocal_state["verdict"]

            verdict = "OCCUPIED" if held else "EMPTY"
            pending = "" if held == bool(n_roi) else "?"
            banner = f"{verdict}{pending}  n={n_roi}/{len(boxes)}  {ms:.0f}ms"
            bx, by_ = int(8 * k), int(22 * k)
            cv2.putText(img, banner, (bx, by_), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62 * k, (0, 0, 0), th * 2)
            cv2.putText(img, banner, (bx, by_), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62 * k, (0, 235, 0) if held else (60, 200, 255), th)

            ok, enc = cv2.imencode(".jpg", img, encode_params)
            if not ok:
                raise ValueError("encode failed")
            _stats["detected"] += 1
            publish(enc.tobytes(), seq, len(boxes), n_roi, ms, True)
        except Exception as exc:  # noqa: BLE001 -- see module docstring
            _stats["failed"] += 1
            if _stats["failed"] <= 3:
                print(f"detect failed ({exc}); serving raw", file=sys.stderr, flush=True)
            publish(jpeg, seq, 0, 0, 0.0, False)


INDEX = b"""<!doctype html><meta charset=utf-8><title>webcam + detections</title>
<style>body{margin:0;background:#111;display:flex;align-items:center;
justify-content:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
<img src="/stream">
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send_bytes(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_bytes(INDEX, "text/html; charset=utf-8")
            return
        if self.path.startswith("/roi"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            global _roi
            if "clear" in q:
                with _roi_lock:
                    _roi = None
                self._send_bytes(b'{"roi":null}', "application/json")
                print("roi cleared -- whole frame counts", file=sys.stderr, flush=True)
                return
            try:
                vals = tuple(float(q[k][0]) for k in ("x1", "y1", "x2", "y2"))
            except (KeyError, ValueError):
                self._send_bytes(
                    b'{"error":"want ?x1=&y1=&x2=&y2= or ?clear=1"}',
                    "application/json")
                return
            # Normalise so a rectangle dragged in any direction still works.
            box = (min(vals[0], vals[2]), min(vals[1], vals[3]),
                   max(vals[0], vals[2]), max(vals[1], vals[3]))
            with _roi_lock:
                _roi = box
            import json
            self._send_bytes(json.dumps({"roi": list(box)}).encode(),
                             "application/json")
            print(f"roi set to {','.join(f'{v:.0f}' for v in box)}",
                  file=sys.stderr, flush=True)
            return
        if self.path == "/stats":
            import json
            with _out_cv:
                with _roi_lock:
                    r = list(_roi) if _roi else None
                body = json.dumps({**_stats, "n": _out["n"], "n_roi": _out["n_roi"],
                                   "ms": round(_out["ms"], 1), "roi": r}).encode()
            self._send_bytes(body, "application/json")
            return
        if self.path not in ("/stream", "/raw"):
            self.send_error(404)
            return

        slot, cond = (_out, _out_cv) if self.path == "/stream" else (_raw, _raw_cv)
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
        self.end_headers()

        last = 0
        try:
            while True:
                with cond:
                    while slot["seq"] == last and not _eof.is_set():
                        cond.wait(5.0)
                    if _eof.is_set() and slot["seq"] == last:
                        break
                    frame, last = slot["jpeg"], slot["seq"]
                if frame is None:
                    continue
                self.wfile.write(b"--%s\r\n" % BOUNDARY.encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(frame))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Relay an MJPEG stream with YOLO person boxes drawn on it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--port", type=int, default=8001, help="loopback port to serve on")
    ap.add_argument("--model", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: ultralytics picks)")
    # Annotating means decoding and re-encoding, so every served frame is a
    # second JPEG generation on top of the camera's own. At 80 the quantiser
    # coarsens from the source's 3 to 6 -- visible softening for no useful
    # saving. Measured on a capture frame: q80 is 47 KB at 39.0 dB PSNR, q90
    # is 58 KB at 42.9 dB, against a 61 KB source. So 90 buys back most of
    # the generation loss at roughly the size the camera was already sending.
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality for re-encode")
    ap.add_argument("--roi", help="x1,y1,x2,y2; boxes whose anchor is inside are counted")
    ap.add_argument("--hold", type=int, default=3, metavar="N",
                    help="frames the verdict must persist before it flips (hysteresis)")
    args = ap.parse_args()

    global _roi
    if args.roi:
        try:
            parts = [float(v) for v in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError
            _roi = (min(parts[0], parts[2]), min(parts[1], parts[3]),
                    max(parts[0], parts[2]), max(parts[1], parts[3]))
        except ValueError:
            print("error: --roi wants four comma-separated numbers", file=sys.stderr)
            return 2

    if not Path(args.model).exists():
        print(f"error: weights not found: {args.model}", file=sys.stderr)
        return 2

    os.environ.setdefault("YOLO_CONFIG_DIR", DEFAULT_CONFIG_DIR)

    srv = Server(("127.0.0.1", args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=detector, args=(args,), daemon=True).start()
    print(f"serving http://127.0.0.1:{args.port}/  (/stream annotated, /raw passthrough, "
          f"/roi?x1=&y1=&x2=&y2= or /roi?clear=1 to retune live)",
          file=sys.stderr, flush=True)

    # The reader owns the main thread for the same reason as in mjpeg_relay.py:
    # end-of-stream must end the process, or an orphan keeps holding the port.
    try:
        reader()
    except KeyboardInterrupt:
        pass
    finally:
        _eof.set()
        with _raw_cv:
            _raw_cv.notify_all()
        with _out_cv:
            _out_cv.notify_all()
        srv.shutdown()
        srv.server_close()
    s = _stats
    print(f"in {s['in']} frames, detected {s['detected']}, "
          f"dropped {s['dropped']} (detector busy), failed {s['failed']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
