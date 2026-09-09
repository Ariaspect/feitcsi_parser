#!/usr/bin/env python3
"""Label a capture's webcam frames with YOLO person detections.

The CSI side of this project infers presence from a radio channel. That
inference needs something to be checked against, and the webcam frames shot
alongside every capture are the only independent record of what was actually
in the room. This turns those frames into a machine-readable label track: one
entry per frame, carrying every person box the detector found.

What it deliberately does *not* do is decide "occupied" for you.

Two reasons. The first is that the camera sees the whole room, not just the
chair. In 20260903_195337 there are two people in frame for the entire run --
the experimental subject in the chair, and a bystander working at the desk in
the bottom-left corner. A detector asked for a boolean would have called that
run occupied from the first frame to the last, which is true of the room and
useless as a label for the subject. Boxes keep that distinction recoverable;
a boolean throws it away at the one moment it mattered. Pass ``--roi`` to
score only the region the subject occupies, and note that the ROI is recorded
in the sidecar rather than baked into the numbers, so a later change of mind
costs a re-derivation and not a re-run.

The second reason is that a sitting person is the hard case for a detector
trained on COCO. Sitting bodies are occluded by desks, foreshortened, and
often facing away, so confidence runs lower than for a standing figure and
the useful threshold is scene-specific. Storing the confidence per box lets
that threshold be chosen after looking at the distribution, instead of being
guessed before.

TIME IS THE SUBTLE PART. Frame filenames come from ``webcam_capture.sh``,
which stamps them with the *laptop's* local clock -- KST. This host is a
container running UTC, so a naive parse silently places every frame nine
hours from where it belongs, and it does so without erroring: the strings
look perfectly well-formed either way. That is the same string-versus-epoch
trap that the board's PKT clock sets elsewhere in this pipeline. So the
source timezone is an explicit argument with a KST default, every frame
carries a resolved UTC epoch, and the zone used is written into the sidecar
so a reader never has to infer which clock produced a number.

Usage:

    scripts/cv_presence.py captures/lg_csi_captures/20260903/20260903_195337_frames.tar
    scripts/cv_presence.py .../20260904_*_frames.tar --roi 200,280,400,480
    scripts/cv_presence.py .../frames_dir --conf 0.3 --save-annotated 12

Writes ``<stamp>_cv.json`` beside the input.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA = "cv_presence/1"

# COCO class 0. Nothing else in the taxonomy is a person, so this is a
# constant rather than a lookup -- but it is asserted against the loaded
# model's own names table at startup, because a custom-trained weights file
# would renumber it and the failure would otherwise be silent mislabelling.
PERSON_CLASS = 0

# Absolute, and overridable by env. The default is where the weights live on
# the collection host, which is the only machine that runs this in anger --
# scripts/label_frames.sh drives it over ssh, where nothing sources a profile
# and a relative path would resolve against whatever ssh picked as $HOME. But
# a hardcoded path also makes the script unrunnable anywhere else, so CV_WEIGHTS
# overrides it, matching the CV_PY/CV_SCRIPT convention label_frames.sh already
# uses for the same reason.
DEFAULT_WEIGHTS = os.environ.get("CV_WEIGHTS", "/home/lg_csi/models/yolo11n.pt")

# The capture host is a container on an ephemeral overlay; /home/lg_csi is the
# Windows-backed bind mount that survives a rebuild. Weights belong there, and
# so does ultralytics' own config, which it otherwise scatters into /tmp.
DEFAULT_CONFIG_DIR = "/home/lg_csi/models/.ultralytics"

# Frame names are YYYYMMDD_HHMMSS_mmm.jpg -- 19 characters. The trailing field
# is date's %N truncated to three digits by the capture script, so it is
# milliseconds, not some other fraction.
STAMP_LEN = 19


def parse_frame_time(name: str, tz: ZoneInfo) -> tuple[str, float]:
    """Return (ISO-8601 string, UTC epoch seconds) for a frame filename.

    Raises ValueError on anything that does not match, rather than guessing.
    A frame we cannot place in time is worse than a missing frame: it lands
    somewhere plausible and corrupts the alignment silently.
    """
    stem = Path(name).stem
    if len(stem) != STAMP_LEN:
        raise ValueError(f"unexpected frame name {name!r} (stem {stem!r})")
    naive = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")
    millis = int(stem[16:19])
    aware = naive.replace(tzinfo=tz) + timedelta(milliseconds=millis)
    return aware.isoformat(), aware.timestamp()


def resolve_tz(spec: str):
    """Resolve a zone name, falling back to a literal UTC offset.

    A bare container image often ships no tzdata, so ZoneInfo("Asia/Seoul")
    raises even though the name is perfectly valid. The tzdata wheel is a
    declared dependency and normally covers it, but accepting "+09:00" too
    means this script still runs somewhere that has neither -- and an explicit
    offset in the sidecar is no less precise than a zone name for a country
    that has not observed DST since 1988.
    """
    if spec and spec[0] in "+-":
        try:
            hours, _, minutes = spec[1:].partition(":")
            delta = timedelta(hours=int(hours), minutes=int(minutes or 0))
        except ValueError:
            raise ValueError(f"bad UTC offset {spec!r}; want e.g. +09:00") from None
        return timezone(-delta if spec[0] == "-" else delta)
    try:
        return ZoneInfo(spec)
    except Exception:
        raise ValueError(
            f"unknown timezone {spec!r} -- install tzdata (uv sync --group cv) "
            f"or pass a literal offset such as +09:00"
        ) from None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def frame_source(path: Path):
    """Yield (list_of_names, reader) for a tar, a tar.zst, or a directory.

    A .tar.zst is decompressed to a scratch .tar first rather than streamed.
    Streaming a zstd pipe gives sequential access only, and tar member order
    is *not* chronological -- the archives here come out shuffled. Sorting
    needs random access, and a temporary file buys it for the cost of one
    decompress of a file that is under 100 MB in practice.
    """
    tmpdir = None
    try:
        if path.is_dir():
            names = sorted(p.name for p in path.iterdir() if p.suffix.lower() == ".jpg")

            def read_dir(n: str) -> bytes:
                return (path / n).read_bytes()

            yield names, read_dir
            return

        tar_path = path
        if path.name.endswith(".tar.zst"):
            if not shutil.which("zstd"):
                sys.exit("error: .tar.zst input needs the zstd binary on PATH")
            tmpdir = tempfile.mkdtemp(prefix="cv_presence_")
            tar_path = Path(tmpdir) / "frames.tar"
            with open(tar_path, "wb") as out:
                subprocess.run(["zstd", "-dc", str(path)], stdout=out, check=True)

        with tarfile.open(tar_path, "r:") as tf:
            members = {
                m.name: m
                for m in tf.getmembers()
                if m.isfile() and m.name.lower().endswith(".jpg")
            }
            names = sorted(members, key=lambda n: Path(n).name)

            def read_tar(n: str) -> bytes:
                fh = tf.extractfile(members[n])
                if fh is None:
                    raise OSError(f"unreadable member {n}")
                return fh.read()

            yield names, read_tar
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def in_roi(box: list[float], roi: tuple[float, float, float, float] | None) -> bool:
    """True if the box's centre-bottom lies inside the ROI.

    Centre-bottom, not centroid or overlap: for a seated person the box top
    wanders with posture and the sides with arms, but where they meet the
    chair is stable. It is also what separates a subject in the chair from a
    bystander at the desk whose box may overlap the same columns higher up.
    """
    if roi is None:
        return True
    x1, y1, x2, y2 = box[:4]
    cx, by = (x1 + x2) / 2.0, y2
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= by <= ry2


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Label capture webcam frames with YOLO person detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("source", type=Path, help="_frames.tar, _frames.tar.zst, or a directory of JPEGs")
    ap.add_argument("-o", "--out", type=Path, help="output JSON (default: <stamp>_cv.json beside source)")
    ap.add_argument("--model", default=DEFAULT_WEIGHTS, help="YOLO weights")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence floor for a kept box")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (frames are 640x480)")
    ap.add_argument("--batch", type=int, default=32, help="frames per forward pass")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: ultralytics picks)")
    ap.add_argument("--tz", default="Asia/Seoul", help="timezone the frame filenames were stamped in")
    ap.add_argument("--roi", help="x1,y1,x2,y2 in pixels; boxes scored inside it only")
    ap.add_argument("--save-annotated", type=int, default=0, metavar="N",
                    help="also write N evenly spaced annotated JPEGs for eyeballing")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames (smoke test)")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite an existing sidecar")
    args = ap.parse_args()

    if not args.source.exists():
        return _fail(f"no such source: {args.source}")

    # Name the sidecar off the capture stamp, not the archive filename, so a
    # .tar and a .tar.zst of the same run land on the same output path.
    base = args.source.name
    for suffix in ("_frames.tar.zst", "_frames.tar"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    else:
        base = args.source.name.rstrip("/")
    out_path = args.out or args.source.parent / f"{base}_cv.json"
    if out_path.exists() and not args.force:
        return _fail(f"{out_path} exists; pass --force to overwrite")

    roi = None
    if args.roi:
        try:
            parts = [float(v) for v in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError
            roi = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            return _fail("--roi wants four comma-separated numbers: x1,y1,x2,y2")

    try:
        tz = resolve_tz(args.tz)
    except Exception as exc:
        return _fail(f"{exc}")

    os.environ.setdefault("YOLO_CONFIG_DIR", DEFAULT_CONFIG_DIR)
    with contextlib.suppress(OSError):
        Path(DEFAULT_CONFIG_DIR).mkdir(parents=True, exist_ok=True)

    weights = Path(args.model)
    if not weights.exists():
        return _fail(f"weights not found: {weights}")

    from ultralytics import YOLO  # imported late; it costs seconds and pulls torch

    model = YOLO(str(weights))
    names = getattr(model, "names", {}) or {}
    if names.get(PERSON_CLASS) != "person":
        return _fail(
            f"class {PERSON_CLASS} of these weights is {names.get(PERSON_CLASS)!r}, not 'person' -- "
            "these are not the COCO weights this script assumes"
        )

    annot_dir = None
    if args.save_annotated:
        annot_dir = out_path.parent / f"{base}_cv_annotated"
        annot_dir.mkdir(exist_ok=True)

    frames: list[dict] = []
    started = time.time()

    with frame_source(args.source) as (frame_names, read):
        if args.limit:
            frame_names = frame_names[: args.limit]
        if not frame_names:
            return _fail(f"no .jpg frames found in {args.source}")

        annot_at: set[int] = set()
        if args.save_annotated:
            n = min(args.save_annotated, len(frame_names))
            step = len(frame_names) / n
            annot_at = {int(i * step) for i in range(n)}

        import cv2
        import numpy as np

        for start in range(0, len(frame_names), args.batch):
            chunk = frame_names[start : start + args.batch]
            imgs, kept = [], []
            for name in chunk:
                buf = np.frombuffer(read(name), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    # A truncated JPEG is a real occurrence here: the capture
                    # script kills a hung grab mid-write. Record the gap
                    # rather than dropping the frame, so a reader can tell
                    # "no person" from "no picture".
                    iso, epoch = parse_frame_time(name, tz)
                    frames.append({"name": Path(name).name, "ts": iso,
                                   "epoch": round(epoch, 3), "error": "undecodable"})
                    continue
                imgs.append(img)
                kept.append(name)

            if not imgs:
                continue

            results = model.predict(
                imgs, conf=args.conf, imgsz=args.imgsz, classes=[PERSON_CLASS],
                device=args.device, verbose=False,
            )

            for idx, (name, res) in enumerate(zip(kept, results)):
                iso, epoch = parse_frame_time(name, tz)
                boxes = []
                if res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), c in zip(xyxy, confs):
                        boxes.append([round(float(x1), 1), round(float(y1), 1),
                                      round(float(x2), 1), round(float(y2), 1),
                                      round(float(c), 3)])
                boxes.sort(key=lambda b: -b[4])
                inside = [b for b in boxes if in_roi(b, roi)]
                frames.append({
                    "name": Path(name).name,
                    "ts": iso,
                    "epoch": round(epoch, 3),
                    "n": len(boxes),
                    "n_roi": len(inside),
                    "max_conf": inside[0][4] if inside else 0.0,
                    "boxes": boxes,
                })
                if annot_dir is not None and (start + idx) in annot_at:
                    cv2.imwrite(str(annot_dir / Path(name).name), res.plot())

    elapsed = time.time() - started
    frames.sort(key=lambda f: f["epoch"])

    scored = [f for f in frames if "error" not in f]
    occupied = [f for f in scored if f["n_roi"] > 0]
    gaps = _gaps(scored)
    doc = {
        "schema": SCHEMA,
        "source": str(args.source),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": args.tz,
        "roi": list(roi) if roi else None,
        "model": {
            "weights": weights.name,
            "sha256": sha256(weights),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device or "auto",
            "class": "person",
        },
        "summary": {
            "frames": len(frames),
            "undecodable": len(frames) - len(scored),
            "frames_with_person": len(occupied),
            "fraction_occupied": round(len(occupied) / len(scored), 4) if scored else 0.0,
            "max_persons_in_frame": max((f["n"] for f in scored), default=0),
            "span_s": round(scored[-1]["epoch"] - scored[0]["epoch"], 3) if scored else 0.0,
            "gaps_over_2s": gaps,
            "seconds": round(elapsed, 1),
        },
        "frames": frames,
    }

    out_path.write_text(json.dumps(doc, indent=1))

    s = doc["summary"]
    rate = len(frames) / elapsed if elapsed else 0.0
    print(f"{out_path}")
    print(f"  {s['frames']} frames in {elapsed:.1f}s ({rate:.0f} fps), "
          f"{s['undecodable']} undecodable")
    print(f"  person in {s['frames_with_person']}/{len(scored)} "
          f"({s['fraction_occupied'] * 100:.1f}%), max {s['max_persons_in_frame']} in frame"
          + (f", ROI {args.roi}" if roi else ", whole frame"))
    if gaps:
        print(f"  {len(gaps)} gap(s) over 2s in the frame track: "
              + ", ".join(f"{g['after']}+{g['gap_s']}s" for g in gaps[:4]))
    if annot_dir is not None:
        print(f"  annotated samples: {annot_dir}")
    return 0


def _gaps(scored: list[dict]) -> list[dict]:
    """Frame-track holes over 2s.

    The capture runs at 1 fps with an absolute-deadline loop, so a hole means
    grabs failed -- a camera that stopped answering. It bounds how much of the
    label track is actually evidence, which matters before trusting a run.
    """
    out = []
    for prev, cur in zip(scored, scored[1:]):
        gap = cur["epoch"] - prev["epoch"]
        if gap > 2.0:
            out.append({"after": prev["name"], "gap_s": round(gap, 1)})
    return out


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
