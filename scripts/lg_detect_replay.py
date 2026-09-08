#!/usr/bin/env python3
"""Replay the LG on-board detector over a recorded capture, and emit JSON.

Runs UNDER THE BOARD'S INTERPRETER (.venv-board, numpy 1.26.4), not the
project venv, and that is not a detail. Its TLV length arithmetic is

    field_length = np.uint16(in_bytes[index + 1] + (in_bytes[index + 2] << 8))

on a uint8 array. Under NumPy 1.x the shift promotes to int and yields 512 for
tags 8 and 9; under NumPy 2.x (NEP 50) it stays uint8, evaluates to 0, and the
TLV walk desynchronises at the first CSI field -- silently, producing frames
whose imaginary parts are all zero rather than an error. So a replay on the
project venv would measure the NumPy version, not the detector.

The detector itself is imported and called, never reimplemented:
mtk_read_bf_csi and process_csi_data run exactly as written. Two things are
substituted, both forced by replaying a file instead of a live device:

  the byte source  a capture rather than /proc/net/wlan/csi_data
  the clock        its 10 s absence rule is in seconds, and a replay has no
                   wall time worth measuring against, so the capture's own
                   tag-2 timestamps drive it

The frame-splitting loop below is copied from its main(), including the
`frame_start += length` step that advances three bytes short -- the re-search
for 0xAC masks the drift, and reproducing it is the point.

    lg_detect_replay.py CAPTURE [--threshold 26] [--absence 10]
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "backend", "vendor", "csi_dump_parsing.py")


def load_detector():
    spec = importlib.util.spec_from_file_location("csi_dump_parsing", VENDOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Clock:
    """Its time.time()/time.sleep(), driven by the capture.

    sleep() only advances the clock. Live it blocks the reader for 2 s and the
    CSI arriving meanwhile is lost, so the replay is mildly GENEROUS: it sees
    frames the live detector would have missed.
    """

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def replay(path, threshold, absence):
    mod = load_detector()
    mod.threshold = float(threshold)
    mod.absence_duration = float(absence)
    mod.current_status = False
    clock = Clock()
    mod.time = clock

    with open(path, "rb") as handle:
        buf = handle.read()

    data_buffer = b""
    previous = None
    last_movement = None
    absence_reported = False
    events = []
    t0 = None
    records = 0
    parse_failures = 0

    pos = 0
    while pos < len(buf):
        data_buffer += buf[pos:pos + 1024]      # the same 1024-byte reads
        pos += 1024
        data = np.frombuffer(data_buffer, dtype=np.uint8)
        frame_start = 0
        while frame_start < len(data):
            magic_pos = data[frame_start:].tobytes().find(b"\xac")
            if magic_pos == -1:
                data_buffer = data[frame_start:].tobytes()
                break
            frame_start += magic_pos
            if frame_start + 2 >= len(data):
                data_buffer = data[frame_start:].tobytes()
                break
            length = int.from_bytes(
                data[frame_start + 1:frame_start + 3], byteorder="little"
            )
            if length == 0 or frame_start + length > len(data):
                data_buffer = data[frame_start:].tobytes()
                break
            frame_data = data[frame_start + 3:frame_start + length]
            frame_start += length
            records += 1

            # Advance the clock to this record before its logic runs, so the
            # absence rule is measured in capture seconds.
            try:
                stamp = mod.mtk_read_bf_csi(frame_data)["timestamp_low"]
                seconds = float(stamp) / 1000.0
                if t0 is None:
                    t0 = seconds
                if seconds >= t0:
                    clock.now = max(clock.now, seconds - t0)
            except Exception:
                parse_failures += 1

            with contextlib.redirect_stdout(io.StringIO()):   # it prints "+"/"-"
                previous, moved = mod.process_csi_data(frame_data, previous)
            if moved:
                events.append({"kind": "+", "t": round(clock.now, 3)})
                last_movement = clock.now
                absence_reported = False

        now = clock.time()
        if last_movement is not None and not absence_reported:
            if now - last_movement >= mod.absence_duration:
                events.append({"kind": "-", "t": round(now, 3)})
                mod.current_status = False
                absence_reported = True

    return {
        "events": events,
        "records": records,
        "parseFailures": parse_failures,
        "duration": round(clock.now, 3),
        "threshold": float(threshold),
        "absenceDuration": float(absence),
        "numpy": np.__version__,
        "python": "%d.%d.%d" % sys.version_info[:3],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--threshold", type=float, default=26.0)
    ap.add_argument("--absence", type=float, default=10.0)
    args = ap.parse_args()
    json.dump(replay(args.capture, args.threshold, args.absence), sys.stdout)
