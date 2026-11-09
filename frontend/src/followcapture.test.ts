import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CaptureFile } from "./api";
import { createCaptureFollower, findCapture, MAX_TRIES, MIN_OPEN_BYTES, RETRY_MS, stampOfPath } from "./followcapture";

const BIG = MIN_OPEN_BYTES * 4;
const ROOT = "/home/lg_csi/feitcsi_parser/captures/lg_csi_captures";

function file(path: string, size_bytes = BIG): CaptureFile {
  return { filename: path, path, size_bytes, mtime: 0 } as CaptureFile;
}

/** A follower over a listing the test changes, with the captures it opened. */
function setup(initial: CaptureFile[] = []) {
  const state = { files: initial };
  const opened: string[] = [];
  const list = vi.fn(async () => state.files);
  const f = createCaptureFollower(list, (c) => opened.push(c.path));
  return { state, opened, list, f };
}

const flush = () => vi.advanceTimersByTimeAsync(0);

describe("stampOfPath", () => {
  it("reads the stamp off a capture in any folder", () => {
    expect(stampOfPath(`${ROOT}/live/20261030_143000.bin`)).toBe("20261030_143000");
    expect(stampOfPath("20261030_143000.bin")).toBe("20261030_143000");
  });

  it("gives null for any other name", () => {
    expect(stampOfPath("captures/capture.dat")).toBeNull();
    expect(stampOfPath("captures/x20261030_143000.bin")).toBeNull();
    expect(stampOfPath("captures/20261030_143000_frames.bin")).toBeNull();
  });
});

describe("findCapture", () => {
  it("finds a capture by stamp, wherever it is", () => {
    const hit = file(`${ROOT}/live/20261030_143000.bin`);
    expect(findCapture([file(`${ROOT}/20261030/20261030_142900.bin`), hit], "20261030_143000")).toBe(hit);
  });

  it("does not open a file that has barely begun", () => {
    const short = file("captures/20261030_143000.bin", MIN_OPEN_BYTES - 1);
    expect(findCapture([short], "20261030_143000")).toBeUndefined();
  });

  it("takes the longer of two copies", () => {
    const partial = file(`${ROOT}/20261030/20261030_143000.bin`, BIG);
    const whole = file(`${ROOT}/live/20261030_143000.bin`, BIG * 2);
    expect(findCapture([partial, whole], "20261030_143000")).toBe(whole);
  });
});

describe("createCaptureFollower", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("opens each capture of a batch in turn, through the empty gap between them", async () => {
    const { state, opened, f } = setup();
    let path = "captures/old.bin";
    for (const n of [1, 2, 3, 4, 5]) {
      const stamp = `20261030_14000${n}`;
      const p = `captures/${stamp}.bin`;
      f.update({ live: true, stamp: null, path, failing: false });      // starting: no stamp yet
      state.files = [...state.files, file(p)];
      f.update({ live: true, stamp, path, failing: false });
      await flush();
      expect(opened.at(-1)).toBe(p);
      path = p;
      f.update({ live: true, stamp, path, failing: false });            // the view caught up
    }
    expect(opened).toHaveLength(5);
  });

  it("waits for the file to be long enough, then opens it", async () => {
    const stamp = "20261030_143000";
    const p = `captures/${stamp}.bin`;
    const { state, opened, list, f } = setup();
    f.update({ live: true, stamp, path: "captures/old.bin", failing: false });
    await flush();
    expect(opened).toEqual([]);
    state.files = [file(p, MIN_OPEN_BYTES - 1)];
    await vi.advanceTimersByTimeAsync(RETRY_MS);
    expect(opened).toEqual([]);
    state.files = [file(p)];
    await vi.advanceTimersByTimeAsync(RETRY_MS);
    expect(opened).toEqual([p]);
    await vi.advanceTimersByTimeAsync(RETRY_MS * 5);
    expect(list).toHaveBeenCalledTimes(3);
  });

  it("does nothing while Live is off, and joins the batch when Live is clicked", async () => {
    const stamp = "20261030_143000";
    const { opened, list, f } = setup([file(`captures/${stamp}.bin`)]);
    f.update({ live: false, stamp, path: "captures/old.bin", failing: false });
    await flush();
    expect(list).not.toHaveBeenCalled();
    f.update({ live: true, stamp, path: "captures/old.bin", failing: false });
    await flush();
    expect(opened).toEqual([`captures/${stamp}.bin`]);
  });

  it("stops searching when Live is paused", async () => {
    const stamp = "20261030_143000";
    const { state, opened, f } = setup();
    f.update({ live: true, stamp, path: "captures/old.bin", failing: false });
    await flush();
    f.update({ live: false, stamp, path: "captures/old.bin", failing: false });
    state.files = [file(`captures/${stamp}.bin`)];
    await vi.advanceTimersByTimeAsync(RETRY_MS * 3);
    expect(opened).toEqual([]);
  });

  it("leaves a capture picked by hand until the next one starts", async () => {
    const first = "20261030_143000";
    const next = "20261030_143100";
    const { state, opened, f } = setup([file(`captures/${first}.bin`)]);
    f.update({ live: true, stamp: first, path: "captures/old.bin", failing: false });
    await flush();
    f.update({ live: true, stamp: first, path: "captures/picked.bin", failing: false });
    await vi.advanceTimersByTimeAsync(RETRY_MS * 3);
    expect(opened).toEqual([`captures/${first}.bin`]);
    state.files = [...state.files, file(`captures/${next}.bin`)];
    f.update({ live: true, stamp: next, path: "captures/picked.bin", failing: false });
    await flush();
    expect(opened.at(-1)).toBe(`captures/${next}.bin`);
  });

  it("does not reopen the capture already on screen", async () => {
    const stamp = "20261030_143000";
    const p = `captures/${stamp}.bin`;
    const { opened, f } = setup([file(p)]);
    f.update({ live: true, stamp, path: p, failing: false });
    await flush();
    expect(opened).toEqual([]);
  });

  it("finds a capture again when its file moves out of live/, once the job is over", async () => {
    const stamp = "20261030_143000";
    const moved = `${ROOT}/20261030/${stamp}.bin`;
    const { opened, list, f } = setup([file(moved)]);
    const live = `${ROOT}/live/${stamp}.bin`;
    f.update({ live: true, stamp: null, path: live, failing: true });
    await flush();
    expect(opened).toEqual([moved]);
    expect(list).toHaveBeenCalledTimes(1);
  });

  it("looks for a failing path once, not on every update", async () => {
    const path = "captures/20261030_143000.bin";
    const { list, f } = setup([file(path)]);
    f.update({ live: true, stamp: null, path, failing: true });
    await flush();
    f.update({ live: true, stamp: null, path, failing: true });
    await flush();
    expect(list).toHaveBeenCalledTimes(1);
  });

  it("does not move a failing capture that has no stamp to a same-named file", async () => {
    const { opened, list, f } = setup([file("elsewhere/capture.dat")]);
    f.update({ live: true, stamp: null, path: "captures/capture.dat", failing: true });
    await flush();
    expect(list).not.toHaveBeenCalled();
    expect(opened).toEqual([]);
  });

  it("goes to the newest capture when a slow listing for the last one comes back late", async () => {
    const older = "20261030_143000";
    const newer = "20261030_143100";
    const files = [file(`captures/${older}.bin`), file(`captures/${newer}.bin`)];
    const replies: ((v: CaptureFile[]) => void)[] = [];
    const opened: string[] = [];
    const list = () => new Promise<CaptureFile[]>((r) => replies.push(r));
    const f = createCaptureFollower(list, (c) => opened.push(c.path));
    f.update({ live: true, stamp: older, path: "captures/old.bin", failing: false });
    f.update({ live: true, stamp: newer, path: "captures/old.bin", failing: false });
    replies[1](files);
    await flush();
    replies[0](files);
    await flush();
    expect(opened).toEqual([`captures/${newer}.bin`]);
  });

  it("gives up on a file that never appears", async () => {
    const { list, f } = setup();
    f.update({ live: true, stamp: "20261030_143000", path: "captures/old.bin", failing: false });
    await vi.advanceTimersByTimeAsync(RETRY_MS * MAX_TRIES * 2);
    expect(list).toHaveBeenCalledTimes(MAX_TRIES);
  });

  it("keeps trying through a failed listing", async () => {
    const stamp = "20261030_143000";
    const opened: string[] = [];
    const list = vi.fn<() => Promise<CaptureFile[]>>()
      .mockRejectedValueOnce(new Error("HTTP 502"))
      .mockResolvedValue([file(`captures/${stamp}.bin`)]);
    const f = createCaptureFollower(list, (c) => opened.push(c.path));
    f.update({ live: true, stamp, path: "captures/old.bin", failing: false });
    await flush();
    expect(opened).toEqual([]);
    await vi.advanceTimersByTimeAsync(RETRY_MS);
    expect(opened).toEqual([`captures/${stamp}.bin`]);
  });
});
