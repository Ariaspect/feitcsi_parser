import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TileScheduler } from "./tilefetch";

type Key = { t0: number; t1: number };

/** A run() whose promises are resolved by the test, one at a time. */
function deferredRuns() {
  const calls: { key: Key; signal: AbortSignal; settle: (v: string) => void; fail: (e: unknown) => void }[] = [];
  const run = (key: Key, signal: AbortSignal) =>
    new Promise<string>((resolve, reject) => {
      calls.push({ key, signal, settle: resolve, fail: reject });
      signal.addEventListener("abort", () => {
        const err = new Error("aborted");
        err.name = "AbortError";
        reject(err);
      });
    });
  return { calls, run };
}

function abortError() {
  const e = new Error("aborted");
  e.name = "AbortError";
  return e;
}

const sameKey = (a: Key, b: Key) => a.t0 === b.t0 && a.t1 === b.t1;

describe("TileScheduler", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("debounces a burst of requests into one run", async () => {
    const { calls, run } = deferredRuns();
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 1 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("poll");
    s.request("poll");
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(1);
  });

  it("skips a run whose key matches the last one issued", async () => {
    const { calls, run } = deferredRuns();
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 1 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    calls[0].settle("tile");
    await vi.advanceTimersByTimeAsync(0);
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(1);
  });

  // The bug: a live capture re-renders every poll with a slightly newer
  // window. Aborting the in-flight tile on every one of those starves the
  // panel whenever a tile takes longer than the poll period.
  it("does not abort an in-flight request for poll-driven refetches", async () => {
    const { calls, run } = deferredRuns();
    const delivered: string[] = [];
    let t1 = 10;
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1 }),
      sameKey,
      run,
      deliver: (_k, v) => delivered.push(v),
      debounceMs: 100,
    });

    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(1);

    // Ten polls land while the first tile is still in flight (poll period 300,
    // tile latency ~3s). None of them may abort it.
    for (let i = 0; i < 10; i++) {
      t1 += 0.3;
      s.request("poll");
      await vi.advanceTimersByTimeAsync(300);
    }
    expect(calls).toHaveLength(1);
    expect(calls[0].signal.aborted).toBe(false);

    calls[0].settle("first");
    await vi.advanceTimersByTimeAsync(0);
    expect(delivered).toEqual(["first"]);

    // Exactly one follow-up, coalesced onto the newest window — not ten.
    expect(calls).toHaveLength(2);
    expect(calls[1].key.t1).toBeCloseTo(13, 10);
  });

  it("aborts an in-flight request for a user-driven refetch", async () => {
    const { calls, run } = deferredRuns();
    let t1 = 10;
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    t1 = 20;
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls[0].signal.aborted).toBe(true);
    expect(calls).toHaveLength(2);
    expect(calls[1].key.t1).toBe(20);
  });

  it("a user gesture during the debounce window still aborts", async () => {
    const { calls, run } = deferredRuns();
    let t1 = 10;
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    t1 = 20;
    s.request("poll");
    s.request("user"); // user wins the merged reason
    await vi.advanceTimersByTimeAsync(100);
    expect(calls[0].signal.aborted).toBe(true);
    expect(calls).toHaveLength(2);
  });

  // An aborted window was never drawn, so an identical follow-up request must
  // not be deduped away: otherwise a capture that stops growing leaves the
  // panel on stale pixels with nothing in flight.
  it("clears the dedup key when a request is aborted", async () => {
    const { calls, run } = deferredRuns();
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 10 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    s.abortInFlight();
    await vi.advanceTimersByTimeAsync(0);
    s.request("poll"); // same key as the aborted request
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(2);
  });

  it("clears the dedup key when a request fails", async () => {
    const { calls, run } = deferredRuns();
    const errors: unknown[] = [];
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 10 }),
      sameKey,
      run,
      deliver: () => {},
      onError: (e) => errors.push(e),
      debounceMs: 100,
    });
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    calls[0].fail(new Error("500"));
    await vi.advanceTimersByTimeAsync(0);
    expect(errors).toHaveLength(1);
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(2);
  });

  it("drops a stale response that resolves after a newer request started", async () => {
    const { calls, run } = deferredRuns();
    const delivered: string[] = [];
    let t1 = 10;
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1 }),
      sameKey,
      run,
      deliver: (_k, v) => delivered.push(v),
      debounceMs: 100,
    });
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    t1 = 20;
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    // The aborted request resolves anyway — abort() is not instantaneous.
    calls[0].settle("stale");
    await vi.advanceTimersByTimeAsync(0);
    calls[1].settle("fresh");
    await vi.advanceTimersByTimeAsync(0);
    expect(delivered).toEqual(["fresh"]);
  });

  it("plan() returning null issues no request", async () => {
    const { calls, run } = deferredRuns();
    const s = new TileScheduler<Key, string>({
      plan: () => null,
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(0);
  });

  it("reset() aborts and lets an identical key be refetched", async () => {
    const { calls, run } = deferredRuns();
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 10 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    calls[0].settle("tile");
    await vi.advanceTimersByTimeAsync(0);
    s.reset();
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(2);
  });

  it("dispose() aborts in flight work and ignores later requests", async () => {
    const { calls, run } = deferredRuns();
    const delivered: string[] = [];
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 10 }),
      sameKey,
      run,
      deliver: (_k, v) => delivered.push(v),
      debounceMs: 100,
    });
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    s.dispose();
    expect(calls[0].signal.aborted).toBe(true);
    await vi.advanceTimersByTimeAsync(0);
    s.request("poll");
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toHaveLength(1);
    expect(delivered).toEqual([]);
  });

  it("a deferred poll that resolves to the same window does not loop", async () => {
    const { calls, run } = deferredRuns();
    let t1 = 10;
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1 }),
      sameKey,
      run,
      deliver: () => {},
      debounceMs: 100,
    });
    s.request("poll");
    await vi.advanceTimersByTimeAsync(100);
    t1 = 11;
    s.request("poll"); // deferred behind the in-flight request
    await vi.advanceTimersByTimeAsync(100);
    t1 = 10; // window slid back to what request #1 already asked for
    calls[0].settle("first");
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(1);
  });

  it("keeps following live at the backend's pace, not the poll's", async () => {
    const { calls, run } = deferredRuns();
    const delivered: string[] = [];
    let t1 = 10;
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1 }),
      sameKey,
      run,
      deliver: (_k, v) => delivered.push(v),
      debounceMs: 100,
    });
    // Poll every 300ms; each tile takes ~1.5s. Every tile must still land.
    for (let cycle = 0; cycle < 3; cycle++) {
      for (let i = 0; i < 5; i++) {
        t1 += 0.3;
        s.request("poll");
        await vi.advanceTimersByTimeAsync(300);
      }
      const pending = calls[calls.length - 1];
      expect(pending.signal.aborted).toBe(false);
      pending.settle(`tile${cycle}`);
      await vi.advanceTimersByTimeAsync(0);
    }
    expect(delivered).toEqual(["tile0", "tile1", "tile2"]);
    // One request per completed tile, not one per poll: the 15 polls collapsed
    // into 3 requests plus the one still in flight.
    expect(calls.length).toBeLessThanOrEqual(4);
  });

  it("reports abort errors raised by run() without calling onError", async () => {
    const { calls, run } = deferredRuns();
    const errors: unknown[] = [];
    const s = new TileScheduler<Key, string>({
      plan: () => ({ t0: 0, t1: 10 }),
      sameKey,
      run,
      deliver: () => {},
      onError: (e) => errors.push(e),
      debounceMs: 100,
    });
    s.request("user");
    await vi.advanceTimersByTimeAsync(100);
    calls[0].fail(abortError());
    await vi.advanceTimersByTimeAsync(0);
    expect(errors).toEqual([]);
  });
});
