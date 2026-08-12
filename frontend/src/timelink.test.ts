import { describe, it, expect, vi } from "vitest";
import { createTimeLink, type TimeWindow } from "./timelink";

describe("createTimeLink", () => {
  it("publish from A delivers to B and not to A (via source identity)", () => {
    const link = createTimeLink();
    const a = { id: "a" };
    const aFn = vi.fn();
    const bFn = vi.fn();
    link.subscribe(aFn);
    link.subscribe(bFn);
    const w: TimeWindow = { tMin: 1, tMax: 5 };
    link.publish(a, w, false);

    expect(aFn).toHaveBeenCalledWith(w, false, a);
    expect(bFn).toHaveBeenCalledWith(w, false, a);
    // A can filter its own publish by comparing source identity.
    expect(aFn.mock.calls[0][2]).toBe(a);
    expect(bFn.mock.calls[0][2]).toBe(a);
  });

  it("subscriber can ignore its own publish via source identity", () => {
    const link = createTimeLink();
    const a = { id: "a" };
    const b = { id: "b" };
    const aReceived = vi.fn();
    const bReceived = vi.fn();
    link.subscribe((w, fl, source) => {
      if (source === a) return;
      aReceived(w, fl, source);
    });
    link.subscribe((w, fl, source) => {
      if (source === b) return;
      bReceived(w, fl, source);
    });

    link.publish(a, { tMin: 0, tMax: 10 }, true);
    expect(aReceived).not.toHaveBeenCalled();
    expect(bReceived).toHaveBeenCalledWith({ tMin: 0, tMax: 10 }, true, a);

    vi.clearAllMocks();

    link.publish(b, { tMin: 2, tMax: 8 }, false);
    expect(aReceived).toHaveBeenCalledWith({ tMin: 2, tMax: 8 }, false, b);
    expect(bReceived).not.toHaveBeenCalled();
  });

  it("unsubscribe stops delivery", () => {
    const link = createTimeLink();
    const a = { id: "a" };
    const fn = vi.fn();
    const unsub = link.subscribe(fn);
    unsub();
    link.publish(a, { tMin: 0, tMax: 1 }, false);
    expect(fn).not.toHaveBeenCalled();
  });

  it("multiple subscribers all receive", () => {
    const link = createTimeLink();
    const a = { id: "a" };
    const fns = [vi.fn(), vi.fn(), vi.fn()];
    fns.forEach((f) => link.subscribe(f));
    const w: TimeWindow = { tMin: 2, tMax: 8 };
    link.publish(a, w, true);
    fns.forEach((f) => expect(f).toHaveBeenCalledWith(w, true, a));
  });

  it("unsubscribe does not affect other subscribers", () => {
    const link = createTimeLink();
    const a = { id: "a" };
    const fn1 = vi.fn();
    const fn2 = vi.fn();
    link.subscribe(fn1);
    const unsub2 = link.subscribe(fn2);
    unsub2();
    link.publish(a, { tMin: 0, tMax: 1 }, false);
    expect(fn1).toHaveBeenCalledOnce();
    expect(fn2).not.toHaveBeenCalled();
  });
});
