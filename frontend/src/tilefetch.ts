/**
 * Tile request scheduling: debounce, dedup, abort policy and sequencing.
 *
 * A live capture re-renders on every /api/meta poll (default 300ms), and every
 * one of those renders slides the follow-live window forward by the new data,
 * so every poll wants a slightly different tile. The naive policy -- abort the
 * in-flight request, issue the new one -- gives each request only one poll
 * period to complete. A full-extent tile takes longer than that (measured
 * 0.36-0.62s alone, 0.9-1.8s with eight panels competing), so under follow-live
 * every request was aborted before it landed and the panels never drew:
 * NS_BINDING_ABORTED on repeat, forever.
 *
 * The fix is to separate the two reasons a refetch is wanted:
 *
 * - "user": a gesture (zoom, pan, dblclick, resize, a linked plot's window).
 *   The in-flight result is worthless the moment the user moves, and gestures
 *   arrive at human speed, so aborting is both correct and safe.
 * - "poll": the follow-live window slid because new data arrived. The in-flight
 *   result is still the same picture one poll older, and polls arrive faster
 *   than tiles come back. Deferring rather than aborting keeps the panel
 *   refreshing at whatever pace the backend can sustain.
 *
 * A deferred poll is not queued: when the in-flight request settles the
 * scheduler re-plans from scratch, so any number of polls that arrived while
 * waiting collapse into one request for the newest window.
 */

/** Why a refetch is wanted. Decides whether an in-flight request is aborted. */
export type RequestReason = "poll" | "user";

export interface TileSchedulerOptions<K, T> {
  /**
   * Snapshot what should be fetched right now, or null if nothing can be
   * (no geometry, empty capture, folded panel). Called at request time, not at
   * schedule time, so a deferred request always uses the newest window.
   */
  plan: () => K | null;
  /** Whether a planned request is the one already issued, and can be skipped. */
  sameKey: (a: K, b: K) => boolean;
  /** Issue the request. Rejecting with an AbortError is normal control flow. */
  run: (key: K, signal: AbortSignal) => Promise<T>;
  /** Hand over a result that is neither aborted nor superseded. */
  deliver: (key: K, value: T) => void;
  /** Report a genuine failure. Aborts never reach here. */
  onError?: (error: unknown) => void;
  /** Trailing debounce, in ms. */
  debounceMs?: number;
}

function isAbortError(e: unknown): boolean {
  return e instanceof Error && e.name === "AbortError";
}

export class TileScheduler<K, T> {
  private readonly opts: TileSchedulerOptions<K, T>;
  private readonly debounceMs: number;

  private timer: ReturnType<typeof setTimeout> | null = null;
  /** Strongest reason seen during the current debounce window; user wins. */
  private pendingReason: RequestReason | null = null;

  private controller: AbortController | null = null;
  private inFlight = false;
  /** A poll arrived while a request was in flight; re-plan once it settles. */
  private deferred = false;
  /**
   * The key of the request last issued, for dedup. Cleared whenever a request
   * ends without drawing (abort, failure) so that window stays refetchable --
   * otherwise a capture that stops growing would leave the panel on stale
   * pixels with nothing in flight and every identical retry deduped away.
   */
  private lastKey: K | null = null;
  /** Monotonic request counter, so a superseded response can be dropped. */
  private seq = 0;
  private disposed = false;

  constructor(opts: TileSchedulerOptions<K, T>) {
    this.opts = opts;
    this.debounceMs = opts.debounceMs ?? 100;
  }

  /** Ask for a refetch, trailing-debounced. */
  request(reason: RequestReason): void {
    if (this.disposed) return;
    // A user gesture anywhere in the debounce window makes the whole window
    // user-driven: the gesture must not be demoted by the polls around it.
    this.pendingReason = this.pendingReason === "user" ? "user" : reason;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      const reasonNow = this.pendingReason ?? "poll";
      this.pendingReason = null;
      void this.fire(reasonNow);
    }, this.debounceMs);
  }

  /**
   * Drop the dedup key and abandon in-flight work. For a change of capture or
   * metric, where the pending result belongs to something no longer displayed
   * and an identical-looking window must be fetched again.
   */
  reset(): void {
    this.lastKey = null;
    this.abortInFlight();
  }

  /** Abandon the in-flight request, leaving its window refetchable. */
  abortInFlight(): void {
    this.deferred = false;
    if (!this.controller) return;
    this.controller.abort();
    this.controller = null;
    this.inFlight = false;
    this.lastKey = null;
  }

  /** Tear down on unmount: no further requests, nothing left in flight. */
  dispose(): void {
    this.disposed = true;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.pendingReason = null;
    this.abortInFlight();
  }

  private async fire(reason: RequestReason): Promise<void> {
    if (this.disposed) return;

    const key = this.opts.plan();
    if (key === null) return;
    if (this.lastKey !== null && this.opts.sameKey(this.lastKey, key)) return;

    if (this.inFlight) {
      if (reason === "poll") {
        this.deferred = true;
        return;
      }
      this.abortInFlight();
    }

    const controller = new AbortController();
    this.controller = controller;
    this.inFlight = true;
    this.lastKey = key;
    const seq = ++this.seq;
    // True only while this request is the newest one; a later request makes
    // this one's completion, failure and bookkeeping all irrelevant.
    const current = () => seq === this.seq;

    try {
      const value = await this.opts.run(key, controller.signal);
      // Aborting is not instantaneous: a request can still resolve between
      // abort() and the abort taking effect, so check both.
      if (!current() || controller.signal.aborted) return;
      this.opts.deliver(key, value);
    } catch (e) {
      if (current()) this.lastKey = null;
      if (isAbortError(e)) return;
      this.opts.onError?.(e);
    } finally {
      if (current()) {
        this.inFlight = false;
        this.controller = null;
        if (this.deferred && !this.disposed) {
          this.deferred = false;
          void this.fire("poll");
        }
      }
    }
  }
}
