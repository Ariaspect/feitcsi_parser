// Pure module for linking the time axes of stacked heatmaps. Kept out of the
// component so it is unit-testable without a DOM.
//
// The two heatmaps (amplitude + phase) are stacked to be read against each
// other at the same instant. Without linking, zooming one leaves the other at
// full extent and the two can no longer be compared. This module broadcasts
// time-window changes from one plot to all the others, while keeping
// subcarrier zoom per-plot (genuinely independent).
//
// A subscriber receives the publisher's identity and is expected to ignore
// its own publishes — otherwise a link-driven d3 resync would be mistaken for
// user interaction and re-broadcast, creating an infinite loop.

export interface TimeWindow {
  tMin: number;
  tMax: number;
}

export interface TimeLink {
  subscribe(fn: (w: TimeWindow, followLive: boolean, source: object) => void): () => void;
  publish(source: object, w: TimeWindow, followLive: boolean): void;
}

export function createTimeLink(): TimeLink {
  const subscribers = new Set<(w: TimeWindow, followLive: boolean, source: object) => void>();

  return {
    subscribe(fn) {
      subscribers.add(fn);
      return () => {
        subscribers.delete(fn);
      };
    },
    publish(source, w, followLive) {
      for (const fn of subscribers) {
        fn(w, followLive, source);
      }
    },
  };
}
