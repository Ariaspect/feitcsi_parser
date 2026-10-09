import type { ReactNode } from "react";

/** Wraps a chart so a reader can always tell measurement from staleness.
 *
 * A panel that silently keeps drawing the previous range while a new one
 * decodes is the same failure the presence strip is built to avoid: it shows a
 * confident trace for a window nobody has measured yet. A decode of ten
 * minutes at full rate takes seconds, so the gap is long enough to be believed.
 *
 * Three states, and they are different claims:
 *
 *   ready     the trace belongs to the range on the axis
 *   busy      it belongs to the PREVIOUS range; a new one is on its way
 *   empty     nothing has been drawn yet
 *
 * `busy` dims rather than blanks, because a stale trace is still the best
 * guess available and hiding it would make every parameter nudge flash the
 * panel empty. `empty` draws a skeleton at the chart's real height so the page
 * does not jump when the data lands.
 */
export function ChartBusy({
  busy,
  empty,
  height,
  label = "updating",
  children,
}: {
  busy: boolean;
  /** No data at all yet — draw the skeleton instead of the children. */
  empty?: boolean;
  /** The chart's height, so the skeleton reserves exactly its space. */
  height: number;
  label?: string;
  children?: ReactNode;
}) {
  if (empty) {
    return (
      <div
        className="relative w-full animate-pulse rounded-md bg-muted/60"
        style={{ height }}
        role="status"
        aria-label={`${label}…`}
      >
        <Badge label={label} />
      </div>
    );
  }
  return (
    <div className="relative" aria-busy={busy}>
      <div
        // Dimmed enough to read as stale, not so far that a drag -- which
        // refetches on every pointer move -- spends its whole time greyed out.
        // The trace under it is still the best guess for the window being
        // dragged to, so it stays legible.
        className={
          busy ? "opacity-60 transition-opacity duration-150"
               : "opacity-100 transition-opacity duration-150"
        }
      >
        {children}
      </div>
      {busy && <Badge label={label} />}
    </div>
  );
}

/** Centred spinner with a word, so the state is readable and not just felt. */
function Badge({ label }: { label: string }) {
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      <span className="flex items-center gap-2 rounded-full bg-background/85 px-2.5 py-1 text-[11px] text-muted-foreground shadow-sm">
        <Spinner />
        {label}…
      </span>
    </div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg
      className={`h-3 w-3 animate-spin ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z"
      />
    </svg>
  );
}
