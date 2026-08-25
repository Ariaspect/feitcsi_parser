/** Reconciling filter selections against whatever capture just loaded.
 *
 * Two things have to hold at once. A selection that names something this file
 * does not contain would filter every frame away and draw an empty plot with
 * nothing on screen to explain it. And a default is only a default: once the
 * user has chosen for themselves, reapplying it on the next capture is the app
 * arguing with them.
 */

/** Preferred MIMO geometry when a capture offers it.
 *
 * Every capture in this repo carries 2x1; the MTK files pair it with 1x1 and
 * the FeitCSI ones with 2x2. Landing on `all` mixes geometries into a single
 * plot, which is rarely what you want to look at first. */
export const DEFAULT_MIMO = "2x1";

/**
 * Pick the MIMO selection for a newly loaded capture.
 *
 * @param previous  the current selection
 * @param available modes this capture actually contains
 * @param touched   whether the user has changed the dropdown themselves
 */
export function pickMimo(
  previous: string,
  available: string[],
  touched: boolean,
): string {
  if (touched) {
    // Honour the user's choice, but not past a capture that lacks it.
    return previous === "all" || available.includes(previous) ? previous : "all";
  }
  return available.includes(DEFAULT_MIMO) ? DEFAULT_MIMO : "all";
}
