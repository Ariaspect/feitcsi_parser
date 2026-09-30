# Hybrid 2: the complex frame step for motion, FarSense for breath

`backend/hybrid2.py`, `/api/hybrid2`, the **Hybrid 2** tab.

Same shape as [hybrid 1](hybrid.md) — motion opens presence, breathing keeps it
open, a hold carries it across the gaps — with the motion channel replaced by
[the ratio-complex frame step](frame_step.md). The breathing half is hybrid 1's,
*called* rather than copied, and a test asserts the two produce bit-identical
`breath_peak` and `breath_rpm`, so a disagreement between the tabs can never be
the breath.

## Why it is not a new detector

The two motion quantities are the same formula. `presence.fractional_motion`
divides by the midpoint magnitude, the frame step divides by the sum:

```
fractional_motion = |r − r′| / [½(|r| + |r′|)]
frame step        = |r − r′| /  (|r| + |r′|)
```

so one is exactly half the other — verified per subcarrier at 0.500000, min and
max, over 14 captures. Every **relative** default therefore carries over
untouched and only the absolute floor halves, 0.10 → 0.05. Arithmetic, not
tuning.

What the substitution buys is the plumbing around the number:

| | hybrid 2 | hybrid 1 |
|---|---|---|
| fold over subcarriers | median | mean |
| frame set | one MAC, one MIMO mode, one bandwidth | whatever the caller filtered to |
| time base | native frame times | uniform grid, interpolated |
| pair spanning a dropout | blanked | interpolated through |
| lag | selectable | one frame |

## KPI evaluation, 2026-09-30

The user's 55-capture labelled set, every second scored against the camera
through `backend.truth`, margin 5, current defaults, pooled per class.
All 55 were found and all carry truth.

| class | n | seconds | hybrid 1 | | | | hybrid 2, lag 0 | | | | **hybrid 2, lag 2 s** | | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | acc | rec | spec | bal | acc | rec | spec | bal | acc | rec | spec | bal |
| phone (폰봄) | 14 | 4 049 | 88.9 | 89.5 | 88.6 | 89.0 | 88.6 | 89.0 | 88.4 | 88.7 | 87.5 | 94.9 | 82.3 | 88.6 |
| static occupancy (정적 재실) | 12 | 3 369 | 86.6 | 90.1 | 80.2 | 85.1 | 85.3 | 89.0 | 78.3 | 83.7 | 85.8 | 92.8 | 72.7 | 82.8 |
| empty (빈 방) | 11 | 3 292 | 98.4 | — | 98.4 | 98.4 | 98.4 | — | 98.4 | 98.4 | 98.4 | — | 98.4 | 98.4 |
| walking (서서 돌아다님) | 11 | 3 170 | 84.6 | 89.1 | 81.6 | 85.3 | 84.4 | 90.6 | 80.1 | 85.3 | 86.7 | 100.0 | 77.5 | 88.7 |
| seated movement (앉아서 움직임) | 7 | 2 068 | 23.2 | 23.2 | — | 23.2 | 21.0 | 21.0 | — | 21.0 | 43.6 | 43.6 | — | 43.6 |
| **POOLED** | **55** | **15 948** | **81.0** | **70.6** | **89.7** | **80.1** | **80.3** | **69.8** | **89.0** | **79.4** | **83.5** | **80.5** | **86.1** | **83.3** |

### What the table says, in order

1. **At lag 0 it ties hybrid 1** — 80.3 against 81.0 pooled balanced 79.4
   against 80.1. That is the correctness check, not a disappointment: the same
   quantity should score the same, and the plumbing differences are worth
   under a point on this set.

2. **The lag is the whole gain.** At 2 s, pooled balanced accuracy goes
   80.1 → **83.3** and recall 70.6 → **80.5**.

3. **Empty rooms are untouched: 98.4 % specificity in all three columns**,
   to the decimal. The recall the lag buys is not paid for with false
   positives on an empty room — which was the risk, and it did not happen.

4. **Seated movement nearly doubles, 23.2 → 43.6 %**, and is still far and
   away the worst class. See below; the fault is not in the metric.

5. **The recall is paid for inside occupied captures.** Specificity on the
   static-occupancy class falls 80.2 → 72.7 and on phone 88.6 → 82.3 — those
   are the camera-empty seconds *within* an occupied capture, so the longer lag
   holds presence further into the stretch before and after the person. Some of
   that is genuinely the person still being near; some is over-trigger. The
   class where it costs nothing is the one where the room is empty for the
   whole capture.

### The seated-movement class

At 43.6 % it is the only class below usable, and the reason is the floor, not
the frame step. The threshold is `max(2 × P20, 0.05)` of the range's **own**
per-second level, and these captures are occupied from start to end, so their
own 20th percentile *is* the occupant: measured at 0.073–0.081 while the same
session's empty level is 0.0123. The threshold lands at 0.15 and the occupant
sits at 0.08, so nothing fires.

With a session floor instead of the range's own, the same six captures recall
100 %. That floor was removed on 2026-09-22 because a pooled floor on a link
whose idle level had risen sets the threshold under an empty room, so it is not
reinstated here — but `motion_floor` accepts one explicitly, and the tab draws
the floor and the threshold on the chart so the failure is visible rather than
inferred.

## What it cannot do, and why it is not fixable here

The motion channel's empty-room floor spans ~10× between link states
(`docs/frame_step.md`), and on the noisiest links an *empty* room reads what a
sitting person reads on a clean one — 0.95–1.13× at every lag from 24 ms to
4 s. The metric is dimensionless but its noise floor is not, so two captures are
comparable only once their link states are known to match. That bounds any
threshold rule built on it, this one included.


## Capture-level rule on the 1-minute captures, 2026-09-30

The 235 one-minute captures recorded every five minutes from 2026-09-29 20:25
to 2026-09-30 15:40 (200 camera-empty, 35 with a person; one of the 35 is
53 % occupied, the rest 98–100 %). Verdict per *capture*, not per second, as
the user defined it: present when the motion is clearly above the empty
level; else present when breathing is seen at all; else empty. Inputs are
this tab's series — the ratio-complex frame step per second and the
FarSense peak per window (positive-only first peak, gate off).

Searched: motion statistic (P90 of the per-second step, or a 2 s burst) at
lag 0 and lag 2 s, thresholds 0.02–0.30; breathing = the longest run of
consecutive windows with peak ≥ P (P 0.15–0.30, with and without ±3 rpm
agreement) ≥ K windows (1–10).

**Chosen:** lag 2 s, `P90(step) > 0.035` OR `peak ≥ 0.25 for ≥ 5 consecutive
windows`.

| rule | accuracy | balanced | fp | fn |
|---|---|---|---|---|
| chosen | 98.3 % | 99.0 % | 4 | 0 |
| motion only, lag 2, P90 > 0.025 | 97.0 % | 97.1 % | 6 | 1 |
| breathing only, peak ≥ 0.2, run ≥ 4 | 91.9 % | 85.8 % | 11 | 8 |
| best at lag 0 (P90 > 0.0175, peak ≥ 0.3 run ≥ 2) | 96.2 % | 93.0 % | 5 | 4 |

The optimum is flat: T 0.030–0.040 and K 5–15 and P 0.25–0.35 all give the
same 4 / 0; T = 0.05 starts losing sitters (2 fn), T = 0.02 adds empties
(9 fp). Empty captures' P90(step, lag 2) sit at 0.0108 median, 0.0134 at
the 90th percentile, 0.0194 at the 95th; occupied ones start at 0.0157
(5th percentile 0.033). Of the 35 occupied captures, 20 are caught by motion
alone, 13 by both, 2 by breathing only (`20260930_133002`, `_143502`).

The four false positives are all motion, none breathing, and all fall on
2026-09-30 between 11:25 and 12:45 (`112502`, `120503`, `121503`, `124503`):
P90 0.06–0.09 with 12–36 s above the threshold each, in captures the camera
called empty. The 12:50 capture is the one the person arrives in.
