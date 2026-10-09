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

### Held-out, 2026-09-30

The 98.3 % above is in-sample: the thresholds were chosen on the captures
they are scored on. Two held-out views.

**Within the session** (the same 235 one-minute captures, thresholds re-chosen
on the training part of each split by balanced accuracy):

| split | held-out accuracy | balanced | fp | fn | thresholds chosen |
|---|---|---|---|---|---|
| 5-fold, blocked in time | 95.3 % | 94.9 % | 9 | 2 | (0.035, 0.25, 5) in 3 folds; (0.06, 0.25, 5); (0.025, 0.2, 6) |
| 3-fold, blocked | 95.7 % | 95.1 % | 8 | 2 | |
| 2-fold, blocked | 95.3 % | 97.2 % | 11 | 0 | |
| chronological, train early / test late | 90.7 % | 94.5 % | 11 | 0 | (0.02, 0.2, 6) |
| chronological, train late / test early | 100 % | 100 % | 0 | 0 | (0.035, 0.25, 5) |
| leave-one-out | 97.4 % | 98.5 % | 6 | 0 | |
| in-sample | 98.3 % | 99.0 % | 4 | 0 | (0.035, 0.25, 5) |

**Across sessions** (the fixed rule applied unchanged to 60 s windows of the
113 labelled five-minute captures from 09-03 … 09-29, windows more than 10 %
mixed skipped; 448 windows, 166 occupied, 282 empty):

| day | windows | accuracy | recall | specificity | empty-window P90 of the lag-2 step, median |
|---|---|---|---|---|---|
| 09-21 (lab_a, 42 Hz) | 142 | 93.0 % | 100 % | 89.9 % | 0.0108 |
| 09-29 | 34 | 82.4 % | 100 % | 72.7 % | 0.0131 |
| 09-11 | 53 | 67.9 % | 100 % | 51.4 % | 0.0255 |
| 09-14 | 32 | 65.6 % | 100 % | 56.0 % | 0.0275 |
| 09-22 | 30 | 63.3 % | 100 % | 54.2 % | 0.0289 |
| 09-15 | 55 | 47.3 % | 100 % | 23.7 % | 0.0715 |
| 09-16 | 46 | 67.4 % | 100 % | 0 % | 0.1061 |
| 09-17 | 27 | 63.0 % | 100 % | 0 % | 0.1183 |
| all | 448 | 73.4 % | 100 % | 57.8 % | |

Recall is 100 % on every day; every error is an empty window on a link whose
idle step floor sits above 0.035 (the one-minute set's empty median is
0.0108, the same as 09-21's). Re-fitting the motion threshold on the
cross-session set gives 0.09 for 85.5 % there, which would then sit above the
still sitters of 09-11 (occupied P10 0.029). So: the breathing half
transfers; the motion threshold is a property of the link state and holds on
the current one (09-21 onward, this AP placement), not on September's.

## Held out on the days after the fit, 2026-10-02

The rule above, unchanged (lag 2 s, `P90(step) > 0.035` OR `peak ≥ 0.25 for
≥ 5 windows`, one call per capture through `compute_hybrid2`'s
`range_verdict`), on every one-minute capture with a camera sidecar recorded
*after* the fitted set ends at 09-30 15:40: 173 captures, 94 camera-empty, 79
occupied (> 50 % of frames; none in between). The thresholds never saw any of
them.

| block | n | occupied | empty | accuracy | recall | specificity | fp | fn | empty-capture P90 median / max |
|---|---|---|---|---|---|---|---|---|---|
| 09-30 15:45–20:25 | 56 | 24 | 32 | 85.7 % | 100 % | 75.0 % | 8 | 0 | 0.0227 / 0.0814 |
| 10-01 15:15–16:06 | 18 | 9 | 9 | 94.4 % | 100 % | 88.9 % | 1 | 0 | 0.0193 / 0.0447 |
| 10-01 18:42–20:12 | 30 | 27 | 3 | 100 % | 100 % | 100 % | 0 | 0 | 0.0146 / 0.0146 |
| 10-02 02:00–04:27 | 50 | 0 | 50 | 100 % | — | 100 % | 0 | 0 | 0.0150 / 0.0237 |
| 10-02 17:18–18:27 | 19 | 19 | 0 | 100 % | 100 % | — | 0 | 0 | — |
| all | 173 | 79 | 94 | 94.8 % | 100 % | 90.4 % | 9 | 0 | 0.0162 / 0.0814 |

Balanced 95.2 %, against 99.0 % in-sample and 94.9–95.1 % under the blocked
folds — the folds predicted it. Of the 79 occupied captures 73 are caught by
motion (17 of those also carry a 5-window breathing run) and 6 by breathing
alone. The gap the threshold sits in is still there: occupied P90 min 0.0225,
5th percentile 0.0316; empty P90 median 0.0159 and 95th percentile 0.0263
once the block below is set aside. The night of 10-02 reads 0.0150 median,
a little above the fitted set's 0.0108, well under the threshold.

**The nine false positives, all by motion.** Six are consecutive:
09-30 17:55, 18:05, 18:10, 18:15, 18:20, 18:25, P90 0.073–0.081 with 58–59
of 60 seconds above the threshold — a steady level for half an hour, not a
burst. The camera has no box in any of their frames. But the 18:00 capture
between them is camera-occupied (P90 0.14, breathing run 23), the person
sits on camera from 18:30 to 20:20, the near chair has moved between the
17:55 and 18:10 stills, a laptop is on the desk at 18:30, and the 20:25
capture after they leave is back at 0.018. The other three are short:
09-30 16:30 (P90 0.071, 9 s above), 17:45 (0.040, 9 s), 10-01 15:33 (0.045,
13 s) — the last lies between camera-occupied 15:27 and 15:36.

Whether the half-hour block is a person out of the camera's field or a link
episode, the per-frame step cannot say. Spectrum of the one-frame-lag
ratio-complex step over each capture's first minute (fraction of variance
above 5 Hz, lag-1 autocorrelation):

| captures | median step | > 5 Hz | ρ₁ |
|---|---|---|---|
| the five block captures 17:55–18:25 | 0.069–0.075 | 0.71–0.74 | 0.14–0.15 |
| camera-occupied, sitting: 18:00, 20:10 | 0.081, 0.075 | 0.71, 0.71 | 0.15, 0.09 |
| clean empties: 17:30, 10-02 03:00, 04:00 | 0.013–0.014 | 0.57–0.61 | 0.31–0.45 |
| noisy-link empties 09-16 ×2, 09-17 | 0.083–0.101 | 0.48–0.65 | 0.05–0.16 |
| the three short bursts | 0.014–0.021 | 0.04, 0.37, 0.44 | 0.94, 0.85, 0.63 |

A sitting person and the noisy link share one signature — broadband, ρ₁
near 0.1 — so the block matches the sitter at 18:00 exactly and the 09-17
empty room nearly as well; the same thing `docs/frame_step.md` found on
09-16, from the other side. The three short bursts are low-frequency
(ρ₁ 0.6–0.9): something moved. None of the nine is adjudicated; the camera
labels stand until the operator says where they were.

So on the current link the rule holds out at 95 % balanced with every miss on
the camera-empty side, and the one sustained miss is a half hour whose truth
the camera cannot vouch for.
