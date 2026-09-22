# The hybrid detector: calibration-free presence

`backend/hybrid.py`, `/api/hybrid`, the **Hybrid** tab. Presence from two
kinds of evidence and a hold, with no empty-room reference of any kind.

## Rule

A human in a room cannot be quiet in both senses at once: over any half
minute they either move — irregularly — or sit still, and a still human
breathes. A pushed chair does neither once it is down.

- **Motion burst**: per-second median |Δr|/|r| of the CSI ratio above a
  threshold for ≥ 2 s. Opens presence, refreshes it.
- **Breathing**: the FarSense normalised autocorrelation peak (30 s window,
  0.1 Hz high-pass, *first positive* local maximum) ≥ 0.15 through 15 s of
  consecutive windows, 80 % of which agree on the rate within ±3 rpm of the
  run's median. Opens presence too; keeps it open while the person sits.
- **Hold**: presence persists 20 s after the last evidence, then drops.
- **Leading hold** (`lead_hold`, on): breathing evidence also holds
  presence 20 s *before* it. A gap between two pieces of evidence whose
  holds meet — a burst's trailing hold reaching a breathing run's leading
  hold, or two trailing holds — is present throughout (state `bridged`).

The motion threshold is `max(2 × floor, 0.10)`, where *floor* is the link's
quiet level — the 20th percentile of the per-second motion level over the
captures within ±2 h of this one (see §3 for why not the range's own).

Scored on a 1 s grid against the camera through `backend.truth` (empty
frames within 5 s of a transition not scored).

## 1. Corpus, 75 labelled captures (September 20 Hz + 2026-09-21 42 Hz)

| variant | recall | specificity | balanced |
|---|---|---|---|
| first cut (all-or-nothing breathing run, paper's first peak, own floor) | 39.6 % | 88.9 % | 64.2 % |
| + tolerant run (80 %), first *positive* peak, own floor | 56.3 % | 82.0 % | 69.1 % |
| + floor from the whole day | 75.8 % | 74.9 % | 75.3 % |
| **+ floor from ±2 h (default)** | **75.7 %** | **80.5 %** | **78.1 %** |
| ±2 h floor, hold 30 | 79.1 % | 76.6 % | 77.8 % |
| ±2 h floor, hold 20, + leading hold | 80.6 % | 76.7 % | 78.7 % |

Leading hold on/off, hold 20, by floor and link (recall / specificity /
balanced):

| floor | lead | all 75 | 2026-09-21 (32) | September (43) |
|---|---|---|---|---|
| ±2 h | off | 75.7 / 80.5 / 78.1 | 97.5 / 85.2 / 91.4 | 63.9 / 76.0 / 69.9 |
| ±2 h | on | 80.6 / 76.7 / 78.7 | 99.1 / 82.2 / 90.7 | 70.7 / 71.4 / 71.1 |
| none (fixed 0.10) | off | 95.4 / 59.0 / 77.2 | 97.5 / 85.2 / 91.4 | 94.3 / 34.1 / 64.2 |
| none (fixed 0.10) | on | 96.7 / 56.2 / 76.4 | 99.1 / 82.2 / 90.7 | 95.4 / 31.4 / 63.4 |
| own range | off | 56.3 / 82.0 / 69.1 | 49.8 / 85.2 / 67.5 | 59.8 / 78.9 / 69.4 |
| own range | on | 62.4 / 78.2 / 70.3 | 54.5 / 82.2 / 68.4 | 66.7 / 74.4 / 70.5 |

Fixed thresholds without any floor, lead off (all / 0921 / September,
balanced): 0.05 → 74.5 / 90.7 / 59.0; 0.10 → 77.2 / 91.4 / 64.2; 0.20 →
74.0 / 81.4 / 67.6; 0.30 → 68.6 / 68.2 / 68.4. Breathing only, no motion
channel: 66.8 / 65.4 / 67.2.

By link, ±2 h floor, hold 20:

| link | n | recall | specificity | balanced |
|---|---|---|---|---|
| **2026-09-21, 42 Hz, lab_a** | 32 | 97.5 % | 85.2 % | **91.4 %** |
| September, 20 Hz | 43 | 63.9 % | 76.0 % | 69.9 % |

For comparison, the calibrated amplitude-offset detector with in-capture
calibration and the same 5 s margin scores 84.8 % balanced on its 20-capture
September subset — and needs a labelled empty stretch that a moved chair
invalidates. On the current link the calibration-free detector is above
that with no reference at all.

### KPI #1 set: the 42 September captures the calibrated number used

The 2026-09-18 KPI #1 report scored the amplitude-offset detector with
in-capture calibration on 42 captures: the 53 September (09-09 … 09-16)
captures with any camera occupancy, minus the 11 without a 30 s
camera-empty stretch long enough to calibrate. The hybrid on the same 42,
every second scored (it holds nothing out), ±2 h floor, hold 20, through
the `:8002` endpoint on lg:

| detector | margin | accuracy | recall | specificity | balanced |
|---|---|---|---|---|---|
| calibrated, in-capture, 15 s matched window (report) | 0 | 87.4 % | 82.4 % | 93.6 % | 88.0 % |
| calibrated, in-capture, 1 s tiles (report) | 0 | 86.1 % | 83.5 % | 89.0 % | 86.3 % |
| hybrid, lead hold off | 0 | 71.8 % | 86.3 % | 64.3 % | 75.3 % |
| hybrid, lead hold on | 0 | 70.0 % | 90.6 % | 59.3 % | 75.0 % |
| hybrid, lead hold off | 5 | 73.9 % | 86.3 % | 67.1 % | 76.7 % |
| hybrid, lead hold on | 5 | 72.0 % | 90.6 % | 61.8 % | 76.2 % |

By day, hybrid, lead hold on, margin 5 (balanced; lead off in brackets):
09-09 (1) 93.0 [93.0]; 09-10 (2) 77.1 [80.9]; 09-11 (19) 75.1 [75.9];
09-14 (7) 65.1 [68.7]; 09-15 (12) 81.1 [79.4]; 09-16 (1) 96.2 [96.2].
The ±2 h floor on these captures is 0.019–0.084 (one at 0.136), so on 28
of the 42 the threshold is the 0.10 absolute minimum and the floor plays
no part. The errors are false positives in the empty stretches: 2 980 fp
against 4 830 tn at margin 5, lead on.

## 2. What the corpus taught, in the order it was found

1. **Two neighbouring windows agreeing is no evidence.** At a 1 s hop, 30 s
   windows share 29 s; the first cut's "2 consecutive windows" accepted
   noise in an empty room (specificity 0.60 on a walk-in capture). The run
   has to span half a window.
2. **All-or-nothing runs reject real breathing.** 17 straight windows of a
   real breath (peaks 0.16–0.40, rates 16.7–20.0 rpm) failed on one window
   at 0.142 and a 3.3 rpm max−min spread against a 3 rpm tolerance. Now 80 %
   of a run must qualify, and agreement is to the run's median.
3. **The paper's "first peak" catches wiggles.** On a seated occupant with
   finger movement the first local maximum of the autocorrelation sat on
   the rising slope out of the half-period trough: −0.35 to −0.47 at the
   shortest lag for a minute at a time, while the spectrum held a 15–18 rpm
   line. The first *positive* maximum reads +0.37/+0.44 there; empties stay
   at 0.05–0.10. Recall on that capture 0.63 → 0.96.
4. **A range's own floor fails when the range is occupied throughout.** Its
   20th percentile *is* the occupant. The six 0921 evening captures sat at
   0.15–0.18 for half an hour against the link's 0.019 when empty — a body
   attenuating the link raises the ratio's noise nine-fold, which is
   evidence, and only a floor from outside the range can see it (0921:
   67.5 → 91.4 % balanced).
5. **But the floor must be *recent*, not daily.** On 2026-09-15 the link's
   noise was 0.10–0.12 at midday and 0.02 in the evening; a whole-day floor
   sits under the midday empties and the `negative:furniture` captures fire
   throughout (specificity 65 %). A ±2 h window keeps every day at or above
   its own-range score.
6. **The amplitude frame-diff channel stays off.** Offered for the
   shadowing the ratio cannot see, it fires in an empty September room
   (specificity 0.85 → 0.42) and pooled 54.6 % balanced. Raw, not
   AGC-corrected: the correction fires on a person's own variation and adds
   jitter (0.10 → 0.63 dB frame-diff on a still occupant).

7. **A burst inside a breathing window reads as breathing at the band's
   top.** On a synthetic empty room, a 5 s walk makes every 30 s window
   that contains it return the same peak (0.19) at the same rate
   (34.9 rpm): the burst's energy makes the autocorrelation a triangle and
   the first positive local maximum sits at the shortest lag. Fifteen such
   windows agree, so the run rule accepts them and "breathing" is flagged
   from 10 s before the walk to 15 s after it. A 3 s walk does not do this
   (no peak found). `breathing &= ~burst` removes only the burst seconds
   themselves, not the windows that contain them. Not yet measured on the
   corpus; the windows around every walk-in count as breathing evidence
   there, right or wrong.

## 3. What still fails, and why it is the link

On September, `small_gestures`, `seated_low_movement`, `seated_moving`
score 0–20 % recall. A fidgeting person there sits at 0.16–0.18 while the
*empty* room sits at 0.19–0.20 — the empty room is noisier than the person —
so motion cannot separate them and the fidgeting corrupts the breathing
rhythm. The same activities on the 0921 link (empty 0.019–0.027) are caught
at 97 %. Nothing in the rule fixes a link whose idle noise exceeds a
person's movement; that is a radio and geometry question.

Irregular breathing — a chest moving with no stable period (20260916_202702,
130–200 s, rates bouncing 11–22 rpm) — is the one occupant state no rule
here sees. Every period-free quantity tried (in-band energy fraction,
band-to-noise energy ratio, BNR, arc excursion) reads higher in today's
*empty* room than in a September *occupied* one, because a quieter link
makes its residual look tonal. The normalised autocorrelation peak is the
one number that is ~0 in every empty room on every link. The lever left is
the hold length, which buys recall at the empty room's expense (table §1).

## 4. Not yet recorded

A fan, a curtain in an AC draught, a robot vacuum, a pet: the rule assumes
a periodic non-human mover is rare and a constant-level one distinguishable
by its lack of bursts. Neither has been tested for want of a capture.
