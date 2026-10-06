# The one-minute capture classifier: the test programme

Decision of 2026-10-06: the application verdict is **one call per one-minute
capture**, not a per-second state. A one-minute capture is almost always a
single state (0 mixed captures in the labelled set), which rules out every
within-capture self-calibration — the own-floor P20, VE-Otsu, FRID's
adjacent-window ratio — and leaves three routes to the link-state problem:
a noise-invariant feature, an explicit link-state normaliser that does not
respond to occupancy, or a per-link-state refit from camera-verified
captures. Seven tests were agreed (`docs/literature.md` for the sources),
run in order, each stopped and evaluated before the next:

1. WiDetect's lag-1 autocorrelation on the ratio level — does its empty
   floor hold across link states, and do still sitters sit above it?
2. A link-state indicator that ignores occupancy (across-subcarrier
   roughness of the ratio within a frame).
3. The breathing channel at 60 s: 30 s windows, cross-subcarrier agreement,
   an 11 rpm floor.
4. A capture-level feature set with a small classifier, leave-one-day-out
   and leave-one-link-state-out.
5. Seated-still against noisy-empty on cross-subcarrier coherence.
6. Non-human movers (`docs/plans/nonhuman_motion_plan.pdf`).
7. The operational fallback: per-link-state thresholds refreshed from
   camera-confirmed-empty captures.

**The tab.** `backend/classifier.py`, `/api/classifier`, the **Classifier**
tab: the feature bank. A range is cut into units (one-minute captures are
one unit; a longer range into equal units nearest the requested length) and
every feature the programme has produced is a scalar per unit, listed in
`classifier.FEATURES` with its origin test, a status (*in rule* /
*candidate* / *context*) and, where one has been measured, a reference
operating point drawn as a guide. The tab renders that list — a strip of
unit cells per referenced feature against the camera, the per-window and
per-second series behind the scalars, and a table of units × features — so
a feature a later test adds appears without new column code. The references
are measured operating points, not the verdict; the verdict rule is what
the remaining tests decide.

Units throughout: a one-minute capture is one unit; a longer capture is cut
into 60 s windows. Truth per unit from the camera sidecar: occupied when
more than half the frames show a person, empty when none does, mixed
otherwise. Captures tagged `robot_vacuum_no_person` (10-05 evening, all of
10-06) are non-human movers, kept out of the empty class on the user's
instruction and held for test 6.

## Test 1 — the lag-1 autocorrelation of the ratio level (2026-10-06)

**What was computed.** For every labelled capture on 14 days (640
captures, 1,043 units, 09-04 → 10-06): the complex CSI ratio of the two RX
chains on the uniform frame set, 244 live subcarriers. Per T-frame window
(T = 84 ≈ 2 s at 42 Hz; WiDetect's T = 60 at 30 Hz is also 2 s), per
subcarrier, a linear detrend and the lag-1 sample autocorrelation; averaged
over subcarriers → ψ̂ per window; per unit the median and the 90th
percentile over its ~30 windows. Three series: ratio amplitude |r|,
unwrapped ratio phase, and the raw amplitude plane (power) for reference.
Beside it the lag-2 s frame step's per-second P90, the range rule's input.
Under WiDetect's model the no-motion ψ̂ is N(−1/T, 1/(F·T)) = N(−0.012,
0.007²) for F = 244, and a threshold η has false-alarm rate
Q(√(F·T)·(η + 1/T)). Scripts: `lg:/tmp/test1_acf.py`,
`/tmp/test1_spectrum.py`; units in `/tmp/test1_acf.json`.

### The empty-room floor by day

Camera-empty units, median over units [IQR]:

| day | fs | n | step P90 | ψ̂ amplitude | ψ̂ phase | ψ̂ raw power | gain crossings / min |
|---|---|---|---|---|---|---|---|
| 09-15 | 19.6 | 29 | 0.068 [0.014, 0.078] | 0.038 [0.021, 0.062] | 0.026 | 0.37 | 289 |
| 09-16 | 19.6 | 14 | **0.105** [0.101, 0.107] | **0.022** [0.004, 0.057] | 0.049 | 0.11 | 470 |
| 09-17 | 41.7 | 12 | **0.123** [0.111, 0.127] | **0.008** [−0.004, 0.052] | 0.008 | 0.04 | 360 |
| 09-21 | 41.7 | 95 | 0.011 [0.011, 0.016] | −0.037 [−0.043, 0.002] | −0.029 | 0.02 | 1271 |
| 09-29 | 43.5 | 59 | 0.011 [0.011, 0.012] | −0.011 [−0.016, −0.003] | −0.004 | −0.01 | 212 |
| 09-30 | 43.5 | 186 | **0.011** [0.011, 0.014] | **−0.009** [−0.017, 0.004] | −0.001 | −0.01 | 170 |
| 10-01 | 43.5 | 12 | 0.019 [0.016, 0.022] | −0.003 [−0.007, 0.003] | 0.000 | 0.52 | 110 |
| 10-02 | 41.7 | 50 | 0.015 [0.015, 0.016] | 0.003 [−0.002, 0.013] | 0.014 | 0.12 | 49 |

The step's floor moves 10× between 09-30 and 09-16/17; ψ̂'s does not:
−0.009 against +0.022 / +0.008, inside one null width (see below). That is
the property the paper promised, and it holds on our link. The raw-power
column is 0.02–0.5 on empty rooms — the AGC, as expected; the ratio is
required. (09-04 / 09-11 / 09-14 / 09-22 "empties" are the camera-empty
windows of occupied captures and read 0.07–0.19 with huge spread — the
person is near, not gone; they are not clean empties and are left out of
the floor comparison.)

### The null is not WiDetect's null

Window-level ψ̂ on camera-empty units, pooled per day; F_eff = 1/(T·var):

| day | windows | mean | std | F_eff |
|---|---|---|---|---|
| theory, F = 244 | | −0.012 | 0.007 | 244 |
| 09-29 | 1,116 | −0.005 | 0.046 | 5.7 |
| 09-30 | 4,062 | 0.018 | 0.099 | 1.2 |
| 10-01 | 220 | 0.006 | 0.034 | 10.4 |
| 10-02 | 1,165 | 0.015 | 0.040 | 7.3 |
| 09-16 | 144 | 0.034 | 0.072 | 2.3 |
| 09-17 | 213 | 0.007 | 0.093 | 1.4 |
| 09-21 | 2,212 | 0.011 | 0.129 | 0.7 |
| current link pooled | 6,453 | 0.014 | 0.082 | 1.8 |

Two departures. **F_eff ≈ 2–10, not 244**: the ratio's subcarriers move
together, so averaging over them buys almost nothing and the null is
0.04–0.08 wide instead of 0.007. The mean is right (−0.005 to +0.018
against −0.012), the width is not. And **the tails are heavy**: at the
Q-function η for 1 % false alarm (0.18 with F_eff = 1.8) the window-level
rate on current-link empties is 4.4 %, on 09-21 9.1 %, on 09-15 6 %, on
09-16/17 2.8 %.

### Separation, same link and across links

AUC, occupied against empty, unit level (1 = perfect, 0.5 = chance):

| statistic | still sitters vs current-link empties | still sitters vs 09-16/17 empties | seated movement vs 09-16/17 empties | quiet 1-min occupied (step < 0.06) vs current empties | all 1-min occupied vs current empties |
|---|---|---|---|---|---|
| step P90 | **0.996** | **0.459** | 0.771 | **0.941** | **0.974** |
| ψ̂ amp, median of windows | 0.908 | 0.810 | 0.627 | 0.878 | 0.923 |
| ψ̂ amp, P90 of windows | 0.909 | **0.824** | 0.852 | 0.853 | 0.923 |
| ψ̂ phase, P90 of windows | 0.899 | 0.765 | **0.901** | 0.853 | 0.924 |

A threshold set on the current link's clean empties (301 units, the 09-30
18:05–18:25 block removed), then applied unchanged elsewhere:

| statistic | spec. on current link | thr | recall still | recall seated | recall 1-min | recall quiet 1-min | spec. 09-16/17 | spec. 09-15 | spec. 09-21 |
|---|---|---|---|---|---|---|---|---|---|
| step P90 | 90 % | 0.021 | 1.00 | 1.00 | 0.90 | 0.77 | **0.00** | 0.31 | 0.82 |
| step P90 | 99 % | 0.070 | 0.91 | 1.00 | 0.54 | 0.00 | **0.00** | 0.52 | 0.96 |
| ψ̂ amp P90w | 90 % | 0.20 | 0.56 | 0.63 | 0.63 | 0.19 | **0.96** | 0.93 | 0.83 |
| ψ̂ amp P90w | 95 % | 0.34 | 0.38 | 0.26 | 0.60 | 0.13 | 1.00 | 0.97 | 0.87 |
| ψ̂ amp P90w | 99 % | 0.54 | 0.18 | 0.03 | 0.52 | 0.07 | 1.00 | 0.97 | 0.96 |
| ψ̂ phase P90w | 90 % | 0.22 | 0.56 | 0.69 | 0.63 | 0.19 | 0.92 | 0.93 | 0.81 |
| ψ̂ phase P90w | 95 % | 0.36 | 0.41 | 0.37 | 0.60 | 0.15 | 1.00 | 0.97 | 0.87 |

The trade is plain. On its own link the step is the sharper statistic
(0.996 against 0.91). Across link states the step collapses to chance and ψ̂
keeps 0.82–0.90; a ψ̂ threshold chosen at home holds 92–100 % specificity on
the 09-16/17 link where the same exercise with the step gives 0 %. The
price is recall: at 90 % home specificity ψ̂ finds 56 % of still sitters and
19 % of the quiet one-minute occupants. The still sitters it misses are the
ones buried in the ratio's own noise — all five 09-17 units (ψ̂ P90w
0.10–0.12 against an empty 0.05 that day), the 09-15 21:31 capture, half of
09-16 — and the whole 10-05 room (below).

### What carries the autocorrelation

Band fractions of the variance of the subcarrier-mean detrended |r| over
one minute, and ψ̂ after a moving-average high-pass at 0.5 and 2 Hz:

| unit | < 0.1 Hz | 0.1–0.5 | 0.5–2 | 2–5 | > 5 Hz | ψ̂ | ψ̂ hp 0.5 | ψ̂ hp 2 |
|---|---|---|---|---|---|---|---|---|
| empty, clean night 10-02 03:00 | 0.06 | 0.10 | 0.16 | 0.19 | 0.48 | 0.013 | 0.016 | −0.024 |
| empty, clean 09-30 17:30 | 0.08 | 0.12 | 0.11 | 0.16 | 0.53 | 0.066 | 0.064 | −0.006 |
| empty, noisy 09-17 20:13 | 0.01 | 0.02 | 0.10 | 0.17 | **0.70** | 0.045 | 0.048 | 0.014 |
| empty, noisy 09-16 14:03 (19.6 Hz) | 0.01 | 0.09 | 0.22 | 0.39 | 0.28 | −0.010 | −0.019 | −0.114 |
| empty by camera, 09-30 06:35 | 0.11 | 0.22 | 0.26 | 0.19 | 0.22 | **0.520** | 0.539 | 0.417 |
| empty by camera, 09-30 08:35 | 0.09 | 0.17 | 0.18 | 0.17 | 0.38 | **0.382** | 0.387 | 0.226 |
| still sitter 09-15 13:38 | 0.14 | 0.23 | 0.28 | 0.17 | 0.17 | 0.189 | 0.115 | −0.060 |
| still sitter 09-17 19:58 | 0.20 | 0.10 | 0.12 | 0.19 | 0.39 | 0.050 | 0.044 | 0.017 |
| still sitter 10-01 15:48 | **0.72** | 0.13 | 0.08 | 0.02 | 0.04 | 0.387 | 0.385 | 0.112 |
| still sitter, 10-05 room | 0.05 | 0.06 | 0.07 | 0.17 | 0.66 | −0.002 | 0.000 | −0.025 |
| robot vacuum 10-06 | 0.08 | 0.32 | 0.36 | 0.18 | 0.05 | 0.430 | 0.440 | 0.345 |

The noisy 09-17 link is white above 5 Hz and ψ̂ reads it as the null — the
statistic does for receiver noise exactly what the paper says. What it
reacts to is slow coherent variation in 0.1–2 Hz: a sitter's micro-motion
and breathing (09-15, 10-01), the robot, and the camera-empty mornings of
09-30. Those mornings — 06:35, 08:25–08:50, 10:00 — are five-minute
episodes with ψ̂ 0.38–0.52 for every window of the minute, the step only
2–3× its floor (0.018–0.028, under the rule), gain crossings normal, and
nothing on camera. The spectrum is not receiver noise; something near the
link moved, out of view. The step nearly ignores it; ψ̂ does not. They set
the current link's 99th percentile (0.54) and are why the ψ̂ operating
point costs so much recall.

### Side findings

- **10-05 is a different room** (whiteboard, shelving; the stills show it;
  the sidecars carry no `room` tag until the evening). 66 camera-occupied
  units, a person seated at a laptop: step P90 0.019–0.037 in the quiet
  half-hours, ψ̂ P90w 0.07–0.15 — **below the range rule (0.035) and below
  any ψ̂ threshold** that keeps the home specificity. A still sitter in that
  geometry reads where an empty room reads in this one. No empty captures
  of that room exist yet; nothing can be said about its floor.
- **Robot vacuum** (10-05 20:55–21:10, 10-06 13:27–14:55, 40 units, no
  person): step P90 0.038–0.045, over the rule; ψ̂ P90w 0.52–0.62, over any
  threshold. Both channels call it motion. Held for test 6.
- 09-30 11:25 and 12:05, the range rule's two daytime false positives, have
  ψ̂ 0.25 / 0.16 with 601 / 250 gain crossings and steps 0.098 / 0.062 —
  real motion by every statistic; the camera did not see it.

### Verdict

- *Is the empty-room noise white on both kinds of day?* No in WiDetect's
  sense — F_eff ≈ 2, the null is ten times wider than the formula gives,
  and the tails are heavy — but **the part that matters holds**: receiver
  noise, clean or 10× noisier, lands ψ̂ at the same place. The 09-16/17
  link-state shift that defeats the step does not move ψ̂.
- *Do still sitters sit above both floors?* About half of them, and not the
  ones whose signal is under the ratio's noise (09-17, the 10-05 room). On
  the home link the step remains the better detector by a wide margin.
- **So ψ̂ is a feature, not a replacement.** Its value is a floor that
  transfers across link states; its cost is sensitivity to slow motion of
  any origin, in or out of the room. It goes into test 4 beside the step.
  The Q-function threshold is not usable as such — F_eff must be measured
  and the tails are not Gaussian — so η is an empirical operating point
  like any other.
- The phase variant is marginally better for seated movement against the
  noisy link (0.90 against 0.85) and otherwise equal; both are kept.

## Test 2 — a link-state indicator that ignores occupancy (2026-10-06)

**The hypothesis.** Motion changes the channel *between* frames; estimation
noise is white *across* subcarriers within one frame, where the channel is
smooth. So the across-subcarrier roughness of a single frame — the second
difference along subcarriers, median over the frame, median over the unit —
should read the receiver's noise and not the room, and could normalise the
step. Computed on the same 1,050 units as test 1 (ratio with nulls left as
NaN; also the fourth difference, the raw plane, and the far-delay fraction of
the subcarrier FFT), with the lag-2 s step beside it. Scripts
`lg:/tmp/test2_rough.py`, `test2_ramp.py`, `test2_delay.py`, `test2_lag1.py`.

### Roughness ignores the room — and the link

Camera-empty units, medians per day:

| day | fs | step P90 | roughness Δ² | Δ⁴ | raw plane | FFT tail | step / Δ² |
|---|---|---|---|---|---|---|---|
| 09-14 | 19.6 | 0.027 | 0.0147 | 0.038 | 0.137 | 0.011 | 1.9 |
| 09-15 | 19.6 | 0.068 | 0.0179 | 0.046 | 0.148 | 0.006 | 3.8 |
| 09-16 | 19.6 | **0.105** | **0.0220** | 0.056 | 0.144 | 0.014 | 4.8 |
| 09-17 | 41.7 | **0.123** | **0.0184** | 0.043 | 0.169 | 0.002 | 6.7 |
| 09-21 | 41.7 | 0.011 | 0.0156 | 0.038 | 0.159 | 0.001 | 0.7 |
| 09-30 | 43.5 | **0.011** | **0.0131** | 0.034 | 0.111 | 0.002 | 0.8 |
| 10-02 | 41.7 | 0.015 | 0.0229 | 0.056 | 0.195 | 0.035 | 0.7 |

Spearman across days between the step floor and any roughness: 0.2–0.5. The
roughness sits at 0.012–0.023 everywhere while the floor spans 10×; 10-02,
the cleanest step floor of October, has the roughest frames. It *is*
occupancy-blind — occupied / empty medians 0.91–1.26×, AUC 0.13–0.75, mostly
near 0.5, on eleven days with both classes — but it does not measure the
thing that moves the floor. Normalising the step by it lifts the still-vs-
noisy-empty AUC from 0.46 only to 0.64.

### So what is the noisy state?

Three mechanism checks on one unit per case:

- **Not a per-frame scalar.** Splitting the lag change *q_t(k) = r_t/r_{t−L}*
  into the component shared by every subcarrier and the remainder: on 09-16/17
  the shared part is 0.041 of a 0.110 floor and the remainder 0.105. On the
  current link 0.003 of 0.011. The fluctuation is frequency-selective.
- **Not a per-frame ramp.** A per-frame fit of offset + slope (and + curvature)
  in log-amplitude and phase across subcarriers, removed before the step,
  changes clean units by nothing and makes noisy ones worse — the fit itself
  is noise there.
- **Partly a transmit-delay switch, on one day.** The ratio is a transmit
  pair (README, *The ratio is a transmit pair*), so a change in the AP's
  per-chain delay is a change in the ratio's phase ramp. Per frame, the ramp's
  delay from the FFT peak: clean days constant to the 0.8 ns resolution, no
  frame jumping; **09-17: 44 % of consecutive frames jump by > 3 ns, four
  states, IQR 6.5 ns** — but the step restricted to pairs in the same state is
  still 0.090 against 0.110, and 09-16 (floor 0.103) never jumps at all.
  Still sitter 09-17 and seated movement 09-21 also show jumps (29 %, 3 %).

RSSI does not separate the states (09-16/17 −42 dBm, 09-21 −41, 09-30 −43,
10-02 −47) and neither do gain crossings (09-16/17 360–470 per minute, 09-21
1,271, 10-02 49). The noisy state is a frame-to-frame change of the
transmit-pair ratio that is smooth and selective in frequency, which is what
a change of the *channel* looks like; its cause is not identified here.

### What does read the link: the step at one frame of lag

At one frame apart a still or slow occupant adds almost nothing to the step
(`docs/hybrid2.md`: 0.94× the empty level) while the link's jitter is all of
it. Its 20th percentile over the unit's seconds (`lag1_p20`):

| day | fs | n | lag-1 P10 | lag-1 P20 | lag-1 P50 | lag-2 P90 (floor) | floor / lag-1 P20 |
|---|---|---|---|---|---|---|---|
| 09-11 | 19.6 | 22 | 0.0189 | 0.0194 | 0.0206 | 0.0262 | 1.35 |
| 09-15 | 19.6 | 29 | 0.0412 | 0.0430 | 0.0464 | 0.0676 | 1.57 |
| 09-16 | 19.6 | 14 | 0.0728 | 0.0768 | 0.0851 | 0.1050 | 1.37 |
| 09-17 | 41.7 | 12 | 0.0910 | 0.0955 | 0.1019 | 0.1230 | 1.29 |
| 09-21 | 41.7 | 95 | 0.0101 | 0.0102 | 0.0105 | 0.0108 | 1.06 |
| 09-29 | 43.5 | 59 | 0.0100 | 0.0101 | 0.0103 | 0.0110 | 1.09 |
| 09-30 | 43.5 | 186 | 0.0099 | 0.0100 | 0.0102 | 0.0108 | 1.08 |
| 10-01 | 43.5 | 12 | 0.0146 | 0.0149 | 0.0154 | 0.0185 | 1.24 |
| 10-02 | 41.7 | 50 | 0.0137 | 0.0138 | 0.0141 | 0.0150 | 1.09 |

It tracks the empty floor one-to-one across a 10× range. Against the same
day's empties, by class (median [p10, p90] of the ratio): **still sitters
0.97× [0.80, 1.61], phone 1.07× [0.63, 1.68], robot 1.02× [0.99, 1.07]** —
blind, as hoped — but **seated movement 7.5× [1.1, 7.9], walking 2.6×, all
1-min occupied 1.38× [1.0, 7.0]**: anyone moving continuously moves
consecutive frames too. So as a per-unit normaliser it is flawed (the
normalised step's AUC: 0.91 same link, 0.76 still vs noisy empty, and seated
movement's denominator is inflated along with its numerator), and the
operational form is a **trailing minimum**: the smallest `lag1_p20` of the
previous ten units, whatever their occupancy, since occupancy can only push
the jitter up. Estimate against the day's true empty floor, median [p10,
p90]: 09-16 0.90 [0.70, 0.97], 09-17 0.92 [0.80, 0.96], 09-22 0.96, 09-29
0.98, 09-30 0.99 [0.98, 1.29], 10-01 0.98, 10-02 0.99 [0.99, 1.23]; 09-21 1.06
[0.98, 7.0] (a person moving through more than ten consecutive units lifts
even the minimum) and 09-11 0.62 (that day's "empty" windows are the
near-presence ones). On the occupied units alone the estimate is the same —
the floor is read through the occupant.

### A threshold relative to that floor

Present when lag-2 P90 > α × trailing floor, no absolute minimum, on every
unit with an estimate (units in time order; 505 empty, 394 occupied):

| rule | spec. 09-16/17 | spec. 09-15 | spec. 09-21 | recall still | recall seated | recall all occupied | spec. all | balanced |
|---|---|---|---|---|---|---|---|---|
| fixed P90 > 0.035 | **0.00** | 0.33 | 0.93 | 1.00 | 1.00 | 0.83 | 0.84 | 0.835 |
| α = 2.0 | **1.00** | 0.85 | 0.83 | 0.14–0.40 | 0.63–0.80 | 0.74 | 0.87 | 0.802 |
| α = 2.5 | 1.00 | 0.85 | 0.89 | 0.14–0.20 | 0.40–0.47 | 0.67 | 0.90 | 0.782 |
| α = 3.0 | 1.00 | 0.89 | 0.92 | 0.14–0.20 | 0.40–0.43 | 0.64 | 0.93 | 0.782 |

The step over the trailing floor, by class: empty 1.14 [1.06, 2.53]; still
sitters 2.99 [1.25, 11.5]; seated movement 2.32 [1.45, 12.5]; phone 8.8;
1-min occupied 7.3 [1.8, 19.9]; env2 occupied 1.72 [1.22, 14.9]; **robot
2.50 [2.22, 2.83]**. The relative rule does exactly what a relative rule
can: it trades the noisy link's false alarms (82 → 68 of 505 empties) for
its misses, and the balanced accuracy does not move, because on 09-16/17 a
still sitter's lag-2 step is 1.25–1.5× a floor that is itself the jitter.
The fixed rule's 100 % recall there was never detection; it was calling
every minute present.

### Verdict

- *Can a link-state indicator be read without knowing the occupancy?*
  **Yes** — not the within-frame roughness that was proposed, but the
  one-frame-lag step's low percentile, taken as a trailing minimum over
  units. It tracks the floor 1:1 across every link state seen and a still
  person does not move it. It belongs in the bank (`lag1_p20`, `step_norm`)
  and is what test 7's refit should key on.
- *Does normalising the step by it solve the still sitter across links?*
  **No.** Specificity transfers; recall does not, because on the noisy link
  the sitter's slow signal is below the link's own jitter. No motion
  statistic will see under that — test 1's ψ̂ reached half of them by
  looking at persistence rather than size, and that is its ceiling. Still-
  sitter recall on a noisy link has to come from breathing (test 3) or from
  a feature that sees structure the jitter lacks (test 5).
- The noisy state's cause is still open. It is a smooth, frequency-selective,
  frame-to-frame change of the transmit-pair ratio, partly (on 09-17) a
  per-chain delay switch; not receiver noise, not AGC, not RSSI.
