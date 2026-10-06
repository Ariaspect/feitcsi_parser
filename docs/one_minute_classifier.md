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

**What each step separates, so far.** One line per completed test; the
sections below carry the numbers.

| step | feature | separates | does not separate |
|---|---|---|---|
| 1 | ψ̂ — lag-1 autocorrelation of the ratio level (persistence of change) | slow coherent motion from receiver noise, with an empty floor that holds across link states (0.011 → 0.11 step floor; ψ̂ −0.009 → +0.02); ~half of still sitters | a still sitter whose signal is under the link's jitter from an empty room; anything that moves slowly near the link (out-of-view people, robot) from a person |
| 2 | one-frame-lag step, P20, trailing minimum (the link's own jitter) | clean link from noisy link, blind to a still person, a phone user or the robot (0.97×, 1.07×, 1.02×); as a ruler, "moves more than the link wobbles" (step/jitter > 2) from empty on every link | a still sitter on a noisy link from empty (1.3–1.5× vs 1.1–2.5×); a robot from a person (2.5×); a very still sitter in the far room (1.7×) |
| 3 | FarSense breathing run (10 s as in the rule; 20 s + 11 rpm floor) | a still person from an empty room on every link, including the noisy one (85 % → 96 % at 20 s) and the far room (75 %), at 0–1 % false alarms on true empties; a person from the robot (robot breathes 5–10 %) | a fidgeting person (23–34 %; motion does that); "motion without breathing" from non-human, since walkers show no peak 55 % of the time |
| 4 | the bank as a whole: hand rule B (step > max(2 × jitter, 0.035) OR 20 s run) and a six-feature logistic P(occupied) | occupied from empty at 0.92 balanced on a link state never seen — the noisy link at 0.85 specificity where the range rule has 0 — with breathing the strongest input and no absolute feature allowed; the learned score also rejects most robot units unseen | more than the rules on the days the rules already work (current link 0.97–0.98 either way); seated fidgeting under a jitter-tied threshold (0.71–0.91); camera-empty windows with a person a metre from the link — label-limited |

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

## Test 3 — the breathing channel at 60 s (2026-10-06)

**What was computed.** On the same units: (A) the range rule's breathing
half as it stands — FarSense 10 s windows, hop 1 s, per-second normalised
autocorrelation peak, longest run of seconds with peak ≥ 0.25, present at
run ≥ 5; (A′) the same with a rate floor at 11 rpm (the empty-room artefact
sits at 7.5–10.9); (B) the same pipeline with 20 s and 30 s windows
(DeMan's finding that 30 s beats 10 s); (C) a DeMan-style estimator built
beside FarSense — per subcarrier, per 30 s window at 50 % overlap, the
periodogram peak in 8–42 rpm on |r| and on unwrapped arg(r), a prominence
gate Q, the window's rate as the median over passing subcarriers, agreement
as the share of all live subcarriers that pass and sit within ±2 rpm of it;
breathing when ≥ 2 of 3 windows agree at ≥ A with rates within 3 rpm and
above 11. Script `lg:/tmp/test3_breath.py`, analysis `test3_analyse.py`.

### Breathing alone

Fraction of units declared breathing — recall for occupied classes, false
alarms for empties:

| rule | still sitters (34) | of which noisy link (26) | seated movement | phone | 1-min quiet (step < 0.06) | env2 sitter (66) | walking | **empty, current link (307)** | empty 09-21 (95) | empty 09-15/16/17 (55) | empty other Sept (69) | **robot (40)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A  10 s, peak ≥ .25, run ≥ 5 (the rule) | **0.85** | **0.85** | 0.23 | 0.68 | 0.74 | 0.76 | 0.45 | **0.00** | 0.00 | 0.09 | 0.06 | 0.05 |
| A′ + rate floor 11 rpm | 0.85 | 0.85 | 0.23 | 0.68 | 0.74 | 0.74 | 0.45 | 0.00 | 0.00 | 0.09 | 0.04 | 0.05 |
| B  20 s + floor | **0.94** | **0.96** | 0.34 | 0.68 | 0.68 | 0.74 | 0.45 | 0.01 | 0.01 | 0.11 | 0.09 | 0.10 |
| B  30 s + floor | 0.91 | 0.92 | 0.29 | 0.64 | 0.65 | 0.68 | 0.45 | 0.01 | 0.02 | 0.05 | 0.04 | 0.07 |
| B  30 s, peak ≥ .30 + floor | 0.76 | 0.73 | 0.14 | 0.64 | 0.56 | 0.65 | 0.32 | 0.01 | 0.00 | 0.02 | 0.01 | 0.03 |
| C  DeMan either, 30 s, Q3, A ≥ .30 | 0.53 | 0.50 | 0.11 | 0.29 | 0.47 | 0.50 | 0.00 | 0.00 | 0.02 | 0.07 | 0.04 | 0.15 |
| C  DeMan either, 30 s, Q4, A ≥ .20 | 0.53 | 0.54 | 0.03 | 0.32 | 0.47 | 0.39 | 0.05 | 0.00 | 0.00 | 0.00 | 0.03 | 0.07 |

Three things. **Breathing reaches the still sitter on the noisy link** —
85 % with the rule as it stands, 96 % at 20 s — where tests 1 and 2 showed
no motion statistic can (test 2: the sitter's step is 1.3–1.5× a floor that
is the link's own jitter). It reaches the far room too: 74–76 % of the
env2 sitters that the 0.035 step missed. And it is **near-silent on true
empties**: 0 of 307 current-link units and 0 of 95 on 09-21 at 10 s, 1 % at
20 s; the 6–13 % on the September "empties" are the camera-empty windows of
occupied captures, where the person is near. **The robot does not
breathe**: 5–10 % against 85–96 % for sitters — the human check test 6
needs, though walking people also lack a peak 55 % of the time, so "motion
without breathing" is not "non-human" by itself.

**DeMan's estimator loses to FarSense's** on this link: 53 % of still
sitters at matched false alarms against 85 %. The per-subcarrier agreement
is dominated by the artefact band — of 411 agreeing empty windows 260 claim
8–11 rpm, and the sitters' agreeing windows put 91 of 335 there too, so the
11 rpm floor that removes the artefact also costs real slow breathers.
FarSense's projection, BNR selection and combined autocorrelation are the
better estimator here; **DeMan's window finding transfers** (20 s: still
sitters 85 → 94 %, noisy-link sitters 85 → 96 %, for 0 → 1 % false alarms on
the current link), his agreement test does not. The 11 rpm floor is free on
the rule (recall unchanged, other-Sept false alarms 6 → 4 %).

### Breathing with the motion channel

The motion rules of tests 1–2 OR a breathing rule, on the 1,006 units with
a trailing floor (robot excluded from the empties):

| rule | still clean | still noisy | seated mov. | phone | 1-min occ. | env2 occ. | empty current | empty 09-21 | **empty noisy** | empty other | robot | spec. | recall | balanced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step > 0.035 (motion only) | 0.88 | 1.00 | 1.00 | 0.88 | 0.85 | 0.37 | 0.05 | 0.07 | 0.81 | 0.38 | 0.92 | 0.838 | 0.832 | 0.835 |
| step > 0.035 OR breath 10 s **(= the range rule)** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.05 | 0.07 | **0.81** | 0.41 | 0.92 | 0.834 | 0.997 | 0.916 |
| step > 2 × trailing floor OR breath 10 s | 1.00 | 1.00 | 0.77 | 1.00 | 1.00 | 1.00 | 0.07 | 0.17 | **0.15** | 0.46 | 0.97 | 0.857 | 0.947 | 0.902 |
| step > max(2 × floor, 0.035) OR breath 10 s | 1.00 | 1.00 | 0.77 | 1.00 | 1.00 | 1.00 | 0.05 | 0.07 | **0.15** | 0.35 | 0.89 | 0.903 | 0.944 | **0.924** |
| step > max(2 × floor, 0.035) OR breath 20 s | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 0.05 | 0.08 | 0.19 | 0.37 | 0.89 | 0.889 | 0.957 | 0.923 |

The range rule's 100 % recall on the noisy link was the motion channel
calling every minute present (specificity 0.19); with the motion threshold
tied to the link floor and breathing carrying the still sitter, the same
units read 0.85 specificity at 100 % still-sitter recall. What the relative
floor costs is seated movement (0.77–0.83: fidgeting raises the one-frame
jitter with the step, test 2) and the clean-link empties when the floor is
allowed under 0.035 (09-21: 0.07 → 0.17, restored by the absolute minimum).
Overall balanced accuracy 0.924 against the range rule's 0.916 — a small
number because most units are on the current link, where the fixed rule
already works; the difference is all on the days it did not.

### Verdict

- *Does the breathing channel reach the quiet occupant the motion channel
  cannot?* **Yes** — on the noisy link (85–96 %), in the far room (74–76 %),
  at 0–1 % false alarms on true empties. This is the still sitter's channel
  and it already works; the range rule's failures on that link were never
  the breathing half.
- *DeMan's improvements:* the **20 s window** transfers (+9 points on still
  sitters, +11 on the noisy link); the **11 rpm floor** is free; the
  **cross-subcarrier agreement** does not beat FarSense's estimator and is
  not adopted. `breath_run` (the rule's) and `breath_run20` (+ floor) are
  in the bank, with the rate.
- *For test 6:* breathing is the human check — robot 5–10 % — but not a
  sufficient one, since walking people show no peak 55 % of the time.

## Test 4 — a capture-level feature set and a small classifier (2026-10-06)

**What was computed.** The per-unit features of tests 1–3 joined on 969
units (929 without the robot: 526 camera-empty, 403 occupied), three feature
sets, two models, two held-out schemes, three fixed rules for comparison.
Feature sets: *transfer* — dimensionless only (step over the trailing floor,
log; median level over the floor; step over own jitter; ψ̂ amp P90 and
median; ψ̂ phase P90; breathing runs at 10 s and 20 s; breathing peaks);
*rule inputs* — log step and the 10 s run; *all* — transfer plus the
absolute step, level, floor, jitter, RSSI and gain crossings. Models:
class-balanced logistic regression on standardised inputs, and a depth-3
gradient-boosted tree. Held out **leave-one-day-out** (13 days) and
**leave-one-link-state-out** (six groups: 09-04/11/14 at 19.6 Hz, 09-15,
09-16/17, 09-21/22, 09-29→10-02, env2). Script `test4_classifier.py`;
table `test4_features.json`.

### Pooled held-out

| rule / model | by day: spec | rec | bal | AUC | by link state: spec | rec | bal | AUC |
|---|---|---|---|---|---|---|---|---|
| range rule (step > .035 OR 10 s run ≥ 5) | 0.816 | 0.998 | 0.907 | — | 0.816 | 0.998 | 0.907 | — |
| hand A: step > max(2 × floor, .035) OR 10 s run | 0.897 | 0.945 | 0.921 | — | 0.897 | 0.945 | 0.921 | — |
| hand B: … OR 20 s run | 0.888 | 0.955 | 0.922 | — | 0.888 | 0.955 | **0.922** | — |
| LR / transfer | 0.930 | 0.921 | 0.925 | 0.975 | 0.924 | 0.906 | 0.915 | 0.970 |
| LR / rule inputs | 0.880 | 0.933 | 0.907 | 0.969 | 0.884 | 0.908 | 0.896 | 0.962 |
| LR / all | 0.924 | 0.948 | 0.936 | 0.982 | 0.922 | 0.928 | **0.925** | 0.978 |
| HGB / transfer | 0.916 | 0.933 | 0.925 | 0.975 | 0.920 | 0.888 | 0.904 | 0.968 |
| HGB / all | 0.928 | 0.958 | **0.943** | 0.981 | 0.895 | 0.921 | 0.908 | 0.968 |

By link state, the harder split, group by group (balanced [spec / rec]):

| held-out link state | nE / nO | range rule | hand B | LR transfer | LR all | HGB all |
|---|---|---|---|---|---|---|
| 09-04/11/14 (19.6 Hz) | 47 / 54 | 0.75 [0.51/0.98] | 0.79 [0.60/0.98] | 0.81 [0.64/0.98] | 0.82 [0.66/0.98] | 0.82 [0.66/0.98] |
| 09-15 | 29 / 22 | 0.66 [0.31/1.00] | 0.91 [0.83/1.00] | **0.94** [0.93/0.95] | 0.88 [0.76/1.00] | 0.86 [0.86/0.86] |
| **09-16/17 (noisy)** | 26 / 54 | **0.50 [0.00/1.00]** | **0.82 [0.85/0.80]** | 0.80 [0.96/0.63] | 0.78 [0.69/0.87] | **0.58 [0.15/1.00]** |
| 09-21/22 | 117 / 58 | 0.94 [0.88/1.00] | 0.88 [0.87/0.90] | 0.88 [0.91/0.84] | 0.94 [0.97/0.91] | 0.94 [0.93/0.95] |
| 09-29 → 10-02 (current) | 307 / 149 | **0.98** [0.95/1.00] | 0.97 [0.95/1.00] | 0.97 [0.97/0.96] | 0.94 [0.98/0.91] | 0.91 [0.98/0.83] |
| env2 (10-05), occupied only | 0 / 66 | rec 1.00 | rec 1.00 | rec 0.98 | rec 0.95 | rec 1.00 |

Per class, held out by day (recall):

| class | n | range rule | hand B | LR transfer | HGB all |
|---|---|---|---|---|---|
| still sitters, clean / noisy link | 8 / 26 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| seated movement | 35 | 1.00 | 0.83 | 0.71 | 0.91 |
| phone, walking | 28, 22 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1-min occupied / of which quiet | 131 / 34 | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.94 | 0.95 / 0.91 |
| env2 occupied | 66 | 1.00 | 1.00 | 0.98 | 1.00 |
| other September occupied | 87 | 0.99 | 0.86 | 0.78 | 0.91 |

Robot units, never trained on, called occupied: range rule 0.90, hand rules
0.88, **LR transfer 0.17, HGB transfer 0.12, HGB all 0.07, LR all 0.85**.

Logistic coefficients on the standardised *transfer* set, fit on all units:
breath peak 10 s **+1.97**, ψ̂ phase P90 **+1.81**, breath run 20 s **+1.63**,
log step over floor +1.08, level over floor +0.92, ψ̂ amp median −0.77,
breath run 10 s +0.57, ψ̂ amp P90 −0.30, breath peak 20 s +0.10, step over
own jitter −0.04.

### Reading it

- **On days the fixed rule already works, nothing beats it; on the days it
  fails, everything beats it.** The range rule is 0.98 on the current link
  and 0.50 on 09-16/17 — specificity zero, every minute called present. The
  hand rule and the learned models are 0.78–0.82 there with specificity
  0.69–0.96, and give up 0.01–0.07 on the current link for it. Pooled, the
  spread between the hand rule and the best learned model is one point.
- **Absolute features do not cross link states** — M-WiFi's lesson, measured
  here: the tree on *all* features is the best model by day (0.943) and
  collapses on the held-out noisy link (specificity 0.15), because it
  learned a step threshold in absolute units. The logistic model survives
  the same features because it weights the dimensionless ones; the
  *transfer* sets lose nothing between the two splits.
- **The learned models earn their point on the September "empties"** — the
  camera-empty windows next to a present person (09-04/11/14: everyone's
  specificity is 0.5–0.7) — and on seated movement they lose it, as every
  rule that ties the threshold to the link jitter does (test 2).
- **Robot:** the dimensionless classifiers reject most robot units without
  having seen one (12–17 % called occupied against the rules' 88–90 %), on
  the strength of no breathing peak and a ψ̂ profile unlike a sitter's. Not
  designed, not yet trusted; test 6 measures it.
- **Breathing is the strongest input** (peak +1.97, 20 s run +1.63), then
  persistence of change (ψ̂ phase +1.81), then the normalised step (+1.08).
  The 20 s run outweighs the 10 s run three to one.

### What goes in the bank

A six-feature **own-floor** logistic regression — step over the unit's own
jitter (log), ψ̂ phase P90, ψ̂ amp median, the 20 s run, the 10 s peak, the
10 s run — so a unit can be scored on its own without a trailing window.
Held out by link state it reads 0.926 / 0.911 / **0.918**, the same as the
trailing form (0.915) and the hand rule (0.922); on the noisy fold 0.92 /
0.72; on the current fold 0.98 / 0.97. Fit on all 929 units it is
`P(occupied)` in the bank, with `rule B` beside it as the candidate verdict
(its floor the smallest jitter seen so far in the range). The robot, on this
model: 47 % called occupied.

### Verdict

- *Does a small learned classifier on the bank beat the rules?* **By a
  point, at most, and only with dimensionless inputs.** Under the honest
  split the hand rule (0.922), the logistic model on transferable features
  (0.915–0.925) and its six-feature own-floor form (0.918) are
  indistinguishable; the tree with absolute inputs looks best by day and is
  the worst on a new link state.
- *What decides between them* is not accuracy but what each misses: the
  rules keep seated movement (0.83 vs 0.71–0.91) and the learned models keep
  more of the near-presence empties and reject the robot. That is a choice
  for the user, and test 6 (non-human movers) bears on it directly.
- The remaining errors are label-limited: the September windows the camera
  calls empty with a person within a metre or two of the link, and the 09-30
  mornings. No feature set in the bank separates those, and they should be
  adjudicated rather than modelled.
