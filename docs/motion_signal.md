# The motion signal: the CSI ratio → two features → one scalar

`backend/motionsig.py`, `/api/motion-signal`. The **Motion signal** tab was
removed from the web UI on 2026-10-01; the module and endpoint remain.
It reduces a window of CSI to one number: *how much is the channel moving
right now*. No hold, no breathing channel, no second chance.

Separate from [the hybrid detector](hybrid.md), which answers *is someone
there* from motion **or** breathing. A still occupant is a miss here by
construction, and covering them is what the hybrid's breathing channel is
for. As a motion **channel** this one is much the stronger of the two —
within-capture AUC 0.878 against the hybrid's 0.606, and 0.96 against 0.46
on a perched occupant — but as a presence **detector** on its own it tops
out near 75 %, because half of what the hybrid knows comes from breathing.
See *Against the hybrid's own motion channel* below.

## Pipeline

```
CSI frames
  └─ RATIO            H_tx1 / H_tx0 per subcarrier, uniform grid
                      (tiles._presence_grid — the same road the presence
                       detector takes, so a change of decode cannot read
                       as motion).  COMPLEX: both halves are used.
  └─ per-stream       magnitude: divide each subcarrier by its own median
                      phase:     unwrap along TIME, restart at each dropout
  └─ Hampel           11 samples, 3σ; only strays are replaced
  └─ high-pass        0.05 Hz, 4th-order Butterworth  ──► for variance only
  └─ windows          variance: 4 s   ·   lag-1: 15 s   ·   hop 0.5 s
                      (one grid: the longer window's centres, the shorter
                       feature read from the window centred at the same
                       instant)
  └─ features         variance  of |ratio|, high-passed        ─┐
                      lag-1 ACF of ∠ratio, BEFORE the filter    ─┤ median over
                                                                 ─┘ ~245 subcarriers
  └─ normalise        per capture, against its own quiet level
  └─ score            fixed logistic weights, fixed threshold
```

Three choices in there were each settled by a measurement that overturned
the plan, and each is a different kind of mistake worth keeping on the page.

**Lag-1 is taken before the high-pass.** A high-pass leaves neighbouring
noise samples anticorrelated, and lag-1 on a filtered empty room reads that
as structure. The plan's §4.2 note, confirmed here — and the sweep confirms
the code obeys it, since lag-1 is identical digit for digit at both corner
frequencies tried.

**The two features ride different windows.** A sweep over 2/4/8/10/15 s
found lag-1 rising monotonically with window length in every condition, most
on the one that matters — still posture, AUC 0.624 at 2 s against 0.756 at
15 s — while variance peaks near 4 s and falls away. A window of T seconds
resolves nothing below 1/T, so a 2 s window cannot see a slow occupant at
all; variance is a spread rather than a rhythm and a longer window averages
a burst into a calm. The cumulative-sum machinery makes the second window
nearly free.

**The high-pass corner was a bug.** At 0.3 Hz under a 2 s window it sat
*below* that window's first non-DC bin (0.5 Hz): the filter was discarding a
band the window could not have reported either way, and every slow occupant
with it. At 0.05 Hz under a 4 s window, still posture goes 0.733 → 0.766 and
perched breathing 0.888 → 0.942. The invariant is now a test: the corner
must stay below 1/window.

## Which half of the ratio

The ratio is a complex number and most of this work used only its magnitude.
Phase is the sensitive half — 2.87 cm of path change is a full 2π at
5.24 GHz, so a chest moving millimetres turns it while leaving the magnitude
flat.

| feature | source | why |
|---|---|---|
| variance | **magnitude** | phase variance is noise-dominated when nobody moves (still-posture AUC 0.595 against 0.766); its fitted weight came out at zero |
| lag-1 | **phase** | held out, it wins — see below |

The lag-1 move is a **swap, not an addition**. Magnitude lag-1 and phase
lag-1 are two views of one thing: fitted together the weight splits between
them (+0.156 / +0.255) and the held-out result is worse than the swap alone.

### Why the held-out comparison was necessary

In-corpus, magnitude won: 88.8 % against 87.9 % median balanced accuracy.
Held out over 200 capture-level stratified splits — weights *and* threshold
fitted on the train half only, paired on the same splits — phase won:

| | in-corpus | held out | overfit |
|---|---|---|---|
| lag-1 on magnitude | 88.8 % | 79.4 % | −9.4 |
| lag-1 on phase | 87.9 % | 81.3 % | −6.6 |

| paired, label-free | Δ | phase wins |
|---|---|---|
| balanced accuracy | +1.90 | 75.0 % of splits |
| accuracy | +3.52 | 84.5 % of splits |

**The ranking of two feature sets can invert when both are scored on the
captures they were fitted to**, and the amount of inversion is the
difference in how much each overfits. Splits are stratified by condition
because condition dominates performance far more than feature choice does:
an unstratified half can put four of the six still captures on one side and
settle the comparison by luck. They are split at the capture level, never
the window level — a 15 s window at 0.5 s hop shares 97 % of its samples
with its neighbour, so a window split puts near-copies of one moment on both
sides and every feature set scores near-perfectly.

## The two normalisations

The tab draws both, because the gap between them is what a deployment pays.

| | centre | scale | deployable |
|---|---|---|---|
| `label` | median of the camera-empty windows | their σ | **no** |
| `free` | this capture's 5th percentile | p25 − p5 | yes |

The percentile pair has to sit **low**, and it has come down twice. At 20/80
the occupant of a capture 40 % occupied is inside the upper percentile and
scales their own signal away: walking recall fell from 100 % to 25.8 % that
way — the same self-defeat the hybrid's motion floor has on a range that is
mostly motion. 10/40 fixed that at a 2 s window. At 15 s it had to come down
again for a second reason: neighbouring windows share 97 % of their samples
and every transition smears 7.5 s either side, so the middle of the
distribution now holds transition windows that used to be a thin tail.

| pair | median balanced accuracy |
|---|---|
| 10/40 | 69.7 % |
| 5/30 | 87.7 % |
| **5/25** | **88.8 %** |
| 5/20 | 87.7 % |
| 10/30 | 78.3 % |

The pair was chosen on the same corpus the weights are fitted to, so treat
88.8 % as the in-corpus figure it is; the held-out number is 79–81 %.

## Weights

Fixed constants, fitted once over the 29 marked captures (`motionsig.CORPUS`)
under each normalisation, class-weight balanced, threshold at 90 %
specificity on the corpus's empty windows. **Not** re-fitted per request: a
panel that fits on the capture it is displaying is reporting its own
training error, and the tab says so when the capture is one of the 29.

```
label   variance +0.142381   lag1 +0.746296   intercept -0.882194   threshold +0.224398
free    variance +0.008124   lag1 +0.372754   intercept -1.181105   threshold +0.206971
```

Note the ratio: lag-1 carries 5.2× the variance weight label-based and 46×
label-free. **The linear model is very nearly lag-1 alone.** Variance
survives as a tie-break.

Regenerate with `python -m scripts.fit_motionsig --captures captures`;
`--holdout STAMP...` leaves captures out of the fit and scores them
separately.

## Measured

Per capture, own scale, asymmetric ±5 s label margin (`backend.truth`), the
weights above. Scenarios are the operator's own labels. `—` is no
denominator: a capture with no empty windows has no specificity, and
reporting 0 % would be a different claim. Grouped by what the person was
doing, sorted by recall inside each group.

### Gross motion — solved

| capture | Hz | occ | scenario | label R / S | free R / S |
|---|---|---|---|---|---|
| 20260915_132712 | 19.6 | 40 % | 빈방 → 서서 돌아다님 → 빈방 | 100.0 / 91.3 | 88.9 / 97.7 |
| 20260921_121406 | 41.7 | 41 % | 빈방 → 서서 돌아다님 → 빈방 | 100.0 / 86.9 | 100.0 / 88.7 |
| 20260910_203337 | 19.6 | 3 % | 의자만 옮김 | 100.0 / 93.6 | 100.0 / 92.5 |
| 20260915_120831 | 19.6 | 5 % | 의자만 오른쪽으로 옮김 | 100.0 / 92.3 | 88.9 / 93.0 |
| 20260915_195845 | 19.6 | 4 % | 의자 오른쪽 1 m 옮김 | 100.0 / 93.0 | 100.0 / 90.4 |
| 20260921_133234 | 43.5 | 8 % | 노트북 거치대 50 cm 이동 | 100.0 / 85.1 | 100.0 / 75.5 |

Walking and the four furniture-move captures (a person entering, displacing
something and leaving) are at **100 % recall in every case, under both
normalisations**, with specificity 85–94 %.

### Small motion — the interesting band

| capture | Hz | occ | scenario | label R / S | free R / S |
|---|---|---|---|---|---|
| 20260922_161219 | 43.5 | 41 % | 빈방 → 걸터앉아서 호흡 → 빈방 | 99.1 / 92.6 | 98.3 / 93.5 |
| 20260922_155219 | 43.5 | 42 % | 빈방 → 걸터앉아서 호흡 → 빈방 | 98.8 / 92.8 | 97.5 / 96.7 |
| 20260911_142929 | 19.6 | 42 % | 앉아서 폰함 | 97.1 / 92.2 | 89.5 / 94.9 |
| 20260911_140353 | 19.6 | 42 % | 앉아서 폰함 | 82.2 / 86.0 | 61.8 / 95.1 |
| 20260911_144658 | 19.6 | 42 % | 앉아서 폰함 (말도 함) | 80.6 / 91.0 | 67.9 / 97.0 |
| 20260911_153305 | 19.6 | 42 % | 앉아서 폰함 / 테이블에 물체 추가 | 73.9 / 82.8 | 37.0 / 96.7 |
| 20260911_150435 | 19.6 | 42 % | 앉아서 폰함 | 67.8 / 94.2 | 61.5 / 94.8 |
| 20260922_151145 | 43.5 | 31 % | 빈방 → 의자에 앉아서 폰봄 → 빈방 | 65.6 / 87.3 | 48.3 / 95.2 |
| 20260911_100145 | 19.6 | 41 % | 앉아서 폰함 | 26.4 / 90.1 | 33.6 / 88.3 |
| 20260911_095127 | 19.6 | 42 % | 앉아서 폰함 | 17.0 / 87.2 | 22.4 / 84.1 |

**The spread inside one nominal condition is the result that needs
explaining.** Six captures labelled 앉아서 폰함 at the same rate, the same
distance and the same occupancy fraction run from 17 % to 97 % recall. A
rate effect is ruled out — the extremes are both 19.6 Hz. What the label
does not record is how much the person actually moved while holding the
phone, and that is the variable the signal is measuring. The label is
coarser than the phenomenon; a future protocol that wants to predict this
number has to record motion, not posture.

The two 걸터앉아서 호흡 captures reaching 99 % are the counter-example to
reading this pipeline as motion-only: perched and breathing, they land with
walking rather than with the still-posture group below.

### Still — outside this pipeline's reach

| capture | Hz | occ | scenario | label R / S | free R / S |
|---|---|---|---|---|---|
| 20260916_202702 | 19.6 | 91 % | 정자세 → 심호흡 → 빠른 호흡 | 64.1 / 86.5 | 16.7 / 100.0 |
| 20260915_150502 | 20.0 | 64 % | 앉아서 숨만 쉼 → 15초 숨참음 | 34.3 / 85.3 | 23.8 / 87.2 |
| 20260915_143211 | 19.6 | 42 % | 책상에 옆으로 누워있음 | 30.7 / 89.9 | 22.8 / 93.0 |
| 20260915_133849 | 19.6 | 42 % | 스탠드 옮긴 후 2분 앉아 있음 | 28.5 / 87.8 | 22.1 / 88.4 |
| 20260909_195450 | 19.6 | 48 % | (비고 없음) | 26.1 / 88.9 | 21.0 / 89.6 |
| 20260904_193228 | 19.6 | 40 % | 앉아서 가만히 있음 | 23.3 / 92.1 | 20.7 / 95.3 |
| 20260916_143259 | 19.6 | 99 % | 정자세 폰X | 13.3 / 100.0 | 22.2 / 100.0 |
| 20260917_195828 | 41.7 | 35 % | 빈방 → 정자세 → 빈방 | 12.4 / 89.1 | 21.4 / 85.6 |
| 20260917_202029 | 41.7 | 100 % | 옆으로 앉은 정자세 5분 | 11.8 / — | 17.7 / — |

Recall 12–34 %, except the deep-breathing capture at 64 %. **This is not a
failure of this pipeline** — a motionless person produces no motion, and
covering them is what the breathing channel in [the hybrid](hybrid.md)
exists for. It is recorded here so nothing downstream is built on the
assumption that a motion feature sees a still occupant.

### Empty

| capture | Hz | scenario | label S | free S |
|---|---|---|---|---|
| 20260914_193002 | 19.6 | 빈방 | 97.3 | 93.5 |
| 20260915_202020 | 19.6 | 빈방 | 90.8 | 92.6 |
| 20260916_140908 | 19.6 | 빈방 | 89.7 | 81.0 |
| 20260921_125836 | 43.5 | 빈방 (낮) | 85.1 | 74.9 |

### Corpus

| | label | free |
|---|---|---|
| median balanced accuracy | **79.7 %** | **75.0 %** |
| mean balanced accuracy | 77.1 % | 74.6 % |
| captures with both denominators | 24 | 24 |
| pooled windows | 16 449 | 16 449 |

Specificity holds at **85–100 %** across every condition and both
normalisations — the part that survives without labels. Recall is what the
condition decides, and the label-free variant costs 0–37 points of it
depending on how much quiet the capture contains.

## Limits

- **The corpus is 29 captures**, and the effective sample is smaller than
  16 449 windows suggests: a 2 s window at 0.5 s hop shares three quarters
  of its samples with its neighbour. The numbers above are training numbers
  for those 29.
- **The model is linear**, and that costs measurably. Lag-1 alone reaches
  AUC 0.841 on the seated captures while the fitted linear combination
  converts that into 62.9 % two-class accuracy: the interaction that
  separates *high variance with high lag-1* (a person) from *high variance
  with low lag-1* (an impulsive artefact) cannot be written as a sum. A
  hand-made product term is the cheap next test; a shallow GBM needs more
  captures than exist.
- **No non-human motion in the corpus.** Fan, curtain and robot-vacuum
  captures are zero, so the case a linear model provably cannot express is
  also the case there is no data for. Note that the four furniture-move
  captures are *not* that case: a person was in the room in all four, and
  the camera called them occupied.
- The label-free scale needs the range to contain some quiet. A range that
  is nothing but motion has no low percentile to find, exactly as the
  hybrid's own-floor rule does.
