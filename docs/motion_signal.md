# The motion signal: amplitude → two features → one scalar

`backend/motionsig.py`, `/api/motion-signal`, the **Motion signal** tab.
The front end of a CSI → signal processing → ML pipeline: it reduces a
window of CSI to one number meant for a back-end classifier, and nothing
else. No hold, no breathing channel, no second chance.

Separate from [the hybrid detector](hybrid.md), which answers *is someone
there* from motion **or** breathing. This one answers only *how much is the
channel moving right now*, and a still occupant is a miss by construction.
The two are complementary, not competing.

## Pipeline

```
CSI frames
  └─ RATIO            |H_tx1 / H_tx0| per subcarrier, uniform grid
                      (tiles._presence_grid — the same road the presence
                       detector takes, so a change of decode cannot read
                       as motion)
  └─ per-stream       divide each subcarrier by its own median
  └─ Hampel           11 samples, 3σ; only strays are replaced
  └─ high-pass        0.3 Hz, 4th-order Butterworth  ──► for variance only
  └─ window           2 s, hop 0.5 s
  └─ features         variance   of the high-passed window
                      lag-1 ACF  of the window BEFORE the high-pass
                      ── median over ~245 subcarriers ──
  └─ normalise        per capture, against its own quiet level
  └─ score            fixed logistic weights, fixed threshold
```

Lag-1 is deliberately taken before the filter. A high-pass leaves
neighbouring noise samples anticorrelated, and lag-1 on a filtered empty
room reads that as structure — the plan's §4.2 note, confirmed here.

## Why these choices

Established by the stage-1/2 experiment (`report_motion_signal.md`),
29 captures, capture-level splits, pre-registered predictions.

**Source.** RATIO won on AUC (0.993 against 0.983 for raw amplitude) and on
empty-room stability (CV 0.33 against 0.78). The plan predicted `RAW-PC1`
would be the AGC common-mode component and should be dropped; it is not —
**no** candidate source correlated with common gain above 0.29, so nothing
was dropped on that ground. A single PC of the ratio came within 0.02 of
the full stream set on walking, which by the plan's own tie-break rule
(§3.6, "within 0.02 → take the lighter one") would have won. It was not
taken: it loses by 0.084 on *small* motion, and the stage-1 comparison was
run on walking only, so the tie-break was deciding on the case that does
not matter.

**Features.** Variance and lag-1. MAD, kurtosis, skewness, the low-band
energy ratio, spectral entropy and mean (the control) earned nothing on
top. Lag-1 is the one that carries small motion: on the seated-with-phone
captures it reaches AUC 0.841 against variance's 0.644, because it is scale
free — a random walk and white noise at the *same* variance sit at opposite
ends of it.

**Normalisation.** Per capture, always. This began as a methodology error:
pooling every capture's empty windows into one negative class was measuring
the **link**, not the room. Empty-room variance spans **36×** across the
corpus's sample rates (0.008423 at 42 Hz, 0.0002324 at 43 Hz), and when the
pooling was removed the condition ranking inverted — seated-with-phone
0.575 → 0.860, still-posture 0.868 → 0.527. One capture went from AUC 0.889
to 0.000. Nothing in this pipeline is comparable across captures before its
own capture's scale has divided it.

## The two normalisations

The tab draws both, because the gap between them is what a deployment pays.

| | centre | scale | deployable |
|---|---|---|---|
| `label` | median of the camera-empty windows | their σ | **no** |
| `free` | this capture's 10th percentile | p40 − p10 | yes |

The percentile pair has to sit **low**. At 20/80 the occupant of a capture
40 % occupied is inside the upper percentile and scales their own signal
away: walking recall fell from 100 % to 25.8 % that way — the same
self-defeat the hybrid's motion floor has on a range that is mostly motion.
At 10/40 the gap between the two normalisations nearly closes.

## Weights

Fixed constants, fitted once over the 29 marked captures (`motionsig.CORPUS`)
under each normalisation, class-weight balanced, threshold at 90 %
specificity on the corpus's empty windows. **Not** re-fitted per request: a
panel that fits on the capture it is displaying is reporting its own
training error, and the tab says so when the capture is one of the 29.

```
label   variance +0.135084   lag1 +0.629663   intercept -0.578889   threshold +0.321564
free    variance +0.003270   lag1 +0.630147   intercept -1.059958   threshold +0.439962
```

Note the ratio: lag-1 carries 4.7× the variance weight label-based and 190×
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
