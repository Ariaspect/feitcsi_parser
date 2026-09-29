# The frame step: the amplitude difference on a bounded −1..1 axis

`backend/framediff.py`, `/api/frame-diff`, on the **Motion & presence** tab
behind the `frame step` toggle. Evidence only — nothing in that tab's verdict
reads it, and it does not vote.

The oldest presence signal in this project is the board's own: difference the
raw per-subcarrier amplitude against the frame before, fire when the step is
large. It is still here twice — `vendor/csi_dump_parsing.py` runs it live at a
26 dB trigger and the **Phase 1** tab scores that, and
`hybrid.amplitude_diff` computes the median `|ΔA|` in dB as the hybrid's
amplitude channel. What neither offers is a number a reader can place without
knowing the link, because a dB step is unbounded above.

This panel keeps the difference and changes only the axis:

```
d = (a_t − a_{t−1}) / (a_t + a_{t−1})        a = linear amplitude, per subcarrier
```

bounded in (−1, 1) by construction — +1 a subcarrier appearing from nothing,
−1 one vanishing into it, 0 no change — with no reference, no floor and no
per-room constant.

## The frame set: one transmitter, one mode, one bandwidth

A step is only a step between two frames of the **same shape**, so the frame set
is made uniform before anything is differenced. Left alone it resolves to the
capture's dominant peer and `STRICT_MIMO = (2, 1)` at full width — on the lg
captures, `08:bf:b8:95:80:04`, 2x1, 80 MHz — and an explicit `mimo` or
`source_mac` from the caller is honoured instead. What was used comes back as
`source_mac` / `mimo` / `selection_note` and is printed under the chart.

Why it is not optional (measured over three captures):

| | steps | median `\|d\|` | live subcarriers |
|---|---|---|---|
| both frames 80 MHz | 5 593–12 655 | **0.012–0.044** | 245 |
| across a bandwidth change | 54–91 (0.6–0.9 %) | **0.220–0.385** | 57 |

Nine to eighteen times the level, judged on a quarter of the array, because a
narrow frame is centred and NaN-padded into the wide row so only the middle bins
overlap at all. They are rare, and they are the **extreme tail**: on
20260916_143259 every single bandwidth-change pair exceeds the whole same-width
series' maximum. That is a larger per-step artifact than the AGC, and unlike the
AGC it is pure bookkeeping — those pairs are two bandwidths, not two moments.

What the rule costs and buys, per capture:

| capture | frames kept | p99 | max |
|---|---|---|---|
| 20260921_121406 | 12 653 / 12 967 | 0.1989 → **0.1386** | 0.6748 → 0.6748 |
| 20260916_143259 | 5 621 / 5 737 | 0.2644 → **0.2008** | 0.5454 → **0.2922** |
| 20260921_030003 | 12 517 / 12 831 | 0.1702 → 0.1689 | 0.4923 → **0.1950** |
| 20260915_202020 | 5 656 / 5 669 | 0.1135 → 0.1116 | 0.3870 → **0.1518** |

The loudest step in the series drops 46–61 % on three of four captures. The
mode is the load-bearing filter: every (2, 1) frame measured is full width,
while 43–51 frames per capture are full width at (1, 1), so the width check is a
guard rather than the filter. A capture that never used 2x1 falls back to its own
most common mode and says so, rather than coming back empty.

## It is the same measurement, not a different one

Substituting `a_t / a_{t−1} = 10^(ΔdB/20) = e^u`, `u = ΔdB·ln10/20`:

```
d = (e^u − 1)/(e^u + 1) = tanh(u/2) = tanh(ΔdB · ln10 / 40)
```

exactly. The bounded axis is a monotone squash of the dB step: nothing added,
nothing lost, the ordering of steps preserved, every threshold carried across.
26 dB lands at **d = 0.9045**.

A median commutes with a monotone map, so the median fold over subcarriers here
is the `tanh` of the median fold in dB — this panel's `|step|` trace **is**
`hybrid.amplitude_diff`, on a different scale rather than as a different
series. `tests/test_framediff.py` asserts both identities, the second by
running the two modules over one capture and comparing step for step.

## Three signals

`signal=` picks what gets differenced. The arithmetic is the same in each case;
only the series changes.

| signal | series | reads back as |
|---|---|---|
| `amplitude` (default) | raw \|H\| of tpi slot 0 — what the board differences | dB |
| `ratio_amp` | \|H_tx1/H_tx0\| in dB — the grid every other detector on the tab rides | dB |
| `ratio_complex` | the same ratio kept complex | degrees of rotation |

The complex channel is the interesting one, and it **generalises** rather than
replaces the others. A complex difference has a direction rather than a sign, so
it decomposes into exactly the two traces the chart already draws:

```
signed  = (|r| − |r′|) / (|r| + |r′|)     the radial part — what ratio_amp shows
|step|  =  |r − r′|    / (|r| + |r′|)     the total, bounded by the triangle inequality
```

`|step| ≥ |signed|` always, so **the gap between the traces is the phase
rotation**. With the phase held still the two coincide and equal the amplitude
form to floating point; for a pure rotation by θ at constant magnitude the chord
over the sum gives `|step| = sin(θ/2)`, so a step reads back as an angle. That is
the case [`backend/motion.py`](../backend/motion.py) exists to argue about: a body
at fixed range walks the ratio round a circle, the magnitude never moves, and an
amplitude-only difference is blind to it.

### Measured, occupied / empty within each capture

| group | `amplitude` | `ratio_amp` | `ratio_complex` |
|---|---|---|---|
| walking (2) | 3.87× | 4.36× | **4.55×** |
| phone / standing (3) | 0.73× | 1.03× | 1.03× |
| still sitter (2) | 0.89× | 0.99× | 1.00× |

The ratio channels are better, and the win is not mainly the extra separation on
walking — it is that they stop being **inverted**. On raw amplitude, phone and
still-sitter captures read *quieter* occupied than empty (0.73×, 0.89×), which is
worse than useless; on the ratio they sit at 1.0. Neither still case separates on
any signal, which is the approach's floor, not a tuning problem.

`ratio_complex` beats `ratio_amp` by very little (4.55× against 4.36× on
walking). Per frame at ~42 Hz the rotation a small movement produces is close to
the ratio phase's own frame-to-frame noise, so the extra half of the signal is
mostly not paying yet.

### Against `|dr|/|r|` — it is the same metric (2026-09-28)

Asked directly, and the answer is not the flattering one. `presence.fractional_motion`,
the hybrid's motion channel, computes

```
|r − r′| / [½(|r| + |r′|)]        midpoint denominator
```

and `ratio_complex` computes

```
|r − r′| /  (|r| + |r′|)          sum denominator
```

**One formula, mine exactly half.** Verified per subcarrier over 14 captures:
the ratio between them is 0.500000, min and max, every capture. So the
boundedness is not an advantage this panel introduced — `fractional_motion` is
bounded by 2 for exactly the same triangle-inequality reason, and reading its
axis as ±1 is a rescale, not a new quantity.

What actually differs:

| | this panel | `fractional_motion` |
|---|---|---|
| fold over subcarriers | median | mean |
| time base | native frame times, no resampling | uniform grid, interpolated |
| a pair spanning a dropout | blanked | interpolated through |
| frame set | one MAC, one MIMO mode, one width | whatever the caller filtered to |
| decimation | envelopes, per-frame extremes kept | per-second median |
| readback | dB, or degrees of rotation | none |

The fold is a wash on this corpus: within-capture IQR/median 0.04–0.19 for the
median fold against 0.04–0.26 for the mean, cross-capture spread 9.6× against
10.5×, and identical recall. The rest are plumbing improvements — real, and they
belong in both, but they are not a better metric.

### Stable enough for detection? Calibratable?

Per second, over camera-empty seconds:

| capture | level | IQR / median |
|---|---|---|
| 20260921_020003 / _030003 / _050002 | 0.0104 | 0.04 |
| 20260921_125836 | 0.0125 | 0.06 |
| 20260915_202020 | 0.0102 | 0.07 |
| 20260922_135920 | 0.0187 | 0.19 |
| 20260916_140316 | 0.0849 | 0.18 |
| 20260917_201323 | 0.0983 | 0.12 |

**Stable within a capture, not across them.** An empty room's per-second level
wanders 4–19 % of itself, which is steady enough to threshold against — but the
level itself moves **9.6×** between captures.

So an absolute threshold does not calibrate. Fitting one at the 99th percentile
of one capture's empty seconds and applying it to the others: median specificity
98.8 %, but **34 % of pairs fall below 90 %, and the worst is 0 %**. Most pairs
are fine and a third are catastrophic, which is the worst possible shape for a
fixed number.

The relative rule the hybrid already uses — 2× the range's own 20th percentile —
gives **100 % empty specificity and 9 % occupied recall** per second on this set
(both folds, identically). That recall is the motion channel alone with no burst
rule, no hold and no breathing channel; it is why the hybrid has all three.

**Conclusion.** This panel is a better *instrument* for reading the motion
quantity — bounded axis, honest frame set, no interpolation, per-frame detail,
an angle readback — and not a better detector input. Anyone reaching for it as
one should use the hybrid's own-floor rule, because the number itself calibrates
no better than `|dr|/|r|` does, being the same number.

### The common-mode fold (2026-09-29)

A third fold, from the user's own formula — `y = (1/N) Σ (A_i − A_{i−1})`. The
mean of a difference is the difference of the means, so it is the frame-to-frame
change in the array's **average level**: the channel's **common mode**. On a
complex series the mean is taken as a vector, so steps agreeing in direction add
and opposing ones cancel.

Which makes it the sharpest fold on the ratio and the worst on the raw
amplitude, for one reason: **a receiver gain step is pure common mode**, while a
body's effect is differential across subcarriers and largely cancels in a mean.

Separation, 6 sitting captures (09-21 20:17–20:44) against 3 same-day empties:

| fold / signal | sitting | empty | separation |
|---|---:|---:|---:|
| common mode, **raw amplitude** | 0.263 dB | 1.209 dB | **0.22× — inverted** |
| median \|d\|, ratio complex | 0.0853 | 0.0105 | 8.10× |
| common mode, ratio \|r\| | 0.0174 | 0.0018 | 9.95× |
| **common mode, ratio complex** | **0.0326** | **0.0025** | **13.16×** |

All the ratio forms are clean (no overlap); the raw-amplitude one inverts
because the night empties re-gear on 52 % of pairs. Gain-crossing inflation of
the common fold on raw amplitude reaches **68×** on 20260921_125836, against
1.0–4.4× for the median fold on the same captures — the mean is exactly the
wrong fold for a series the AGC can move.

Two properties asserted in the tests: a step every subcarrier makes together
survives the mean untouched, and half the array up with half down cancels to
zero while the magnitude fold reads it at full size. Opposing *rotations* are
the interesting middle case — they cancel to a tenth rather than to nothing,
because +θ and −θ share the radial component (1−cos θ)/2.

### AGC immunity, measured rather than assumed

Median `|d|` at a gain crossing over the same-state level:

| capture | `amplitude` | `ratio_amp` | `ratio_complex` |
|---|---|---|---|
| 20260921_030003 (night, empty) | 4.41× | **1.02×** | **1.02×** |
| 20260916_143259 (still sitter) | 5.39× | **1.02×** | **0.99×** |
| 20260921_121406 (walking) | 4.45× | 3.44× | 3.76× |

The common gain divides out of the ratio, and on the empty and still captures a
crossing is indistinguishable from any other pair. The walking capture is the
exception and probably not a leak: there the gain is changing *because* the
person is moving, so the crossings really do sit on louder moments.

## What is drawn

| series | what it is |
|---|---|
| signed median | median over live subcarriers of `d`; + the array brightening, − fading |
| signed envelope | the column's min and max, so one frame survives decimation |
| `\|step\|` median | median of `\|d\|`, which cannot cancel — the hybrid channel |
| `\|step\|` peak | the column's largest `\|d\|` (drawn only when columns hold more than one frame pair) |

Per-frame, so a 600 s range at 42 Hz is 25 000 steps and no plot has pixels for
them. Columns are equal in time, one per pixel, and carry **envelopes rather
than averages**: the point of a per-frame signal is the single frame that
moved, and a mean over twelve neighbours is exactly what hides it.

**No threshold line.** The board's 26 dB was drawn here at first and measuring
it removed the reason to — see below: it sits above every step this fold can
produce. A step is still reported in dB (`median_db`, `max_db`), which needs no
threshold.

**Pan and zoom.** Drag pans, the wheel zooms about the cursor, down to a 0.5 s
floor (≈20 frame pairs at 42 Hz), bounded by the panel's own range because that
is what the detector above it was computed over. A zoom **refetches** rather
than stretching: at 300 s the response is 880 decimated columns, at 10 s it is
429 columns of one frame pair each, and at the floor it is 21 individual pairs.
That is the only way a per-frame trace can actually be read. The refetch is
debounced 180 ms so a drag costs one or two requests, and while zoomed the chart
carries its own time axis and says so — it is off the shared axis the charts
above it share, and `reset zoom` puts it back.

The plot is clipped to its own box. Without that the step trace overflows it by
design: the steps cover the whole decoded range while the window-centre series
above start half a window in, which on a 300 s range with a 30 s window is
14.8 s (46 px) past the left edge and 15.7 s (49 px) past the right.

Steps spanning a dropout are blanked, not bridged — `doppler.gap_limit_for`,
the same limit the spectrogram and the presence grid use.

**Raw amplitude, no AGC table**, matching the board and `hybrid.amplitude_diff`.
Unlike that per-second median, this is per frame and the receiver's gain control
reaches it intact — which turned out to be the dominant effect on the tail, see
below.

## Measured, 2026-09-28

21 captures, whole capture, each step attributed to its camera second through
`truth.cell_truth` on the 1 s grid with the 5 s margin. `|d|` is the median
fold; dB is the same number back through `arctanh(d)/(ln10/40)`.

| group | captures | empty `\|d\|` median | occupied `\|d\|` median | occupied / empty |
|---|---|---|---|---|
| empty · night, 42 Hz link | 3 | 0.0106–0.0768 | — | — |
| empty · daytime, declared | 3 | 0.0050–0.0099 | — | — |
| empty · the noisy ones | 3 | 0.0265–0.0388 | — | — |
| walking around | 2 | 0.0064–0.0088 | 0.0229–0.0364 | **3.6–4.2×** |
| phone / standing | 3 | 0.0065–0.0102 | 0.0066–0.0068 | 0.65–1.05× |
| still sitter | 2 | 0.0305–0.0614 | 0.0297–0.0494 | 0.81–0.97× |
| seated fidgeting (fully occupied) | 4 | — | 0.0351–0.0428 | — |

What the table says, in order:

1. **Walking separates, by 3.6–4.2×** against the same capture's own empty
   seconds. This signal can see travel and nothing here suggests otherwise.

2. **Phone and standing do not separate at all** — 0.65–1.05×, and two of the
   three read *lower* occupied than empty. A per-frame amplitude difference has
   no persistence and no breathing channel, so this is the expected floor of
   the approach rather than a tuning problem.

3. **A still occupant is invisible**, 0.81–0.97×, as it must be: a motionless
   body displaces the channel rather than modulating it, which is what the
   channel-offset panel above is for.

4. **The empty level spans 15.5× across captures** (0.0050 to 0.0768, 0.09 to
   1.34 dB) — and ~7× *within the night group alone*, where 05:00 reads 0.077
   against 0.010 at 02:00 in the same room on the same link. So no absolute
   threshold on this axis transfers, exactly as
   [the hybrid's floor note](hybrid.md) found for `|Δr|/|r|`. Bounded is not the
   same as calibrated. **Re-checked with the gain gate on** (see the bimodality
   note below, which is why): 0.0047 to 0.0765, **16.4×**. The conclusion is not
   an AGC artifact.

5. **The fidgeting sitters sit at 0.035–0.043**, which is above the daytime
   empties (0.005–0.010) but inside the range of the *noisy* empties
   (0.027–0.039) and of the still sitters. This signal does not separate the
   unsolved case either; it relabels it in bounded units.

## The AGC is not negligible — it owns the tail (2026-09-28)

Asked directly, and measured over the same 21 captures. The gain state is taken
as the reported `rssi_1`, which is what `backend.agc` uses as its own state
proxy.

| what | measured |
|---|---|
| frame pairs crossing a gain state | **10–52 %** (median ≈ 27 %) |
| frames outside the dominant state | 9–57 % |
| median `\|d\|` at a crossing vs a same-state pair | **1.0–9.5× larger** (median ≈ 3.5×) |
| share of the loudest 1 % of steps that are crossings | **84–100 %** |
| …against a base rate of | 10–52 % |

So the tail of this trace — the part any per-frame trigger would fire on — is
the receiver rather than the room. A per-second median absorbs a single-frame
stripe and this signal cannot, which is the price of reading single frames.

**Two things do not fix it.**

*The AGC correction* (`backend.agc`, per-gain-state shape offsets, applied
through the same block path the amplitude heatmap uses):

| | median `\|d\|` | p99 | max |
|---|---|---|---|
| change from raw | −1 % to +26 % (mostly under 2 %) | **+12 % to +125 %** | ≈ unchanged |

It leaves the level alone and makes the tail *worse*. The correction is
per-frame and shape-only, so differencing two differently-corrected frames
introduces a step that was not there — the same jitter `hybrid` records for its
amplitude channel. Its effect on separation is erratic (0.74× → 4.34× on one
capture, 0.81× → 0.73× on another), which is what a correction that adds
variance looks like. **Amplitude stays raw.**

*Inferring the gain step from the fold.* The plausible idea — a gain step moves
the whole array one way, a body leaves subcarriers disagreeing — does not
survive measurement. `|signed| / |d|`, the share of the step that survives the
signed fold:

| capture | at a gain crossing | at a same-state step |
|---|---|---|
| 20260921_030003 (empty) | 0.972 | 0.967 |
| 20260921_121406 (walking) | 0.616 | 0.551 |
| 20260916_143259 (still sitter) | 1.000 | 0.454 |
| 20260917_202029 (fidgeting) | 0.767 | 0.730 |

One capture of four separates; the rest do not. `fraction_above` is no help
either — at 26 dB its median is 0.0000 on every capture and class, since the
median step is nowhere near the line. **An earlier version of this panel's
caption claimed both diagnostics worked. They do not, and it no longer says so.**

### The 42 Hz captures are bimodal, and a median can land in the void

Found while checking the frame-set change, and it revises how these numbers
should be read. On the 09-21 night captures the receiver changes gain state on
**~52 % of consecutive pairs**, which splits the step series into two clusters
about six times apart — same-state pairs near 0.009–0.010 and crossings near
0.05–0.08 — with almost nothing in between:

| capture | p25 | **p50** | p75 | share < 0.02 | share > 0.05 | p50 gated |
|---|---|---|---|---|---|---|
| 20260921_020003 | 0.0039 | 0.0110 | 0.0808 | 51.4 % | 47.8 % | 0.0098 |
| 20260921_030003 | 0.0040 | **0.0443** | 0.0810 | 49.7 % | 49.4 % | **0.0112** |
| 20260921_050002 | 0.0059 | 0.0771 | 0.0917 | 37.7 % | 61.6 % | 0.0765 |

With the mass split ~50/50 the median sits in the empty gap between the
clusters, so a few percent of composition swings it four-fold: 20260921_030003
reads 0.0443 ungated and 0.0112 gated, while its neighbour an hour earlier
reads 0.0110 either way. **The 0.0443 in the table above is that instability,
not a noisier room.** The extremes of the night group (02:00 and 05:00) are
stable and their ~7× difference survives gating at 7.8×, so the spread finding
stands; it is the middle value that could not be read off a whole-series median.

For a capture whose gain dithers like this, read the gated median or the
quartiles — not the ungated median.

**What works is not an inference.** The gain state is reported per frame, so the
crossings can simply be blanked, the way a pair spanning a dropout already is —
`gate_gain`, `?gate_gain=true`, the `gain gate` button:

| | measured |
|---|---|
| steps kept | 48–88 % (median ≈ 73 %) |
| median `\|d\|` | −1 % to −38 % |
| **p99** | **−0.3 % to −82 %** (median ≈ −50 %) |
| occupied / empty separation | unchanged (walking 4.16→4.93, 3.58→3.26; phone and still sitters unmoved) |
| spread of the empty level across captures | **15.5× → 16.2×, i.e. no gain** |

The last row is the important one: halving the tail does not make the signal
transferable, because the 15.5× spread between links was never the AGC. The gate
is a cleaner *reading* of the same signal, not a fix for the thing that stops an
absolute threshold from working.

`gate_gain` is **off by default**, and `n_gain_crossed` is reported either way —
the measurement arrived after the panel did, and a measurement does not get to
silently change what the panel showed.

## The board's 26 dB line, and why the panel no longer draws it

The loudest **median** step anywhere in the 21 captures is 0.675 = **22.5 dB**,
so the 26 dB trigger is above every step recorded here and cannot fire on this
fold at all. A line at 0.9045 on an axis whose data never leaves ±0.7 marks
nothing, and the per-subcarrier share crossing it (`fraction_above`) had a
median of 0.0000 on every capture and class — a counter that only ever read
zero. Both were removed from the module, the endpoint and the panel; the finding
below is why, and it is kept here rather than in the UI.

What fires the live detector is its per-subcarrier count (`≥ 2` subcarriers past
the threshold), and that is the trouble: **14–26 % of the array crosses 26 dB
in a single step, in empty rooms as readily as occupied ones** (0.14–0.26 of
live subcarriers, measured on both classes). That is where the Phase 1 tab's
false positives come from — not from a threshold set too low, but from a
per-subcarrier vote on a quantity whose tails are the same with and without an
occupant.

## Status

No default changed, and nothing scores a verdict from this. It is a panel and
an endpoint. Whether the bounded axis earns a place in a detector is open; on
this evidence it is a better way to *read* the amplitude channel, not a better
channel.
