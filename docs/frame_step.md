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
   1.34 dB) — and 7.3× *within the night group alone*, where 05:00 reads 0.0768
   against 0.0106 at 02:00 in the same room on the same link. So no absolute
   threshold on this axis transfers, exactly as
   [the hybrid's floor note](hybrid.md) found for `|Δr|/|r|`. Bounded is not the
   same as calibrated.

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
