# FeitCSI Parser

Realtime heatmap for FeitCSI `.dat` captures (Intel AX200/AX210 NIC).

Stack: FastAPI backend parses `.dat` via [CSIKit](https://github.com/Gi-z/CSIKit)
and aggregates it into display-resolution tiles; React + Vite frontend renders
amplitude and phase heatmaps onto a raw `<canvas>`, with
[d3-zoom](https://d3js.org/d3-zoom) driving pan and zoom and
[d3-scale](https://d3js.org/d3-scale) mapping data to pixels. Axes and the
colorbar are drawn directly onto the canvas — there is no charting library.

The same view serves live capture and offline exploration: the backend never
returns more cells than the plot has pixels, so cost tracks the viewport rather
than the file. A 211 MB capture and a 1 MB one open at the same speed.

## Prerequisites

Install once on your system:

| Tool | Why | Install |
|---|---|---|
| [Python](https://www.python.org/) ≥3.12 | Backend runtime | `pyenv install 3.12` or system package |
| [uv](https://docs.astral.sh/uv/) | Python dependency management | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Node.js](https://nodejs.org/) ≥18 | Frontend build | `nvm install 18` or system package |
| [npm](https://www.npmjs.com/) ≥9 | Frontend dependency management | bundled with Node |

## Setup

Backend (Python, `uv`):

```bash
uv sync
```

Frontend + root dev tools (Node, `npm`):

```bash
npm install
npm --prefix frontend install
```

## Run

### Development (single command from repo root)

Both backend (`:8000`) and frontend (`:5173`) in parallel:

```bash
npm run dev:all
```

Open http://localhost:5173

### Development (separate terminals)

Backend on `:8000`:

```bash
npm run dev:backend
```

Frontend on `:5173` (proxies `/api` → `:8000`):

```bash
npm run dev:frontend
```

### Production (single port)

```bash
npm run build         # builds frontend into frontend/dist
npm run serve         # uvicorn serves API + static frontend at :8000
```

Open http://localhost:8000

## Usage

1. Place FeitCSI `.dat` file at `captures/capture.dat` (or enter path in UI).
2. To explore a finished capture, just enter its path — no need to start
   polling. To watch one grow, click **Run realtime**.
3. Every `refresh_ms` the frontend polls `/api/meta`, which reads the frame
   index only and never decodes payloads. Pixels come from `/api/tile`, which
   is fetched only when the view actually changes.
4. Frontend renders eight heatmaps: amplitude (dBm), phase (rad), CSI ratio
   amplitude and phase, then the swap-corrected CSI ratio pair, then the
   time-unwrapped ratio phase, then its channel impulse response (CIR). See
   [Phase views](#phase-views), [Swapped rx streams](#swapped-rx-streams),
   and [Channel impulse response](#channel-impulse-response).

Controls:
- **.dat file** — path to a capture, growing or finished.
- **Refresh (ms)** — polling interval.
- **Run realtime** — toggle polling.
- **Source MAC** — required for the corrected and time-unwrapped views, which
  judge frames against their neighbours. Defaults to a single transmitter.

The four base plots are never modified by any derived view.

Navigation:
- **Wheel** / **drag** — zoom and pan the time axis. Both heatmaps share it, so
  they always show the same instant.
- **Shift + wheel** — zoom the subcarrier axis. This stays per-plot.
- **Double-click** — reset to full extent and resume following the newest
  packet.

Zooming or panning freezes the view (the plot is labelled *frozen*); live polls
then leave it exactly where you put it instead of snapping back. Double-click to
resume following.

## API

The frontend uses `/api/meta` and `/api/tile`. `/api/snapshot` predates them
and is kept for scripted use.

### `GET /api/meta`

Query params:
- `path` — path to `.dat` file

Builds the frame index only — no payload is decoded — so it stays cheap on
large files (48 ms on a 211 MB capture). Returns `filename`, `chipset`,
`bandwidth`, `num_subcarriers`, `total_frames`, `t_min`, `t_max`, `num_rx`,
`num_tx`.

### `GET /api/tile`

Query params:
- `path` — path to `.dat` file
- `t0`, `t1` — time window in seconds, **closed at both ends**
- `width` — output columns, normally the plot width in pixels
- `metric` — one of `amplitude`, `phase`, `csi_ratio_amplitude`,
  `csi_ratio_phase`, `phase_unwrapped`, `phase_detrended`,
  `csi_ratio_phase_unwrapped`, `csi_ratio_phase_time_unwrapped`
  (see [Phase views](#phase-views)), `csi_ratio_phase_corrected`,
  `csi_ratio_amplitude_corrected` (see [Swapped rx streams](#swapped-rx-streams)),
  `csi_ratio_cir` (see [Channel impulse response](#channel-impulse-response))
- `mimo`, `source_mac` — optional filters, `'all'` or a specific value
- `interpolate` — default `true`; see [Interpolation](#interpolation) below

Returns a bare `(num_subcarriers, width)` little-endian float32 array,
row-major, row 0 = highest subcarrier. The body stays a buffer the client wraps
in a `Float32Array`; metadata rides in headers:

| Header | Meaning |
|---|---|
| `X-Tile-Width` / `X-Tile-Height` | Grid shape. Width may be **less** than requested — it is capped at the frame count. |
| `X-Capture-TMin` / `X-Capture-TMax` | The whole file's extent, not this tile's window, so a live view can track growth without a second round trip. |
| `X-Tile-Frames` | Frames decoded (≤ 8192; the range is stride-sampled beyond that). |
| `X-Tile-Total` | Frames in range before sampling. |
| `X-Tile-Exact` | `1` if no stride sampling was needed. |
| `X-Tile-VMin` / `X-Tile-VMax` | Finite extrema. |
| `X-Tile-PLow` / `X-Tile-PHigh` | 1st/99th percentiles — the robust scale the amplitude plot locks to. |
| `X-Tile-Filled` | Columns filled from a neighbouring frame across a sampling gap. |

Columns are max-hold for amplitude and nearest-frame for phase (a maximum of an
angle is meaningless). A column that receives no frame is linearly
interpolated between its two bracketing frames when within 2x the
95th-percentile inter-frame interval; beyond that, or with `interpolate=false`,
it stays NaN, so a real capture dropout stays visible instead of being painted
over. See [Interpolation](#interpolation).

### `GET /api/snapshot`

Predates the tile API and returns decoded values as JSON. Superseded by
`/api/meta` + `/api/tile` for anything interactive — the payload grows with the
window, so it does not stay bounded on large captures.

Query params:
- `path` — path to `.dat` file (default `captures/capture.dat`)
- `max_packets` — trailing window size (default 200)

Returns JSON:
```json
{
  "filename": "capture.dat",
  "chipset": "Intel AX2xx",
  "bandwidth": "80",
  "num_subcarriers": 242,
  "total_packets": 1101,
  "window_packets": 200,
  "time_seconds": [...],
  "amplitude": [[...], ...],
  "phase": [[...], ...],
  "amp_min": 2.7,
  "amp_max": 59.6,
  "phase_min": -3.14,
  "phase_max": 3.14
}
```

### `GET /api/health`

Returns `{"status": "ok"}`.

## Interpolation

One flag, `interpolate` (default `true`), governs filling gaps in two
different axes, and the frontend's **Interpolate** toolbar button toggles
both together:

- **Subcarrier axis.** Structural nulls — pilots, the DC/guard band — are
  filled by interpolation across neighbouring subcarriers within a frame.
  This is `backend.batch.decode_frames`'/`backend.mtk.decode_frames`'
  `interpolate` parameter; see their docstrings for the null-run and
  MAX_NULL_RUN details.
- **Time axis.** A display column with no decoded frame in it — a gap
  between samples, not a real capture dropout — is filled by linear
  interpolation between the two frames bracketing it, weighted by how far
  the column's centre sits between their timestamps. Only gaps within 2x the
  95th-percentile inter-frame interval are touched; a real dropout is wider
  than that and stays NaN regardless of the flag, so turning interpolation
  off never hides one.

`false` leaves both axes exactly as decoded off the wire — every structural
null and every sampling gap NaN. This is the honest view of what the hardware
actually reported; `true` (the default) is the smoothed one most panels are
easier to read in.

The time-axis fill is a plain weighted average for every metric except the
three wrapped-phase ones (`phase`, `csi_ratio_phase`,
`csi_ratio_phase_corrected`). Averaging a wrapped angle directly is wrong at
the branch cut: a frame at +3.1 rad and its neighbour at -3.1 rad are 0.08 rad
apart on the circle, and a plain average lands near 0 rad — the long way
round. Those three metrics are blended as `exp(i*phase)` and converted back
with `atan2`, which follows the circle instead. Every other metric, including
the `*_unwrapped` and `*_detrended` views, is by construction no longer an
angle on a circle and takes the plain average.

## Phase views

Everything the decoder produces comes out of `np.angle`, so the four base
metrics are **wrapped** to (−π, π]. The ±π banding in those plots is the
branch cut, not structure in the channel. They use a cyclic colormap
(matplotlib's twilight) so a wrap does not paint a false hard edge.

Three derived metrics undo parts of that. They are computed per frame on full
subcarrier vectors, before tile column aggregation — aggregation drops frames,
and a phase sequence with holes cannot be unwrapped.

| Metric | Transform | What it fixes |
|---|---|---|
| `phase_unwrapped` | unwrap along subcarriers | Removes the 2π sawtooth *within* a frame. Does nothing across frames. |
| `phase_detrended` | unwrap + per-frame least-squares line removal | Removes the random per-packet offset (CFO/PLL) and the sampling-time-offset slope. This is what makes raw phase comparable across packets. |
| `csi_ratio_phase_unwrapped` | unwrap along subcarriers | Same sawtooth removal for rx1/rx0. |
| `csi_ratio_phase_time_unwrapped` | unwrap along **time**, on the corrected ratio | Removes the sawtooth as the channel moves, so each subcarrier's trace is continuous accumulated phase. This is the motion view. |

The subcarrier-axis metrics remain available over the API but are no longer
plotted; the UI shows the time-unwrapped ratio instead.

Measured on `captures/capture.dat`, mean across-frame standard deviation per
subcarrier: wrapped 1.81 rad → unwrapped **11.60** rad → detrended 0.68 rad.
Unwrapping alone makes raw phase *worse* across frames, because the per-packet
offset and slope are no longer folded back into (−π, π] — which is why the
detrend is a toggle and not applied silently.

Two things the detrend is deliberately not applied to:

- **The CSI ratio.** rx1/rx0 shares an oscillator and clock between the two
  chains, so the division already cancels the common offset and most of the
  slope. Fitting a line there removes signal, not nuisance. (On a MediaTek
  capture the two halves are transmit chains rather than receive ones, which
  cancels the same offsets but leaves a deliberate ramp behind — see
  [MediaTek captures](#mediatek-captures).)
- **Anything needing absolute time-of-flight.** The fit takes any genuinely
  linear-in-frequency component with it. Standard sanitization in the
  SpotFi/PhaseFi lineage, fine for motion sensing, fatal for ranging.

Unwrapped metrics are not angles on a circle any more, so the frontend gives
them a sequential palette and fits the color scale to the first tile's
1st/99th percentile band, exactly as amplitude does. One caveat inherent to
unwrapping: `np.unwrap` anchors each row on its first subcarrier, so a frame
whose lowest subcarrier sits near the branch cut can shift by a whole 2π
relative to its neighbours, appearing as an isolated column jump. The wrapped
panels above are unaffected — that is part of why they stay.

## Swapped rx streams

Two independent corruptions hit the CSI ratio, and they have different
signatures. Both render as inverted-looking colour on a cyclic colormap,
which is why they are easy to confuse by eye:

| | what happens to the ratio | phase | dB amplitude | looks like |
|---|---|---|---|---|
| **swap** | reciprocal (`rx0/rx1`) | negated | negated | isolated columns |
| **rotation** | multiplied by −1 | shifted by π | unchanged | multi-second blocks |

Together they give four states (`r`, `−r`, `1/r`, `−1/r`), and both are
corrected. Measured on one transmitter over 8000 frames of
`csi_20260813_030001.dat`, rotations occur at 62 of 7999 transitions — rare
events, but each one flips a whole block until the next one flips it back.
One such block spanned 2127.9–2135.9 s; correcting it removed all 79 affected
columns from the tile.

The swap's signature is exact: the complex ratio is inverted, which
**negates the phase and negates the dB amplitude** together. On screen it
reads as an isolated column in mirrored colours — visually distinct from
the transparent columns where no frame exists at all.

Measured on one transmitter over 6000 frames, an affected frame deviates from
the mean of its two neighbours by 1.664 rad where a normal frame deviates by
0.116; negating it gives 0.103, back at baseline. The dB amplitude agrees
(5.034 → 0.625). Only the ratio metrics are affected — rx0's own amplitude and
phase are undisturbed.

`csi_ratio_phase_corrected` and `csi_ratio_amplitude_corrected` put them back.
Detection runs on the ratio phase for both, so the two panels always agree
about which frames were flipped.

Effect at short inter-packet gaps, where the channel cannot physically have
moved and any large step is therefore an artefact:

| gap | transitions > 0.5 rad, before | after |
|---|---|---|
| < 2 ms | 4.55% | **1.14%** |
| 80–150 ms | 19.85% | 13.57% |
| > 150 ms | 30.02% | 24.94% |

The residual at longer gaps is largely genuine channel evolution, not missed
swaps.

### The algorithm

Orientation is not observable from a single frame, so every decision is made
by comparing frames against each other. Three phase passes run in a loop, each
covering the others' blind spot, then two anchors settle which way up the
result sits — against a **reference measured once for the whole capture**:

0. **Reference.** Comparing frames to each other can only ever produce an
   answer that is self-consistent *within the batch being looked at*, and
   there are always two such answers. Which one a view lands on then depends
   on which frames the view contains — so panning or zooming inverted whole
   panels, at a measured 12% of positions at a 200-frame zoom. Both anchors
   below originally derived their reference from the batch in front of them,
   which is what made this structural rather than a tuning problem.

   `build_reference` measures the two quantities once, from a few thousand
   frames drawn evenly across the capture: the median dB band profile and the
   mean phase direction. Both are majority statistics and the corruption is
   the minority (4.1% of frames swapped, 14.2% a π out), so raw frames
   already point the right way in bulk and no prior correction is needed to
   measure them — the chicken-and-egg does not arise. Every tile of that
   capture is then judged against the same absolute orientation, and a
   frame's verdict is the same whichever view asked for it. Cost is one
   decode of ~4096 frames per capture per transmitter (~0.2 s), cached.

   It is per *transmitter*, because the band profile is a property of one
   pair of antennas; blending two senders' profiles anchors to neither. So a
   selected `source_mac` is required, and `reference is not None` is the
   single switch on whether the ratio is corrected at all — a view can never
   be half-corrected, nor claim a correction it did not get. No reference is
   also issued when a sender's band is too flat to correlate against or its
   phase names no clear direction. In every such case the ratio is passed
   through exactly as decoded and the tile reports `X-Tile-Anchored: 0`, which
   the heatmap surfaces as *⚠ uncorrected — select a transmitter*.

1. **Chain.** Every adjacent pair is fitted twice — `phi_i` against
   `phi_prev`, and `-phi_prev` — and the better fit wins. Its *offset* then
   says whether a π rotation came along too. Both decisions accumulate as
   parities, so a run of affected frames needs no special handling.
   *Blind spot:* it propagates. One unreadable transition — a dropout, or a
   stride-sampled view where neighbours sit 400 ms apart instead of 100 —
   and everything downstream stays inverted until another miss undoes it.
2. **Refine.** Each frame is re-decided against the circular mean of its
   neighbours, a consensus no single frame can move, so mistakes stay local.
   *Blind spot:* a large inverted region agrees with itself, and the
   symmetric window straddles a boundary and goes incoherent right where it
   matters.
3. **Merge.** Each candidate split point is judged by comparing the mean of
   the frames *before* it against the mean of those *after*. Averaging many
   frames per side lifts the signal far above what any single pair carries,
   so a boundary no adjacent comparison could resolve becomes obvious.
   Non-maximum suppression keeps one detection per boundary.

Scoring uses the correlation's **magnitude** to choose the orientation and
its **angle** to choose the rotation — magnitude alone is offset-blind and
cannot see a rotation at all.

4. **Amplitude anchor.** Everything above compares frames only to other
   frames, which can place a boundary perfectly and still leave the entire
   region *between* two of them inverted — internally consistent, so no
   phase-based check ever objects. The dB ratio amplitude settles it: a swap
   negates it too, and its shape across the band is fixed by the antennas
   rather than the moving channel. Every 2000-frame chunk of an hourly
   capture correlates +0.955 to +0.999 with the file's median profile, so a
   stretch that anti-correlates is simply wrong.

   With a reference the profile comes from the capture rather than from the
   frames being judged, so there is no risk of confirming a window's own
   inversion, no iteration to a fixed point, and no run-length gate — an
   absolute reference can only ever flip a sign that is already wrong. It
   also drops the ≥400-frame minimum, which is what used to leave every
   zoomed-in view with no absolute orientation at all. Without a reference
   the older behaviour stands: only runs of ≥200 frames are re-oriented, and
   isolated frames stay with the phase passes.

   This pass exists because the phase-only version shipped a regression: it
   removed the real isolated swaps and then inverted a 1400-second block on
   top. Across the 20 hourly captures it took frames sitting in the wrong
   orientation from a mean of 4.1% (44.4% on the worst file) to **0.0%**.

   It needs a profile with real shape — below `MIN_PROFILE_STD` it declines
   rather than acting on noise.

5. **Rotation anchor.** The amplitude cannot settle the *rotation* parity,
   because multiplying the ratio by −1 leaves the dB amplitude exactly where
   it was. And rotations are not rare — on an hourly capture ~24% of
   transitions carry a π offset (the distribution is sharply bimodal: 6174
   transitions below 0.3 rad, 1955 between 2.90 and π, only 28 in between),
   so the parity toggles thousands of times and one miscount flips everything
   after it.

   What anchors it is the phase's own mean direction. Each frame's circular
   mean over subcarriers points somewhere, and that direction is set by the
   fixed offset between the antennas rather than the moving channel: measured
   over an hour it holds at +1.3 rad end to end, while a wrongly-rotated
   stretch sits at −1.8. A full π apart, separable by sign. Across the 20
   captures this took columns sitting a π from the capture mean from 14.2% to
   **0.3%**. As with the amplitude anchor, a reference turns this from a
   comparison with the batch into a single pass against a fixed direction.

6. **Stride-sampled views.** A decimated view is not a frame sequence: its
   rows are seconds apart, so the chain and refine passes have nothing to
   compare against and are skipped entirely. The anchors carry the whole
   decision, judging each frame on its own against the reference — and
   *without smoothing*, because smoothing works by borrowing evidence from
   neighbours that share a state, which sampled rows do not. Measured against
   the native-rate answer, this errs on 5% of frames at every stride from 2
   to 64; averaging 5 neighbours errs on 14–23%, and leaving the frames
   uncorrected errs on 28%. What is lost is the isolated single-frame swaps,
   which at those zooms occupy a fraction of one column and cannot be seen.

Windows are corrected with a 128-frame context margin that is then trimmed
off, so the frames at a tile's or a cache block's edge are decided on the same
neighbours a full-capture pass would have given them. Without it a 200-frame
view disagreed with the capture-scale answer somewhere in 28 of 198 positions;
with it, 4 — and none of them an inversion.

Measured across all 20 hourly captures at the full-file view (the worst case,
where stride sampling puts adjacent columns ~4.5 s apart), inverted column
*transitions* fall from **2514 to 4** — and 3 of those 4 are confirmed genuine
channel rotations at full resolution, not misses.

That transition count is a trap worth flagging, because it was believed for
longer than it should have been: a uniformly inverted block has only **two**
boundaries no matter how wide it is, and transitions touching a NaN gap are
skipped entirely. A count of 4 was therefore perfectly consistent with two
enormous inverted regions. The honest metric is the fraction of *frames* whose
orientation disagrees with the amplitude profile, which is what the anchor
above is measured on. Cost is ~0.5 s per round on
a full 8192-frame tile and ~0.04 s on a typical zoomed view.

### Properties worth knowing

Three properties of the method are worth knowing before relying on it:

- **It needs a single transmitter selected, and does nothing without one.**
  Detection is relative — a frame is judged against its neighbours. On
  `source_mac=all` consecutive frames come from different senders (14% and 7%
  same-sender on the two transmitters of an hourly capture), so `_chain`
  compares two senders 86–93% of the time and the confidence gate declines.
  Measured on the same frames, correcting on one transmitter's own sequence
  leaves **0.3–0.6%** of steps above π/2, where correcting the interleaved
  stream leaves **11.2–11.6%** — against 12.4–13.2% uncorrected. It bought
  almost nothing and reported itself as done, so on `all` the correction is
  now skipped outright and the panel shows the raw ratio.
- **The flag means "opposite orientation to frame 0", not "anomalous".**
  Parity accumulates along the sequence, so roughly half the frames in a long
  batch carry the flag even though individual swaps are rare.
- **Orientation needs the capture, not the window.** Which of the two states
  is "correct" is not observable from a single frame and not observable from
  a window either — see step 0. With a reference the answer is a property of
  the capture; without one it falls back to a majority vote over the batch,
  which is stable only as long as the batch is, and which is wrong exactly
  when the minority assumption fails (a window landing inside a long
  corrupted stretch).
- **A genuine π channel rotation is indistinguishable from an artificial
  one** when it lands between two sampled instants. At heavy zoom-out the
  remaining handful of inverted-looking transitions are real events in the
  room, and correcting them would be destroying data. Zoom in and they
  resolve into ordinary continuous motion.

A note on the fitting metric, because it is a trap worth documenting: the
alignment score `|mean(exp(i(x - y)))|` is **invariant to a constant phase
offset**, so a π-rotated block scores a perfect 1.0 against its neighbours
and reads as "identical". That is exactly why rotations went unnoticed at
first. `_fit` therefore returns the offset alongside the quality, and the
rotation decision reads the offset while the swap decision reads the
quality — scoring with the offset folded in would rate a perfectly-explained
rotated frame as unrelated and hide it completely.

The cause of either is unidentified. All 272 header bytes were scanned and
none separates affected frames from normal ones; the documented `antenna_a`
and `antenna_b` bits in `rate_flags` are constant, and bit 20 — the only bit
that varies — does not correlate (its two groups align at 0.999 *as-is*).
No per-frame property identifies them either. Note that CSIKit parses only
about eight fields out of the 272 header bytes, so most of the header has no
known semantics: "nothing found" is not "nothing there".

## Channel impulse response

`csi_ratio_cir` takes the swap-corrected ratio (`backend.ratio`) and inverse-
FFTs it along the subcarrier axis into delay: `backend.cir.ratio_to_cir`.
Where every other panel reads the channel in frequency, this one reads it in
time-of-flight — echoes at different path lengths separate into different
delay taps instead of showing up as ripples across subcarriers.

It is built on the *corrected* ratio for the same reason the time-unwrap is
(see above): an uncorrected swap negates the ratio's phase, and an IFFT would
turn that negation into a spurious second peak rather than leaving a single
clean one.

Two things have to be undone before the IFFT means anything, both because
every metric in this pipeline is already laid out DC-centred rather than in
raw FFT bin order (see [Data Format](#data-format) below):

- **`ifftshift` before the transform.** `np.fft.ifft` expects index 0 at DC
  with positive frequencies ascending and negative ones wrapped to the top;
  the centred array has DC in the middle. Skipping this does not blur the
  result, it relocates every echo to the wrong delay.
- **`fftshift` after it, for display.** Raw IFFT output puts delay 0 at
  index 0, ascending — the ordinary DSP convention, and what
  `backend.cir.ratio_to_cir` returns. But a real channel's true delay rarely
  lands on an exact sample, so the peak's energy splits between tap 0 and
  the *last* tap (a fractional delay just before zero wraps circularly to
  the far end). Measured on both a FeitCSI and an MTK capture, that split
  carries 15-30% of the peak's neighbourhood weight into the wrong-looking
  place — one physical peak rendered as two. `ratio_to_cir_centred`
  (fftshift, delay 0 in the middle) is what `csi_ratio_cir` actually serves,
  and what reunites it: measured the same way, 95-100% of frames then land
  their peak within two taps of the row's centre. It also means the CIR
  panel reuses the same centred axis the frequency-domain panels already
  have — the frontend needs a different *label* (`Delay tap`, via
  `Heatmap`'s `axisLabel` prop) but no different axis logic.

Null subcarriers — the MTK guard band, whatever CSIKit dropped as unusable on
a FeitCSI capture — arrive as NaN and are read as zero energy on that tone
before the transform, which is the standard reading for a punctured
spectrum and is exactly what the MTK hardware's own null-tone encoding
already means. A frame with *no* ratio at all (single-stream) is left NaN
rather than computed as a confident flat zero, which would otherwise be
indistinguishable from "measured, no echoes".

One asymmetry between the two capture formats is worth knowing before
reading fine structure into this panel: MTK's null bins sit at their true
positions in a uniform 256-bin comb, so zero-filling them reconstructs the
transmitted spectrum faithfully. FeitCSI's array has already had its
unusable subcarriers *deleted* by CSIKit rather than zeroed in place, so the
242-wide comb handed to the IFFT there is not perfectly uniform — the result
is still peaked at the true delay but carries extra sidelobe smearing from
the gaps. Good enough to read off relative timing, not to trust to the last
dB, and not comparable dB-for-dB between the two formats in any case — see
[MediaTek captures](#mediatek-captures) for why their ratios are not the
same physical quantity to begin with.

`csi_ratio_cir` uses max-hold aggregation like the other magnitude metrics
(peak-preserving when a display column spans several native frames), and is
exempt from the tile layer's usual "same cells have data" invariant for
derived metrics: a CIR row is a delay tap, not a subcarrier, so there is no
per-cell correspondence to a subcarrier-indexed base to preserve. What does
still hold, cell for cell, is *frame* coverage — a column the base ratio had
no data for gets no CIR either.

## Data Format

FeitCSI `.dat` files are binary: sequence of `272-byte header + CSI block`
records. The header's first word is the payload length, so frames are
self-delimiting and can be decoded as they are appended. CSIKit supplies
header parsing, pilot interpolation, and RSSI scaling.

Subcarriers arrive already centred — index 0 is the lowest subcarrier and
index N/2 is DC. They are **not** in FFT bin order, so `fftshift` must not be
applied: it would split the contiguous spectrum and weld the two outer edges
together.

See https://feitcsi.kuskosoft.com/csi_format/ for the on-wire spec.

### MediaTek captures

Captures pulled off the LG webOS board (`/var/iwtools/iw-priv`, read from
`/proc/net/wlan/csi_data`) are a different format entirely and are detected by
sniffing, not by extension. Records are self-delimiting TLVs —
`magic 0xAC | length u16 LE | tag(1) len(2 LE) value ...` — samples are 14-bit
signed rather than `int16`, and subcarriers arrive in raw FFT bin order, so
here `fftshift` **is** required, the exact opposite of the FeitCSI rule above.
A frame is a *group* of up to four records closed by bit 15 of tag 18, never a
run sharing a timestamp: the millisecond clock ticks mid-group.

**The ratio is a transmit pair.** Records are indexed by `tpi` and `rpi`. The
axes are told apart by the transmitter's cyclic shift, which 802.11 applies
per transmit chain and to nothing else: dividing along `tpi` leaves a ramp,
dividing along `rpi` leaves none. So `tpi` indexes the AP's transmit chains,
and it is `tpi` that gets mapped onto the pipeline's rx axis, because
everything downstream reads the ratio off `rx1/rx0`. **A MediaTek capture's
"CSI ratio" therefore compares two antennas at the far end of the link**,
where a FeitCSI capture's compares two on the receiver. Both cancel the
receiver's CFO/SFO — the two halves come out of one packet, one receive chain
and one timing recovery — but they are not the same physical quantity and
should not be pooled or plotted on a shared scale.

`rpi` plane 1 is real signal, not a dead chain (59.43 dB against plane 0's
59.95 dB, smooth across frequency at 0.995), so it is not obvious from the
file alone what it is on a board documented as 1x1. It is not used, because
its ratio is far noisier per frame: at the shortest frame gap the `tpi` ratio
moves 0.158 rad where the `rpi` ratio moves 0.896 rad, already most of the
1.571 rad a uniformly random phase would give.

**The cyclic shift is removed by default.** Because the two halves of the
ratio are two different transmit chains, the fixed per-chain delay the
standard mandates does not cancel; it survives as a pure linear phase ramp.
It measures 396.6–402.7 ns across all six captures on hand — the −400 ns the
standard specifies for a second stream — with no frame of any file dissenting
on the sign. At 80 MHz that wraps the phase about 30 times across the band,
which is enough to make any statistic taken along the subcarrier axis
meaningless: the raw ratio phase of `capture1.bin` has a circular resultant of
0.010, indistinguishable from uniform, where removing the ramp lifts it to
0.842.

The ramp is measured once per file — not per view, for the reason the
orientation reference is also anchored to the file — and subtracted about each
band's own DC bin, so a 20 MHz and an 80 MHz frame land on one phase
reference. Only `csi_ratio_phase` is affected; `ratio_amp` is unchanged
because the correction is a unit-magnitude rotation, and `amplitude`/`phase`
read rx0 and never see it. Pass `deslope=False` to `mtk.decode_frames` for the
ratio as decoded. A capture whose frames disagree on the ramp's sign, or whose
ramp is shallower than 0.05 rad/subcarrier, gets no correction at all rather
than a number not worth trusting.

Two things to watch when using this ratio. It depends on the AP continuing to
send two streams — 59 of `capture1.bin`'s 1290 groups are single-stream and
have no ratio at all — where a genuine receive pair is always present because
it is your own hardware. And no rate word exists in the format (tag 19 reads 0
on every record), so there is no way to confirm from the file whether the AP
ever applies beamforming; if it did, `tpi` would index precoded combinations
rather than antennas. Nothing in the captures here suggests it happens — no
frame-to-frame jump exceeds 0.664 rad — but it is not provable from the data.
