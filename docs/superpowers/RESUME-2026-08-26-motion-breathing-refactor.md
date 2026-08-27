# Resume: motion / breathing split on the CSI ratio

Branch `feat/presence-reference-baseline`, six commits, stages 2-5 of the
refactor done. Stages 6-8 are not started.

## Where to pick up

§6 three-state classification, §7 automatic validation, §8 output
consolidation. One decision is already forced and should be made before §6
is written — see "The classification table does not survive contact" below.

## What each stage produced

| Stage | Module | Tests | Script |
|---|---|---|---|
| §2 preprocess | `backend/preprocess.py` | `tests/test_preprocess.py` (16) | `scripts/stage2_preprocess.py` |
| §3 motion | `backend/motion.py` | `tests/test_motion.py` (12) | `scripts/stage3_motion.py` |
| §4 breathing | `backend/breathing.py` | `tests/test_breathing.py` (16) | `scripts/stage4_breathing.py` |
| §5 Doppler | `backend/spectro.py` | `tests/test_spectro.py` (13) | `scripts/stage5_doppler.py` |

Full suite: 388 passed, 10 skipped.

Each script reads the previous stage's `.npz` rather than re-decoding, so the
figures redraw without touching a capture. Artefacts land in `artifacts/`
(gitignored, ~106 MB).

```bash
MPLCONFIGDIR=.matplotlib-cache uv run python scripts/stage2_preprocess.py captures/lg/20260825_185637.bin
MPLCONFIGDIR=.matplotlib-cache uv run python scripts/stage2_preprocess.py captures/lg/20260826_0400-0500.bin --t0 1200 --t1 1800
MPLCONFIGDIR=.matplotlib-cache uv run python scripts/stage3_motion.py artifacts/stage2/20260825_185637_stage2.npz \
    --empty-npz artifacts/stage2/20260826_0400-0500_stage2.npz
MPLCONFIGDIR=.matplotlib-cache uv run python scripts/stage4_breathing.py artifacts/stage2/20260825_185637_stage2.npz --no-gate
MPLCONFIGDIR=.matplotlib-cache uv run python scripts/stage5_doppler.py artifacts/stage2/20260825_185637_stage2.npz
```

## Captures (local only, `captures/` is gitignored, ~313 MB)

- `captures/lg/20260825_185637.bin` — the measurement. Occupied from t=0,
  occupant leaves at 383 s. **No entry event**, which is why the §5.3
  opposite-sign test cannot run on it.
- `captures/lg/20260826_0400-0500.bin` — 04:00, nobody home. The empty
  control that every threshold is calibrated against. Slice 1200-1800 s.

Both came from `scp cyphy:/home/lg_csi/feitcsi_parser/captures/lg_csi_captures/...`.

## Settled decisions

- **Both static-component readings are kept** (Q1). `preprocess.remove_static`
  strips it for the motion and breathing paths; `backend.presence` keeps it as
  the channel offset that detects a motionless occupant. Not in conflict —
  different questions.
- **Sample rate is derived, never assumed** (Q2). Nominal 20 Hz (50 ms ping),
  delivered 18.12 Hz. `derive_sample_rate` warns on the gap and on the
  bimodal interval (50 ms 23.4%, 56 ms 24.1%). Every rate-dependent constant
  is in `preprocess`'s constants block.
- **Carrier is 5210 MHz** (Q3), so `WAVELENGTH_M` = 5.754 cm. Not recoverable
  from the capture — the headers carry bandwidth only.
- **The noise floor is calibrated from the night capture**, not from the
  measurement capture's tail. 80 s of tail gives the same mean with twice the
  spread (sd 0.153 against 0.075) and a threshold of 0.657 instead of 0.403.

## The classification table does not survive contact

§6's table decides "still occupant" on clear respiration. Two independent
methods say that does not work at this geometry:

- Autocorrelation (§4): confidence separates occupied from an empty control
  at **65.1%**, and detection and false alarm move together at every
  threshold (0.4 → 17.6% / 12.3%; 0.673 → 4.5% / 0%). Rate continuity, the
  last missing piece of the spec's confidence recipe, adds nothing (66.4%).
- Doppler sidebands (§5): none found in any regime once the detrend slope is
  removed. Occupied scores prominence 2.31 against an empty room's 2.10.

What respiration *can* do: when it fires during occupancy the rate is
coherent — 14.16 rpm, p10-p90 of 10.4-14.7, 87% inside 12-20 rpm. So it is
evidence, not a verdict.

What should carry "still occupant" instead is the channel-offset path already
on this branch (`backend.presence`, commits `8ef24e1` / `47ce354`): 2.92 dB
occupied against 0.36 dB empty on a 1.53 dB threshold, exit at 383 s, no
window misclassified. §6 should be written around motion + channel offset,
with respiration reported alongside.

## Open items

- **§5.3 sign check needs a faster capture.** Maximum unaliased radial
  velocity is `lambda/2 * Nyquist` = 0.26 m/s; a walk is ~1 m/s. The measured
  bias during the walk-out is -0.045 against the empty room's +0.012 —
  indistinguishable. Needs roughly 100 Hz, and an entry event to test the
  opposite sign against.
- **The fidget indicator does not transfer between captures.** It is an
  absolute power where the motion score is a correlation; two stretches that
  were both empty disagree 3.4x (50.8 against 14.8). Calibrated in-capture and
  diagnostic only until understood.
- **383-520 s is unexplained.** The occupant reports leaving at 383 s and
  going well away, yet motion runs at occupied intensity (p50 0.73 against the
  night's 0.18, 83.7% above threshold) and stops as a *step* at 505-515 s
  rather than decaying. Ruled out: source-MAC change, MIMO change, AGC (H0
  median holds 56.5 dB), common-mode gain (12-18%). H0's own variability
  agrees (sd 0.83 over 450-500 s against 0.25 while occupied). Respiration
  says nothing either way there — its 14.8% detection is inside the empty
  control's 12.3% false-alarm rate.
- **Subcarrier 129** (DC+1) leaves a faint line after normalisation, 4.02 dB
  against a 5.24 dB median. One bin of 209; revisit if power averaging shows
  it.

## Deviations from the spec, and why

- No downsampling (§4.1). The spec asks for 20-50 Hz; the link delivers 18.12.
- Fidget band clamped 1-9.06 Hz, not 1-10 (§3.2), and read as absolute power
  rather than an in-band fraction — 1 Hz to Nyquist is 89% of the spectrum
  here, so white noise scores 0.92 as a fraction and nothing can rise above it.
- Motion Doppler axis clamped to ±9.06 Hz, not ±50 (§5), and the window
  lengthened from 0.25 s to 0.99 s (0.25 s is 4.5 samples).
- Gap policy stays `gap_limit_for`'s 95th percentile × 2.0 rather than the
  spec's 3× median: on a capture with thick jitter the median rule marks
  healthy frames as dropouts.
- §7's empty-room calibration window is 520-600 s, not 385-600 s. The room is
  still settling until ~510 s.
