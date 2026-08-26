"""Tests for backend.presence pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from backend.presence import (
    STATE_EMPTY,
    STATE_MOVING,
    STATE_PRESENT,
    STATE_UNKNOWN,
    autocorr_columns,
    bandpass,
    channel_from_ratio,
    classify,
    complex_ratio,
    fractional_motion,
    presence_windows,
)

FS = 10.0
BREATH_HZ = 0.25          # 15 rpm, mid-band
DURATION_S = 120.0


def _synthetic_ratio(
    breath_hz: float | None,
    *,
    noise: float = 0.002,
    depth: float = 0.02,
    n_sc: int = 8,
    duration_s: float = DURATION_S,
    fs: float = FS,
    seed: int = 0,
) -> np.ndarray:
    """A complex CSI ratio with an optional chest modulation.

    Each subcarrier gets its own static term and its own fixed complex
    direction for the perturbation, which is what a real chest produces: one
    physical motion seen through different multipath geometry per subcarrier.
    Collinear per subcarrier, incoherent across them until the modulation is
    switched on.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    static = 1.0 + 0.2 * rng.standard_normal(n_sc) + 0.2j * rng.standard_normal(n_sc)
    direction = np.exp(1j * rng.uniform(0, 2 * np.pi, n_sc))

    ratio = np.tile(static, (n, 1))
    if breath_hz is not None:
        ratio = ratio + depth * np.sin(2 * np.pi * breath_hz * t)[:, None] * direction
    ratio = ratio + noise * (
        rng.standard_normal((n, n_sc)) + 1j * rng.standard_normal((n, n_sc))
    )
    return ratio


# --------------------------------------------------------------------------- #
#  Channel construction                                                        #
# --------------------------------------------------------------------------- #


def test_complex_ratio_inverts_the_amplitude_convention() -> None:
    """dB here is 20*log10, so the inverse is 10^(dB/20), not 10^(dB/10).

    Getting this wrong squares the magnitude ratio between the chains, which
    still looks plausible on a plot and quietly rescales every threshold.
    """
    magnitude = np.array([[0.5, 2.0]])
    phase = np.array([[0.3, -1.2]])
    out = complex_ratio(20 * np.log10(magnitude), phase)

    assert np.allclose(np.abs(out), magnitude)
    assert np.allclose(np.angle(out), phase)


def test_fractional_motion_is_scale_free() -> None:
    """The point of dividing by |r|: gain changes must not read as motion.

    A capture whose chains happen to be gained differently would otherwise
    need its own motion threshold.
    """
    ratio = _synthetic_ratio(BREATH_HZ)
    assert np.allclose(fractional_motion(ratio), fractional_motion(ratio * 37.0))


def test_dead_subcarriers_do_not_flatten_the_motion_trace() -> None:
    """The failure this caught on a real capture, pinned.

    Every capture carries subcarriers that never hold a ratio -- DC, guard
    band, dropped pilots; 11 of 256 on MediaTek. A plain mean across
    subcarriers returns NaN for a frame if even one is missing, so the whole
    trace goes NaN and a downstream nan_to_num renders it as a flat zero.
    Measured on captures/20260821_170002.bin that read 0.0000 at every
    percentile: the most confident possible claim that nothing is moving,
    from 11 dead bins.
    """
    ratio = _synthetic_ratio(BREATH_HZ)
    reference = fractional_motion(ratio)

    ratio[:, 3] = np.nan                      # a structural null
    with_null = fractional_motion(ratio)

    assert np.isfinite(with_null).all(), "a dead subcarrier must not blank the trace"
    assert with_null.max() > 0.0, "and must not flatten it to zero"
    assert np.allclose(with_null, reference, rtol=0.5)


def test_a_dead_subcarrier_is_dropped_rather_than_averaged_in() -> None:
    """Weighting and flatness must be measured over live subcarriers only.

    A dead bin zeroed rather than removed dilutes every average it reaches.
    """
    ratio = _synthetic_ratio(BREATH_HZ)
    clean = presence_windows(ratio, FS)

    ratio[:, 2] = np.nan
    holed = presence_windows(ratio, FS)

    assert np.nanmedian(holed["rate_rpm"]) == pytest.approx(BREATH_HZ * 60.0, abs=1.0)
    assert np.nanmean(holed["score"]) == pytest.approx(
        np.nanmean(clean["score"]), abs=0.15
    )


def test_a_range_with_no_ratio_at_all_is_refused() -> None:
    ratio = np.full((600, 4), np.nan, dtype=complex)
    with pytest.raises(ValueError, match="no subcarrier"):
        presence_windows(ratio, FS)


def test_phase_channel_is_unwrapped_along_time() -> None:
    """A ratio rotating past +/-pi must not produce a saw-tooth channel.

    Wrapped, every revolution injects a 2*pi step -- a broadband impulse
    landing squarely in the respiration band.
    """
    n = 400
    angle = np.linspace(0, 6 * np.pi, n)          # nearly three revolutions
    ratio = np.exp(1j * angle)[:, None]

    phase = channel_from_ratio(ratio, "phase")
    steps = np.abs(np.diff(phase[:, 0]))
    assert steps.max() < 0.1, "unwrapped phase must not jump"


# --------------------------------------------------------------------------- #
#  Signal processing primitives                                                #
# --------------------------------------------------------------------------- #


def test_bandpass_is_zero_phase() -> None:
    """Group delay would shift the ACF peak and bias the reported rate."""
    n = 2048
    t = np.arange(n) / FS
    clean = np.sin(2 * np.pi * BREATH_HZ * t)[:, None]

    out = bandpass(clean, FS, (0.1, 0.6))

    # Compare away from the edges, where filtfilt's padding still shows.
    interior = slice(200, -200)
    lag = int(np.argmax(np.correlate(
        out[interior, 0], clean[interior, 0], mode="same"
    ))) - len(clean[interior, 0]) // 2
    assert lag == 0


def test_bandpass_rejects_out_of_band_energy() -> None:
    n = 2048
    t = np.arange(n) / FS
    in_band = np.sin(2 * np.pi * 0.3 * t)[:, None]
    out_of_band = np.sin(2 * np.pi * 3.0 * t)[:, None]

    kept = bandpass(in_band, FS, (0.1, 0.6))[200:-200, 0]
    killed = bandpass(out_of_band, FS, (0.1, 0.6))[200:-200, 0]

    assert kept.std() > 0.6
    assert killed.std() < 0.05


def test_bandpass_passes_complex_input_through_both_planes() -> None:
    """Real coefficients on a complex signal means filtering re/im separately.

    Not an approximation -- it is what applying a real filter to a complex
    signal means -- but it is worth pinning, because dropping the imaginary
    plane would silently turn the complex channel into a magnitude one.
    """
    n = 2048
    t = np.arange(n) / FS
    sig = np.exp(1j * 2 * np.pi * BREATH_HZ * t)[:, None]

    out = bandpass(sig, FS, (0.1, 0.6))

    assert np.iscomplexobj(out)
    assert out.imag[200:-200, 0].std() > 0.5


def test_autocorrelation_is_normalised_and_finds_the_period() -> None:
    n = 1024
    t = np.arange(n) / FS
    sig = np.sin(2 * np.pi * BREATH_HZ * t)[:, None]

    ac = autocorr_columns(sig)

    assert ac[0, 0] == pytest.approx(1.0)
    period_lag = int(round(FS / BREATH_HZ))
    # Search a neighbourhood, not the global max: lag 0 is always the maximum.
    near = ac[period_lag - 3 : period_lag + 4, 0]
    assert int(np.argmax(near)) == 3
    assert near[3] > 0.8


def test_complex_autocorrelation_keeps_the_signature() -> None:
    """A collinear perturbation's periodicity survives taking the real part."""
    n = 1024
    t = np.arange(n) / FS
    direction = np.exp(1j * 1.1)
    sig = (np.sin(2 * np.pi * BREATH_HZ * t) * direction)[:, None]

    ac = autocorr_columns(sig)
    period_lag = int(round(FS / BREATH_HZ))

    assert ac[0, 0] == pytest.approx(1.0)
    assert ac[period_lag, 0] > 0.8


# --------------------------------------------------------------------------- #
#  The detector                                                                #
# --------------------------------------------------------------------------- #


def test_breathing_is_detected_and_its_rate_recovered() -> None:
    result = presence_windows(_synthetic_ratio(BREATH_HZ), FS)

    assert result["time_s"].size > 0
    scored = result["score"][np.isfinite(result["score"])]
    assert scored.mean() > 0.25, f"breathing must clear the threshold, got {scored.mean():.3f}"

    rate = np.nanmedian(result["rate_rpm"])
    assert rate == pytest.approx(BREATH_HZ * 60.0, abs=1.0)
    assert result["state"].count(STATE_PRESENT) > 0.8 * len(result["state"])


def test_an_empty_room_does_not_score_as_breathing() -> None:
    """Noise alone must not produce a chest.

    This is the test the plain in-band-power gate fails: bandpassing noise
    makes it narrowband, so its autocorrelation oscillates and periodicity
    alone looks respectable. Tonality is what separates them.
    """
    result = presence_windows(_synthetic_ratio(None, noise=0.01), FS)

    scored = result["score"][np.isfinite(result["score"])]
    assert scored.max() < 0.25, f"empty room scored {scored.max():.3f}"
    assert set(result["state"]) == {STATE_EMPTY}


def test_all_three_channels_see_the_same_chest() -> None:
    ratio = _synthetic_ratio(BREATH_HZ)
    for channel in ("complex", "phase", "magnitude"):
        result = presence_windows(ratio, FS, channel=channel)
        rate = np.nanmedian(result["rate_rpm"])
        assert rate == pytest.approx(BREATH_HZ * 60.0, abs=1.5), f"{channel} channel"


def test_walking_reads_as_moving_not_as_breathing() -> None:
    """Gross motion is present, but it is not *static* presence.

    Ungated, a walking person's periodic gait reads as an enormous chest.
    """
    rng = np.random.default_rng(7)
    n = int(DURATION_S * FS)
    ratio = _synthetic_ratio(BREATH_HZ) + 0.3 * (
        rng.standard_normal((n, 8)) + 1j * rng.standard_normal((n, 8))
    )

    result = presence_windows(ratio, FS)

    assert np.nanmedian(result["motion_level"]) > 0.25
    assert set(result["state"]) == {STATE_MOVING}


def test_a_dropout_is_unknown_and_never_empty() -> None:
    """The one lie that matters on a presence panel.

    Interpolating across a hole produces a flat stretch, and flat means no
    motion and no periodicity -- which scores exactly like an empty room.
    """
    ratio = _synthetic_ratio(BREATH_HZ)
    n = ratio.shape[0]
    fabricated = np.zeros(n, dtype=bool)
    hole = slice(n // 2 - 200, n // 2 + 200)      # 40 s, wider than one window
    fabricated[hole] = True
    ratio[hole] = ratio[hole.start - 1]            # what interpolation leaves

    result = presence_windows(ratio, FS, fabricated=fabricated)

    assert STATE_UNKNOWN in result["state"]
    unknown = np.asarray(result["unknown"])
    assert np.isnan(result["score"][unknown]).all(), "a blanked window reports no score"
    assert np.isfinite(result["score"][~unknown]).any(), "windows outside the hole survive"

    # The blanked stretch must not have been called empty.
    states = np.asarray(result["state"])
    assert STATE_EMPTY not in set(states[unknown])
    assert any("interpolated" in w for w in result["warnings"])


def test_smoothing_cannot_leak_into_a_blanked_window() -> None:
    """Blanking runs after the moving average, not before.

    The other order lets a neighbour's score bleed across the hole and gives
    a fabricated window a real-looking number.
    """
    ratio = _synthetic_ratio(BREATH_HZ)
    n = ratio.shape[0]
    fabricated = np.zeros(n, dtype=bool)
    fabricated[n // 2 : n // 2 + 200] = True

    result = presence_windows(ratio, FS, fabricated=fabricated, smooth_windows=9)
    unknown = np.asarray(result["unknown"])

    assert unknown.any()
    assert np.isnan(result["score"][unknown]).all()


# --------------------------------------------------------------------------- #
#  Geometry guards                                                             #
# --------------------------------------------------------------------------- #


def test_a_rate_above_nyquist_is_refused_rather_than_aliased() -> None:
    ratio = _synthetic_ratio(BREATH_HZ, fs=1.0, duration_s=600.0)
    with pytest.raises(ValueError, match="Nyquist"):
        presence_windows(ratio, 1.0, rate_band_rpm=(9.0, 40.0))


def test_a_window_too_short_for_the_band_is_refused() -> None:
    ratio = _synthetic_ratio(BREATH_HZ)
    with pytest.raises(ValueError, match="cannot resolve"):
        presence_windows(ratio, FS, window_seconds=3.0)


def test_the_effective_rate_floor_is_reported_when_it_is_not_what_was_asked() -> None:
    """A 12 s window cannot see 9 rpm, and says so rather than pretending."""
    result = presence_windows(_synthetic_ratio(BREATH_HZ), FS, window_seconds=12.0)

    assert result["rpm_floor_eff"] > 9.0
    assert any("rpm" in w for w in result["warnings"])


def test_a_window_longer_than_the_range_is_clamped_not_refused() -> None:
    """Zooming past the window length is ordinary use of a linked time axis."""
    ratio = _synthetic_ratio(BREATH_HZ, duration_s=20.0)
    result = presence_windows(ratio, FS, window_seconds=60.0)

    assert result["window_seconds"] == pytest.approx(20.0, abs=0.2)
    assert result["time_s"].size >= 1


# --------------------------------------------------------------------------- #
#  Verdict ordering                                                            #
# --------------------------------------------------------------------------- #


def test_verdict_precedence_is_unknown_then_moving_then_present() -> None:
    """Order is the whole content of classify(), so it is pinned here."""
    score = np.array([0.9, 0.9, 0.9, 0.0])
    motion = np.array([0.0, 0.9, 0.0, 0.0])
    unknown = np.array([True, False, False, False])

    assert classify(score, motion, unknown) == [
        STATE_UNKNOWN,     # blanked, despite a high score
        STATE_MOVING,      # moving beats a high score
        STATE_PRESENT,
        STATE_EMPTY,       # what is left when nothing else claimed it
    ]


def test_a_nan_motion_level_is_unknown_not_empty() -> None:
    assert classify(
        np.array([np.nan]), np.array([np.nan]), np.array([False])
    ) == [STATE_UNKNOWN]
