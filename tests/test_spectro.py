"""Tests for backend.spectro -- signed two-scale Doppler."""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectro import (
    doppler_sign_bias,
    find_sidebands,
    stft_complex,
    stft_config,
)

FS = 18.116


def _tone(hz: float, seconds: float = 120.0, n_sc: int = 6, amp: float = 1.0,
          fs: float = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.exp(2j * np.pi * hz * t)[:, None] * np.ones((1, n_sc))


# --------------------------------------------------------------------------- #
#  Configuration against a real sample rate                                    #
# --------------------------------------------------------------------------- #


def test_the_motion_config_is_clamped_to_what_this_rate_can_deliver() -> None:
    """The spec asks for 0.25 s windows and a +/-50 Hz axis at ~100 Hz.

    This link runs 18.12 Hz. A 0.25 s window is 4.5 samples and +/-50 Hz is
    five times Nyquist; both have to be reported as unreachable rather than
    quietly approximated.
    """
    cfg = stft_config(FS, "motion")

    assert cfg["display_hz"][1] == pytest.approx(FS / 2, rel=1e-6)
    assert 0.25 <= cfg["window_seconds"] <= 1.0
    assert cfg["win"] >= 12, "a 0.25 s window would be 4.5 samples at this rate"
    assert any("Nyquist" in w for w in cfg["warnings"])


def test_the_breathing_config_reaches_the_resolution_the_spec_asks_for() -> None:
    cfg = stft_config(FS, "breathing")

    assert 0.025 <= cfg["resolution_hz"] <= 0.05
    assert cfg["display_hz"] == (-1.0, 1.0)


def test_an_unknown_purpose_is_refused() -> None:
    with pytest.raises(ValueError, match="purpose"):
        stft_config(FS, "telepathy")


# --------------------------------------------------------------------------- #
#  Signed spectrogram                                                          #
# --------------------------------------------------------------------------- #


def test_positive_and_negative_doppler_land_on_opposite_sides() -> None:
    """The whole point of keeping the ratio complex.

    A real-valued input has a conjugate-symmetric spectrum, so approaching and
    receding motion are indistinguishable. The CSI ratio cancels CFO, which is
    what makes the sign meaningful rather than an artefact of the oscillator.
    """
    cfg = stft_config(FS, "motion")
    for hz in (2.0, -2.0):
        spec, freqs, _ = stft_complex(_tone(hz), FS, cfg["win"], cfg["hop"])
        peak = freqs[np.argmax(np.nanmean(spec, axis=1))]
        assert np.sign(peak) == np.sign(hz), hz
        assert abs(abs(peak) - 2.0) < 1.0


def test_the_taper_reaches_a_sideband_a_rectangular_window_buries() -> None:
    """Why Blackman-Harris and not a plain window.

    A respiration sideband sits 30-40 dB under the static peak. A rectangular
    window's first sidelobe is -13 dB, so the leak from DC covers it outright.
    """
    cfg = stft_config(FS, "breathing")
    t = np.arange(int(200 * FS)) / FS
    sig = (1.0 + 0.02 * np.exp(2j * np.pi * 0.25 * t))[:, None]

    bh, freqs, _ = stft_complex(sig, FS, cfg["win"], cfg["hop"])
    rect, _, _ = stft_complex(sig, FS, cfg["win"], cfg["hop"], taper="rect")

    def contrast(spec: np.ndarray) -> float:
        mean = np.nanmean(spec, axis=1)
        target = np.argmin(np.abs(freqs - 0.25))
        floor = np.median(mean[np.abs(freqs - 0.25) > 0.1])
        return float(mean[target] / floor)

    assert contrast(bh) > contrast(rect)


def test_a_bridged_dropout_blanks_its_column() -> None:
    cfg = stft_config(FS, "motion")
    sig = _tone(1.0, seconds=60.0)
    fabricated = np.zeros(sig.shape[0], dtype=bool)
    fabricated[500:900] = True

    spec, _, times = stft_complex(sig, FS, cfg["win"], cfg["hop"], fabricated=fabricated)

    blank = np.isnan(spec).all(axis=0)
    assert blank.any()
    assert not blank[0]


# --------------------------------------------------------------------------- #
#  Sidebands, not shifts                                                       #
# --------------------------------------------------------------------------- #


def test_respiration_is_found_as_a_symmetric_pair_not_a_single_peak() -> None:
    """A chest oscillates in place; it does not traverse a wavelength.

    Bessel expansion of a phase modulation puts symmetric sidebands at
    +/-f_b around the static term, so looking for one shifted peak finds
    nothing and looking for the pair finds it.
    """
    cfg = stft_config(FS, "breathing")
    t = np.arange(int(300 * FS)) / FS
    beta = 0.8
    sig = np.exp(1j * beta * np.sin(2 * np.pi * 0.25 * t))[:, None] * np.ones((1, 8))

    spec, freqs, _ = stft_complex(sig, FS, cfg["win"], cfg["hop"])
    found = find_sidebands(np.nanmean(spec, axis=1), freqs, band_hz=(0.1, 0.6))

    assert found["found"]
    assert found["hz"] == pytest.approx(0.25, abs=0.02)
    assert found["symmetry"] > 0.8


def test_a_harmonic_is_not_reported_as_a_second_person() -> None:
    """At a=5 mm the modulation index is ~1 rad, so the 2nd harmonic is real.

    Without rejection, a 0.5 Hz harmonic of a 0.25 Hz chest reads as a second
    occupant breathing at 30 rpm.
    """
    cfg = stft_config(FS, "breathing")
    t = np.arange(int(300 * FS)) / FS
    sig = np.exp(1j * 1.2 * np.sin(2 * np.pi * 0.2 * t))[:, None] * np.ones((1, 8))

    spec, freqs, _ = stft_complex(sig, FS, cfg["win"], cfg["hop"])
    found = find_sidebands(np.nanmean(spec, axis=1), freqs, band_hz=(0.1, 0.6))

    assert found["hz"] == pytest.approx(0.2, abs=0.02)
    assert all(h > 1 for h in found["harmonics_rejected"])


def test_noise_yields_no_sideband_pair() -> None:
    cfg = stft_config(FS, "breathing")
    rng = np.random.default_rng(3)
    n = int(300 * FS)
    noise = rng.standard_normal((n, 8)) + 1j * rng.standard_normal((n, 8))

    spec, freqs, _ = stft_complex(noise, FS, cfg["win"], cfg["hop"])
    found = find_sidebands(np.nanmean(spec, axis=1), freqs, band_hz=(0.1, 0.6))

    assert not found["found"]


# --------------------------------------------------------------------------- #
#  Sign as a diagnostic                                                        #
# --------------------------------------------------------------------------- #


def test_the_sign_bias_follows_the_direction_of_travel() -> None:
    cfg = stft_config(FS, "motion")
    for hz in (1.5, -1.5):
        spec, freqs, _ = stft_complex(_tone(hz, seconds=60.0), FS, cfg["win"], cfg["hop"])
        bias = doppler_sign_bias(spec, freqs)
        assert np.sign(np.nanmedian(bias)) == np.sign(hz), hz


def test_a_symmetric_spectrum_has_no_sign_bias() -> None:
    """Respiration must not read as travel: its sidebands are symmetric."""
    cfg = stft_config(FS, "breathing")
    t = np.arange(int(300 * FS)) / FS
    sig = np.exp(1j * 0.8 * np.sin(2 * np.pi * 0.25 * t))[:, None] * np.ones((1, 8))

    spec, freqs, _ = stft_complex(sig, FS, cfg["win"], cfg["hop"])
    assert abs(np.nanmedian(doppler_sign_bias(spec, freqs))) < 0.1


def test_a_high_pass_slope_does_not_become_a_sideband() -> None:
    """The bug this replaced: every regime peaked at the band edge.

    The 15 s detrend upstream is a high-pass whose corner sits just below the
    respiration band, so in-band power falls monotonically and the largest bin
    is always the lowest one. Scored against the band median that reads as a
    6.00 rpm sideband -- the edge exactly -- in an empty room.
    """
    freqs = np.linspace(-1.0, 1.0, 401)
    slope = 1.0 / (np.abs(freqs) + 0.05)          # no peak anywhere, just 1/f

    found = find_sidebands(slope, freqs, band_hz=(0.1, 0.6))

    assert not found["found"], f"a bare slope was read as {found['rpm']:.2f} rpm"


def test_a_real_peak_on_top_of_a_slope_is_still_found() -> None:
    freqs = np.linspace(-1.0, 1.0, 401)
    slope = 1.0 / (np.abs(freqs) + 0.05)
    peak = 40.0 * np.exp(-(((np.abs(freqs) - 0.25) / 0.01) ** 2))

    found = find_sidebands(slope + peak, freqs, band_hz=(0.1, 0.6))

    assert found["found"]
    assert found["hz"] == pytest.approx(0.25, abs=0.02)
