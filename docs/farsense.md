# FarSense and this pipeline

Zeng, Wu, Xiong, Yi, Gao, Zhang — *FarSense: Pushing the Range Limit of
WiFi-based Respiration Sensing with CSI Ratio of Two Antennas*, IMWUT 3(3):121,
September 2019. [arXiv 1907.03994](https://arxiv.org/abs/1907.03994),
[ACM DL](https://dl.acm.org/doi/10.1145/3351279).

`backend/farsense.py` is a step-for-step copy of the paper's pipeline; the
**FarSense** tab replays it over a capture. This note records what the paper
does, where we already do the same, where we differ, and what the copy does on
the September data.

> **Tab defaults since 2026-09-22** differ from the paper: window 10 s
> (paper 12), band 10–30 rpm (paper 10–37), keep ≥ 0.6 × best (paper 0.7),
> 200 θ steps (paper 100), Savitzky-Golay 1.0 s (was 0.5), high-pass 0,
> min peak 0.2 (the paper has no peak gate). The paper's values remain in
> the text below; every default is a knob on the tab and a query parameter.

## 1. What FarSense does

**The problem it solves.** Respiration sensing from CSI amplitude works to
2–4 m and then dies in noise; CSI phase is unusable because every packet
carries a random offset (CFO, SFO, packet-detection delay). Existing
"blind spots" — positions where the chest moves the signal but the amplitude
does not change — come from the same place.

**The two ideas.**

1. **The ratio of two antennas' CSI** (Sec. 4). Both receive chains share
   one oscillator, so the random per-packet phase `e^{-jθ}` is identical on
   both and divides out. Amplitude impulse noise, which is also common to the
   chains, divides out too. What is left is a complex number whose phase is
   finally meaningful. Their Sec. 3 measurement: a metal plate moving 29 cm
   at 5.5 m LoS shows 5 clear Fresnel peaks in the amplitude *ratio* and none
   readable in either raw amplitude.
2. **The CSI-ratio model** (Sec. 4.2). With `H = H_s + A·e^{-j2πd/λ}` on each
   antenna and `Δd` between the antennas constant for a small motion, the
   ratio is a Möbius transform `(AZ+B)/(CZ+D)` of the unit-circle rotation
   `Z`. Möbius maps circles to circles, so:
   - **P1** a reflection-path change of several wavelengths traces a circle;
   - **P2** the rotation is clockwise when the static path dominates the
     dynamic one (the usual case) and reverses when it does not;
   - **P3** a change of less than one wavelength traces an arc whose angle
     matches the path change — a 5–12 mm chest at λ = 5.7 cm is an arc of
     roughly 60–150°.

**Extracting the breath** (Sec. 5.2). A point `a + bi` projected on an axis
at angle θ is `a cos θ + b sin θ` — a linear combination of I and Q, of which
"pick the better of amplitude and phase" (FullBreathe) is the special case
θ ∈ {0, π/2}. They sweep θ over 100 values and keep, per subcarrier, the
projection with the largest **breathing-to-noise ratio (BNR)**: energy of the
strongest FFT bin inside 10–37 bpm over the energy of the spectrum, on a 12 s
window zero-padded to 8192 points. Periodicity, not variance: at range the
arc is gone and the largest-variance axis is the noise axis.

**Rate** (Sec. 6.4). Autocorrelate each subcarrier's pattern (the biased
estimator, Eq. 10); sum the autocorrelations weighted by BNR over the
subcarriers whose BNR is above 0.7 × the best (Eq. 11); the lag of the first
peak is the period. A motion detector (Li et al. 2018's speed spectrum, fed
the ratio) blanks windows with large motion; a Savitzky-Golay filter smooths
each subcarrier first.

**Setup and result.** Intel 5300, 1 Tx antenna, 2 Rx antennas, 5.24 GHz /
20 MHz / 30 subcarriers, **100 packets per second**, MATLAB on a laptop
(0.54 s per 30-subcarrier sweep). Ground truth from a respiration belt.
Detection = rate within 0.5 bpm; sensing range = farthest distance with
≥ 95 % detection. HRD (amplitude) < 2.9 m, FullBreathe 3.7 m, FarSense
> 8 m; mean error 0.28 bpm at 6 m, 0.64 at 9 m; through one wall 0.34 bpm;
ceiling-mounted over a bed under a quilt < 0.3 bpm in every posture.

**No public code.** The authors released none; the only public reference is
`citysu/csiread`'s `examples/csiratioA.py`, which plots `|csi[..., 0, 0] /
csi[..., 1, 0]|` live and nothing more. The copy here is from the paper.

## 2. Where we stand against it

| | FarSense | This repo |
|---|---|---|
| Base signal | ratio of two **Rx** antennas, same NIC | ratio of the AP's two **Tx** chains at one Rx chain (`backend/mtk.py`): the same per-packet phase divides out; the Rx-pair on the MT7921 is 5.7× noisier per frame and is not used |
| Phase usable | yes, that is the point | yes, same reason (`csi_ratio_phase`, Doppler on the complex ratio) |
| Subcarriers | 30 of a 20 MHz channel | 245 of an 80 MHz channel |
| Packet rate | 100 Hz | 19.6 Hz stimulus (42 Hz on the newest captures) |
| Range | up to 8–9 m | 1.2–3 m in every labelled capture |
| Motion gate | speed spectrum of Li et al. 2018 | fractional change of the ratio, median over the window (`presence.fractional_motion`) — the same gate on both tabs |
| Smoothing | Savitzky-Golay | bandpass 0.1–0.6 Hz (`presence.bandpass`) on the presence tab; S-G on the FarSense tab |
| Axis choice | sweep θ, keep max **BNR** | `breathing.project_to_1d`: closed-form principal axis = max **variance**, exactly the rule Sec. 5.2.2 argues against; `presence_windows` takes the real part of the complex autocorrelation instead of choosing at all |
| Subcarrier weight | BNR, keep those > 0.7 × best | in-band power fraction (`breathing.band_weights`) or in-band-over-shoulder ratio (`presence.in_band_weight`), keep all |
| Rate | first peak of the BNR-weighted autocorrelation sum, 12 s window | `breathing.estimate_rate`: FFT peak with parabolic refinement **and** autocorrelation, 45 s window, must agree; `presence_windows`: largest autocorrelation peak, 30 s window |
| Confidence | none — a rate is shown whenever stationary | `breathing`: PAPR × agreement × group spread; `presence`: periodicity × tonality × motion gate, and the breathing score does not vote on occupancy |
| Ground truth | respiration belt | camera (presence only); no rate reference exists in the corpus |

Two of our choices are things the paper explicitly measured to be worse at
range: **max-variance projection** and, in `presence_windows`, no projection
at all. Two are things the paper does not have and which we found necessary
here: a **confidence** the rate is gated by (the paper's number is always
shown), and the **channel-offset** presence verdict, because a still occupant
does not modulate the channel so much as displace it.

## 3. What the copy does on our data

Defaults are the paper's (12 s window, 10–37 bpm, 100 angles, 8192-point
FFT, keep > 0.7 × best, S-G 0.5 s cubic, gate at |Δr|/|r| > 0.25). MIMO
2×1, all labelled September captures with a camera sidecar (43 captures,
~11 700 stationary windows).

**Synthetic check** (a 3 % chest along a random axis per subcarrier, 3 %
complex noise, 19.6 Hz): the rate is recovered within 0.5 rpm in > 95 % of
windows at 12, 15, 25, 30 and 36 rpm, including a chest purely along the
imaginary axis where |r| is flat; the refinement removes the +0.2 rpm lean
the biased autocorrelation carries. An empty synthetic room still gets a
rate in 60 % of windows, by design.

**Real captures, window by window, camera-occupied vs camera-empty:**

| score | AUC | median occupied | median empty |
|---|---|---|---|
| normalised first-peak height | 0.54 | −0.011 | −0.028 |
| best BNR | 0.69 | 0.0137 | 0.0111 |
| subcarriers kept (> 0.7 × best) | 0.68 | 58 | 35 |

A **threshold on the normalised peak** buys at best 59 % balanced accuracy
(recall 0.34 / specificity 0.84 at 0.10). In other words, on this link the
paper's rate reads a period in the empty room about as readily as with an
occupant, and the median peak behind an occupied rate is *negative* — the
autocorrelation is reading a drift's slope, not a breath.

Per capture the picture is uneven: a handful of still captures carry a
clear breath (peak 0.13–0.46, rate 15–22 rpm, 40–64 % of windows within
±1 rpm of the capture median), most do not (peak ≈ 0, rate scattered across
the band), and no empty capture stays quiet. The captures that work are
not distinguished by distance or activity label.

## 4. What is worth taking

1. **BNR as the axis criterion**, in place of `project_to_1d`'s max
   variance. The paper's argument is right and our own finding agrees with
   it: the periodicity weight was already what made `band_weights` work. The
   BNR sweep is the same idea applied to the axis choice. Cost is nothing;
   `extract_patterns` runs the whole 245 × 100 sweep in ~10 ms per window.
2. **The I/Q arc as a diagnostic.** The detail panel's complex-plane plot
   is the first view in the tool that shows *whether a chest is there at
   all* in the paper's terms — an arc versus a blob. Worth keeping whatever
   happens to the rate.
3. **Keep-fraction selection.** Dropping subcarriers below 0.7 × the best
   BNR is a cleaner version of weighting by evidence; on the corpus the count
   kept separates occupied from empty about as well as BNR itself.
4. **Not the 12 s window, not the unconfidenced rate.** At 20 Hz a 12 s
   window is 240 samples; the paper's is 1200. Our 30–45 s windows exist
   because the shorter one cannot hold two periods at 10 rpm with anything to
   spare, and the corpus numbers above are the result. And a rate shown
   without a confidence is, on this data, a number for the empty room too.

The next experiment is the one the paper could not need: the same sweep
after a high-pass (`highpass_hz`, off by default) and at 30 s (`window_seconds`).
Both are exposed on the tab and the API; §5 has the numbers when they exist.

## 5. Variants

Same corpus, same scoring, four settings of the two knobs the paper did not
need. "Rate stable" is the fraction of a capture's rated windows within
±1 rpm of that capture's median rate, averaged over captures with ≥ 30
stationary windows — a proxy for "is the number the same from window to
window", since no belt exists to compare against.

| window | high-pass | AUC peak | AUC BNR | AUC kept | best balanced acc. (peak >) | captures with median peak > 0.1 | rate stable, occupied | rate stable, empty |
|---|---|---|---|---|---|---|---|---|
| 12 s (paper) | off (paper) | 0.54 | 0.69 | 0.68 | 0.59 (0.18) | 7 / 29 | 0.28 | 0.19 |
| 12 s | 0.1 Hz | 0.55 | 0.73 | 0.68 | 0.61 (0.18) | 7 / 29 | 0.30 | 0.18 |
| 30 s | off | 0.56 | 0.73 | 0.61 | 0.62 (0.14) | 9 / 28 | 0.30 | 0.19 |
| 30 s | 0.1 Hz | 0.54 | **0.78** | 0.66 | 0.63 (0.10) | 10 / 28 | 0.43 | 0.35 |

Neither knob rescues the **rate**: the peak's AUC stays at 0.54–0.56 in
every setting, and the longer window with the high-pass makes the rate more
self-consistent in the *empty* room too (0.19 → 0.35), which is the
signature of a filter shaping noise into a period rather than of a breath
being found. What the knobs do improve is **BNR as a presence score**, from
0.69 to 0.78 — still short of the amplitude-offset detector's in-capture
result, and measured here without any calibration, which is the one thing
in its favour.

So, concretely: the paper's projection-with-maximal-periodicity is worth
carrying into `backend.breathing` as the axis rule, and BNR / kept-count
are worth reporting as calibration-free presence evidence; the paper's
rate estimator, at this packet rate and window, is not a respiration
monitor on this link.
