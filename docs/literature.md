# Literature: CSI presence detection, read against this pipeline

A survey run on 2026-10-03 for one question: what does the published work on
Wi-Fi CSI presence detection — motion, breathing, heartbeat, static-occupant
features, calibration-free thresholds — say about the two things this
pipeline cannot yet do, and what should be tried first. The pipeline it was
read against is the one on the Hybrid 2 tab: the ratio frame step
(`docs/frame_step.md`) with the fixed range rule (`P90 > 0.035` at a 2 s lag,
`docs/hybrid2.md`) OR FarSense breathing (10 s window, 10–30 rpm, peak ≥ 0.25
for 5 windows), one 2×1 Intel link at ~42 Hz, camera truth.

How it was produced: five search angles, 22 sources fetched, 110 claims
extracted, the 25 most relevant verified by three independent readers against
the primary text (a claim survives only if at most one reader refutes it),
16 confirmed, 9 refuted, merged into 13 findings. **Confidence** below is
about whether the finding describes the paper correctly; the confidence that
it transfers to our link is lower in every case — no source uses our NIC,
rate or antenna count (closest: FRID and Sensors 2026 on a 2-antenna Intel
5300, M-WiFi on 2×2 Atheros). Inferences marked *ours* are untested.

The two problems, as stated to the survey:

- **Baseline.** The empty-room floor of the step differs ~10× between link
  states (0.010 on a clean day, 0.08–0.12 on the noisy 09-16/17 link) and
  drifts within a day; a fixed threshold fitted on one state has 0 %
  specificity on another; a floor taken from the capture's own low
  percentile is the occupant when the room is occupied end to end.
- **Seated occupant.** A sitting person's per-frame step is spectrally the
  same as the noisy-link empty room (broadband, ρ₁ ≈ 0.1); seated-fidgeting
  recall is 23–44 %.

## Three answers to the baseline problem, none free

1. **A statistic whose empty-room level is noise-invariant by construction**
   — WiDetect's normalised lag-1 autocorrelation. The only verified feature
   of that kind.
2. **Relative or per-capture thresholds** — FRID's adjacent-window ratio,
   the VE-Otsu split of Sensors 2026. They cancel slow drift and **fail by
   construction when a capture holds one state**. PADS said so about
   FIMD-style clustering in 2014; it is our own-floor failure relocated.
3. **Operational recalibration** — WiSH's scheduled refresh from a
   presumed-empty hour, M-WiFi's re-fit from a few verified labels per site.

And one negative result: **no verified paper separates a seated-still person
from a noisy-link empty room at per-second resolution on one 1×2 link.**
DeMan uses cooperative breathing subjects, M-WiFi five-minute windows on
2×2, PADS / FRID / R-TTWD moving people only. The seated gap needs our own
measurement, not a citation.

## The methods

### WiDetect — Zhang, Wu, Wang, Lai, Han, Liu. IMWUT 3(3) 122, 2019. DOI 10.1145/3351280. *High.*

Detrend each per-subcarrier CSI power series, take the sample
autocorrelation at lag 1/Fs, average over F subcarriers. Under no motion the
average is N(−1/T, 1/(F·T)): a function of window length T and subcarrier
count F only — "not affected by the variance of measurement noise σ²(f)".
False-alarm rate for a preset η is closed-form, Q(√(FT)·(η + 1/T)) (their
Eq. 16). One η = 0.1, T = 60 at 30 Hz, in every environment including a
month-long deployment; 99.68 % of logged motion with no false alarm against
PIRs in a house. Hardware: 3×3, 114 subcarriers per stream, 40 MHz,
dedicated sounding.

- **Applicability.** Aims at our floor directly, *if* the bad-day noise is
  larger white noise rather than coloured. Our ρ₁ ≈ 0.1 was measured on the
  *step* series, which differencing whitens; WiDetect's ACF is on the
  detrended *level* series, where a sitter's slow micro-motion raises it
  whenever its variance exceeds per-sample noise — so it may also reach the
  seated case (*ours*).
- **Limitations.** Invariance needs temporally white noise about a constant
  mean: an AGC step or interference burst inside the window inflates the
  statistic (the paper mentions RF spikes, never AGC) — apply it to the
  ratio and gate steps. 1/(F·T) assumes independent subcarriers and their
  own Fig. 7 diverges at small η; our 2×1 link has a far smaller effective
  F, so the null is wider and T or η must grow. η = 0.1 is an empirical
  margin, not a Q-function output. Authors at Origin Wireless.
- **Change to try.** Lag-1 ACF of the detrended ratio amplitude (and
  phase) per subcarrier at 42 Hz, T ≈ 42 or 84; F_eff from the variance of
  the statistic in camera-empty windows on a clean *and* a noisy day; η
  from Eq. 16 for a chosen false-alarm rate; score against `P90 > 0.035`.
- **Transfer risk.** Medium; one script on existing captures decides it.

### FRID — Gong, Yang, Man, Dong, Yu, Lv. Sensors 15(12) 29896, 2015. DOI 10.3390/s151229896. *High.*

Per subcarrier, the coefficient of variation of linearly sanitised phase over
2 s windows; the statistic is the **ratio of adjacent windows** (SVR), which
"should be equal to 1" in a static room, with a decision interval
1 ± Z·σ from "the experiential standard variation in a static environment"
(80–90 % confidence recommended). A long-window ratio (LVR) against a
reference that is refreshed "if the intruder does not appear". Intel 5300,
2 antennas, 20 Hz; walking subjects only.

- **Applicability.** A 10× floor shift between days cancels in the ratio;
  our 4.3 dB / 2 h daytime drift is nothing between adjacent 2 s windows.
  LVR is the published form of our "reference must be refreshed" rule.
- **Limitations.** In the paper's words, continuous motion makes adjacent
  windows similar and "the SVR may fail" — the occupied-end-to-end case —
  and LVR then seeds from occupied data. σ still comes from a static
  period. A link-state step falls inside one adjacent pair and spikes the
  ratio: a false alarm, not a cancellation. Single-antenna sanitised phase,
  which later work (Zhuo et al., INFOCOM 2017; FarSense) shows is
  unreliable on commodity NICs; the ratio's phase is the transferable form.
  The paper also finds phase more sensitive than amplitude to *slow*
  walking (Fig. 13) — slow motion is the seated-fidgeting regime, but
  seated subjects were never tested.
- **Change to try.** An SVR-style adjacent-window CoV ratio of our step
  level as a **transition detector** feeding a presence state machine;
  refresh the empty reference only in SVR-quiet periods that are also empty
  by an independent cue. Ratio-phase variants of the step, the ACF and the
  breathing peak, scored on seated recall.
- **Transfer risk.** Medium-high for cold start unless an abstain state is
  added.

### PADS — Qian, Wu, Yang, Liu, Zhou. ICPADS 2014. *Medium.*

Maximum eigenvalue of the covariance of power-normalised amplitude and
sanitised phase over a window; one SVM cut line trained once on several
rooms; TN > 98 %, TP > 97 %. Rejects FIMD-style per-batch clustering because
it "implicitly assumes that at least two states are involved in each group
of measurements (otherwise … miss or false detection)". Intel 5300, 1×3,
Hampel filter; a stationary person counts as absent.

- **Applicability.** Confirms that multiplicative gain must be normalised
  out (our ratio does) and that single-state batches defeat any self-floor.
- **Limitations.** The normalisation is never written down; it removes gain
  but not additive receiver noise relative to signal, so an SNR-dependent
  floor — ours, after the ratio — is not addressed. No leave-one-out. A
  median over three RX antennas, impossible with two chains. The
  clustering failure is argued, not shown.
- **Change to try.** If any per-capture threshold is kept, an explicit
  **bimodality test**, with abstain or fall-back to a fixed normalised
  statistic when it fails.
- **Transfer risk.** Feature transfer across NICs was never shown.

### Density + VE-Otsu on the CSI ratio — Wang, Zhang, Shu. Sensors 26(11) 3303, 2026. DOI 10.3390/s26113303. *Medium (one month old, unreplicated).*

KNN density of CSI-ratio samples on the complex plane under a spatiotemporal
constraint, fused across subcarriers by asymmetric truncation into a bimodal
histogram, split by Valley-Emphasised Otsu, morphological post-processing.
Training-free. Event-level FPR / FNR 0.27 % / 0.27 % on a 2-antenna Intel
link at 100 pkt/s, under 0.6 % at 5 pkt/s; matches CNN segmenters (DeepSeg,
LiteWiSys) without training. Its stated motivation is our failure: "as
environmental noise varies over time or across scenarios, this empirically
set threshold fails to generalize, which requires recurrent data collection
and threshold recalibration."

- **Applicability.** Same feature family and RX configuration as ours;
  42 Hz inside the tested range; within-capture adaptivity handles
  within-day drift without a stored floor.
- **Limitations.** Otsu splits a unimodal histogram anyway, so an
  end-to-end occupied or empty capture reproduces the own-floor failure;
  the paper never discusses it. Audio-prompt truth, hyperparameters tuned
  on the two test rooms, data not public, "static" means no gross motion.
- **Change to try.** The truncated-density histogram and VE-Otsu on our
  ratio with a **valley-depth guard**; when unimodal, abstain and fall back
  to the ACF threshold.
- **Transfer risk.** Medium.

### The CSI ratio itself — FarSense (Zeng et al., IMWUT 2019, DOI 10.1145/3351279); Wu et al., CCF TPCI 4:88, 2022; Sensors 2026 above. *High.*

The two-antenna ratio cancels common-mode hardware noise — shared AGC gain
and oscillator (CFO/SFO/STO) — so "both its amplitude and phase" are usable.
Our own numbers agree: `rssi_1` steps through four 3 dB gain levels; p99
per-frame common-mode amplitude 2.34 dB raw vs 1.09 dB on the ratio;
occupied-vs-empty 0.1–0.6 Hz contrast 6.1 dB raw vs 15.2 dB on ratio
amplitude.

- **Limitations.** Common-mode only. The ratio **does not cancel
  independent per-chain noise**, which is why our floor still varies 10×
  after it; it amplifies noise when the denominator antenna's static path
  is weak. Holds only while both chains share clock and gain (true for the
  Intel 2×1; re-check any card that normalises gain per chain).
- **Change to try.** Keep the ratio; feed band-limited ratio amplitude and
  unwrapped ratio phase into every feature instead of raw amplitude — the
  published safe way to use an amplitude channel despite AGC. Do not expect
  it to fix the floor; pair it with a normalised statistic or a refreshed
  reference.
- **Transfer risk.** Low.

### R-TTWD — Zhu, Xiao, Sun, Wang, Yang. IEEE JSAC 35(5), 2017. DOI 10.1109/JSAC.2017.2679578. *Medium (abstract + citing papers; full text closed).*

Moving humans through walls from "the first-order difference of eigenvector
of CSI across different subcarriers" after PCA filtering, majority vote
across antenna pairs, one-class SVM — explicitly instead of time-domain
variance. Its ">99 %" figure did not survive verification as recall and
specificity.

- **Applicability.** A body perturbs subcarriers coherently along
  frequency; receiver noise is independent across them. An
  eigenvector-difference or mean inter-subcarrier correlation is the
  principled candidate for the one separation our spectral check could not
  make — seated-still from noisy-empty (*ours*).
- **Limitations.** Moving targets only; amplitude; majority vote needs more
  antenna pairs than our one.
- **Change to try.** Per-second mean inter-subcarrier correlation of the
  detrended ratio (or drift of the leading eigenvector), tested on
  seated-still against noisy-empty windows.
- **Transfer risk.** Medium-high: an untested regime.

### DeMan — Wu, Yang, Zhou, Liu, Liu, Cao. IEEE JSAC 33(11) 2329, 2015. DOI 10.1109/JSAC.2015.2430294. *High.*

A standing or sitting occupant from breathing: band-pass amplitude to
0.15–0.70 Hz, fit one sinusoid per subcarrier (Nelder–Mead), reject outlier
frequency estimates across subcarriers *and* antennas with a
least-median-of-squares test (|rₖ/σ*| ≤ 2.5), average the survivors, declare
if the fused frequency is in band. TP 94.6 % / TN 94.5 % over five link
conditions; above 95 % for both at a **30 s** window (1500 packets at
50 Hz), "slightly degrades with more packets … also drops with too fewer".
Runs in parallel with a moving-human detector (OR). ~8 h in one building,
subjects breathing naturally without other motion.

- **Applicability.** Three differences from our FarSense configuration:
  a 30 s optimum against our 10 s (our 5-window rule spans 50 s but tests
  each window alone); a cross-subcarrier / cross-antenna **consistency**
  test instead of one median peak ≥ 0.25; both RX pooled before the
  outlier test. Our spurious 7.5–10.9 rpm empty-room peaks
  (`breathing-artifact-below-11-rpm`) sit inside DeMan's 9–42 rpm band, so
  the band would not remove them; the consistency test might (*ours*).
- **Limitations.** Amplitude only — Wang et al. (UbiComp 2016) and
  FarSense show amplitude respiration sensing has Fresnel-zone blind spots
  that ratio phase removes, so DeMan's fusion goes *on top of* the ratio,
  not instead of it. Cooperative subjects; no held-out day. ~95 % should
  not be expected at arbitrary seated positions.
- **Change to try.** Per-subcarrier breathing-frequency estimates on ratio
  amplitude and ratio phase over 20–30 s windows, LMS outlier rejection,
  declare only on agreement; seated recall and empty false-alarm rate
  against the 0.25-peak rule.
- **Transfer risk.** Medium.

### M-WiFi — Soltanaghaei, Sharma, Wang, Chittilappilly, Luong, Giler, Hall, Elias, Rowe. ACM BuildSys 2020, 151–160. DOI 10.1145/3408308.3427983. *High.*

Ninety-four features on a 2×2×56 Atheros CSI matrix at 10 Hz over
five-minute windows: eigenvalues of the covariance across antenna ×
subcarrier (multipath profile) and across time (a Doppler proxy), entropy of
amplitude and of inter-antenna relative phase across subcarriers, and a
per-subcarrier channel-variation factor √(var/RMS²) with mean / median /
max / min / std aggregates. Feature groups aimed explicitly at "a stationary
(e.g. sitting or standing) person" and "a completely still (e.g. sleeping)
person". A two-layer MLP; 98.6 / 97.7 / 96.1 % at 1 / 5 / 10-minute windows.
Seven houses, 25 setups, ~100 days, entrance cameras annotated weekly.

The cross-domain half is the more important one. **Leave-one-house-out, zero
adaptation: 84 % true positive, 28 % true negative** — "the lack of
multipath profile knowledge for the test houses". The repair is
label-efficient self-calibration: an unsupervised change-point detector on
the features (per feature, majority vote, 1 h proximity filter) proposes
transitions — 93 % found at 50 % false alarm, about five confirmations a
day for at most a week — and "with an average of 3 days worth of data …
around 90 % true detection, and less than 15 % false detection", ~98 % at
steady state (within-house CV). Mislabels slow convergence, do not stop it.

- **Applicability.** The only verified system aimed at the seated or
  sleeping occupant with a feature set rather than breathing alone, and
  every feature is computable from our 1×2 raw CSI and ratio (temporal
  covariance eigenvalues ≈ Doppler energy, subcarrier entropy ≈ coherence,
  relative phase = our ratio phase). The 84/28 collapse is the published
  analogue of our 0 % specificity across link states; the repair is the
  strongest verified evidence that a few days of in-situ labels fix the
  negative class — and **we already have what they asked users for, a
  camera**, so change-point detection plus camera-verified transitions
  replaces the human in the loop.
- **Limitations.** No isolated stationary-vs-empty number; MLP ranking
  asserted without a table; five-minute decisions against our per-second
  labels; home-level truth; the shift is cross-building multipath, not
  same-room link state — analogous, not identical, and 28 % TN is not a
  prediction for us. An internal 40 MHz / 56-subcarrier inconsistency;
  three corporate co-authors.
- **Change to try.** (a) Covariance-eigenvalue, subcarrier-entropy and
  std/RMS features over 30–60 s on the ratio, a small per-room classifier.
  (b) **Leave-one-day-out and leave-one-link-state-out** splits in every
  evaluation. (c) Per-link-state thresholds re-fitted from a handful of
  camera-verified transitions, rather than made invariant.
- **Transfer risk.** Low for the protocol, medium for the features and the
  figures; absolute multipath features will not cross link states.

### WiSH — Hang, Zheng, Qian, Wu, Yang, Zhou, Liu, Chen. Tsinghua Sci. Tech. 24(5) 615, 2019. DOI 10.26599/TST.2018.9010091. *High.*

Not calibration-free, and says so: thresholds "need to be determined
preliminarily through training when the system is deployed … might need to
be recalibrated over a few days", from "data collected after midnight (e.g.,
3:00 a.m.–4:00 a.m.) … because moving entities are hardly present", with RX
antenna re-selection by detection accuracy at the same time. A
minimum-duration event filter (1.1 s) against flicker.

- **Applicability.** A small office has predictable empty hours and we
  have a camera to confirm them; a periodic refresh of the empty floor from
  camera-confirmed-empty windows avoids the floor-is-the-occupant failure
  by construction. Antenna selection maps to choosing h₁/h₂ against h₂/h₁.
- **Limitations.** A clock-driven schedule misses within-day drift unless
  re-triggered; presumes night emptiness; cross-day transfer of the
  thresholds is never quantified.
- **Change to try.** Refresh the empty reference from camera-confirmed-
  empty windows, **re-triggered whenever the link-state indicator (idle
  ratio noise, RSSI / AGC level) changes**, instead of once per capture
  (`reference-expires-on-contact-not-time`).
- **Transfer risk.** Low; a workaround, not a feature fix.

## Our numbers against the literature's protocols

The 94–99 % figures above come from within-environment protocols with
cooperative or scripted subjects and coarse truth: DeMan ~8 h in one
building, PADS without leave-one-out, Sensors 2026 tuned on its test rooms
with audio-prompt truth, M-WiFi five-minute windows against weekly-annotated
entrance cameras, WiSH with thresholds and a 1.1 s filter chosen from
labelled data. The only zero-adaptation cross-environment number in the
verified corpus is M-WiFi's 84 / 28. **No verified system reports per-second
accuracy against in-room camera truth, and none fuses motion and breathing
with an HMM** — DeMan ORs two detectors, M-WiFi runs one classifier per
window, WiSH and Sensors 2026 add minimum-duration or morphological filters,
FRID pairs a short- and a long-term test. Our 80–83 % balanced per-second
accuracy at 98 % empty specificity is not comparable to those numbers and
not shown worse by them; the "5 consecutive windows" and "P90 over 60 s"
rules are the published post-filter type. To become comparable: report
per-window accuracy at 10 / 30 / 60 s beside per-second, under
leave-one-day-out splits.

## Ranked menu

Ordered by strength of verified evidence and ease of testing on captures we
already have. A menu, not a decision.

1. **WiDetect lag-1 ACF** on the detrended ratio level series with a
   Q-function threshold. The only noise-invariant statistic found. Risk:
   coloured or bursty noise on bad days, small F_eff.
2. **DeMan-style consistency test and 20–30 s windows** in the breathing
   detector. Aims at both FarSense failures — weak single peaks and the
   7.5–10.9 rpm artefact. Risk: amplitude-only source, cooperative subjects.
3. **M-WiFi feature set over 30–60 s + per-link-state re-fit** from
   camera-verified transitions. The only published route aimed at seated
   recall. Risk: window length costs per-second resolution.
4. **VE-Otsu per-capture threshold with a bimodality guard.** Handles
   within-day drift. Risk: unimodal captures; the guard is mandatory.
5. **FRID adjacent-window CoV ratio** as a transition detector feeding a
   state machine. Risk: blind to continuous occupancy, spikes on steps.
6. **WiSH-style camera-confirmed empty-reference refresh**, triggered by a
   link-state change. Cheapest; a workaround.
7. **Ratio-phase variants of every feature.** Risk: never tested seated.

Heartbeat is deprioritised: no claim about it survived, and nothing citable
exists on feasibility at 42 Hz on a 2×1 link.

## Two measurements that decide the route

Both on captures we have.

1. **Is the empty-room ratio noise temporally white at 42 Hz on both kinds
   of day?** Lag-1 ACF of the detrended ratio in camera-empty windows on
   09-21 (clean) and 09-16/17 (noisy). If it sits at −1/T ± 1/√(F_eff·T) on
   both, the WiDetect route solves the baseline problem; if the noisy floor
   is coloured, it does not, and step gating or a refreshed reference are
   what remain.
2. **What is F_eff on our link**, and is it large enough that a Q-function
   threshold gives a usable false-alarm rate at T ≈ 42 without pushing η
   above the seated signal?

## Refuted, and not answered

Nine claims failed verification and are not to be cited from this survey:
an ESP32 Z-score / "calibration maturity" / abstain-on-drift preprint
(arXiv 2607.26665, three claims, 0–3 each); an RF Doppler-spectrum presence
detector (arXiv 2603.10845, two claims, 0–3); DeMan's eigenvalue-vs-variance
numbers (0–3); PADS's phase-vs-speed result (0–3); WiSH's cross-correlation
feature description (0–3); R-TTWD's ">99 %" read as recall and specificity
(1–2). Several were the only candidates on Doppler-band energy, Z-score
normalisation and explicit abstain states, so those sub-questions are
**unanswered, not answered negatively**. Also without verified evidence:
heartbeat, Fresnel-aware or sensing-SNR subcarrier selection, harmonic
checks, HMM fusion, and the physical origin of the 7.5–10.9 rpm empty-room
peaks. The search budget ran out during verification, so "no contradiction
found" is weak evidence throughout.

## Human against non-human movers (2026-10-06, partial)

Asked after the survey: is there work on telling a person from a non-human
mover (pet, fan, robot vacuum) with CSI? The session's search budget was
spent by then, so this is one verified source, not a survey; the planned
experiment in `docs/plans/nonhuman_motion_plan.pdf` (fixed and oscillating
fan, robot vacuum, with and without a seated person) is the measurement
that will answer it for this room.

**Zhu, Hu, Wu, Wang, Wang, Liu. Scaling WiFi Sensing for Ubiquitous Home
Monitoring: Lessons from Real-World Deployment on Millions of Devices. arXiv
2506.04322 (2025, revised 2026-08).** From a deployment on millions of
consumer devices (the WiDetect lineage), non-human motion from "pets,
robots" is named a critical deployment problem; "a biomechanics-based
classifier … reduces non-human false alarms from 63.1 % to 8.4 %", with
92.61 % human-motion detection accuracy across uncontrolled homes. Only the
abstract was read; the classifier's features and the truth protocol are not
yet extracted, and the figures are the vendor's own. What it establishes is
that at scale, **two thirds of motion alarms in homes were non-human before
the classifier** — the plan's H3 (robot vacuum fires the motion channel) is
the expected outcome, not a corner case, and a motion-only rule cannot be
the final verdict in a room with such movers. Still to acquire: the paper's
reference list on this point, and any CSI work that uses the breathing
channel as the human check (a fan has no 10–30 rpm line; an oscillating fan
at 5–10 s per sweep does, which is the plan's H2).

## Sources

Primary, verified against full text unless noted.

- Zhang F., Wu C., Wang B., Lai H.-Q., Han Y., Liu K. J. R. *WiDetect: Robust Motion Detection with a Statistical Electromagnetic Model.* Proc. ACM IMWUT 3(3):122, 2019. DOI 10.1145/3351280.
- Gong L., Yang W., Man D., Dong G., Yu M., Lv J. *WiFi-Based Real-Time Calibration-Free Passive Human Motion Detection.* Sensors 15(12):29896, 2015. DOI 10.3390/s151229896.
- Qian K., Wu C., Yang Z., Liu Y., Zhou Z. *PADS: Passive Detection of Moving Targets with Dynamic Speed using PHY Layer Information.* IEEE ICPADS 2014.
- Wang X., Zhang L., Shu F. *Human Motion Segmentation via Spatiotemporally Dual-Constrained Density Estimation with Commodity Wi-Fi Device.* Sensors 26(11):3303, 2026. DOI 10.3390/s26113303.
- Zeng Y., Wu D., Xiong J., Yi E., Gao R., Zhang D. *FarSense: Pushing the Range Limit of WiFi-based Respiration Sensing with CSI Ratio of Two Antennas.* Proc. ACM IMWUT 3(3):121, 2019. DOI 10.1145/3351279.
- Wu D., Zeng Y., Zhang F., Zhang D. *WiFi CSI-based device-free sensing: from Fresnel zone model to CSI-ratio model.* CCF Trans. Pervasive Comput. Interact. 4:88–102, 2022.
- Zhu H., Xiao F., Sun L., Wang R., Yang P. *R-TTWD: Robust Device-Free Through-The-Wall Detection of Moving Human With WiFi.* IEEE JSAC 35(5), 2017. DOI 10.1109/JSAC.2017.2679578. (abstract and citing papers only)
- Wu C., Yang Z., Zhou Z., Liu X., Liu Y., Cao J. *Non-Invasive Detection of Moving and Stationary Human With WiFi.* IEEE JSAC 33(11):2329–2342, 2015. DOI 10.1109/JSAC.2015.2430294.
- Soltanaghaei E., Sharma R. A., Wang Z., Chittilappilly A., Luong A., Giler E., Hall K., Elias S., Rowe A. *Robust and Practical WiFi Human Sensing Using On-device Learning with a Domain Adaptive Model.* ACM BuildSys 2020, 151–160. DOI 10.1145/3408308.3427983.
- Hang C., Zheng Y., Qian K., Wu C., Yang Z., Zhou Z., Liu Y., Chen Y. *WiSH: WiFi-Based Real-Time Human Detection.* Tsinghua Science and Technology 24(5):615–629, 2019. DOI 10.26599/TST.2018.9010091.
- Zhu G., Hu Y., Wu C., Wang W.-H., Wang B., Liu K. J. R. *Scaling WiFi Sensing for Ubiquitous Home Monitoring: Lessons from Real-World Deployment on Millions of Devices.* arXiv 2506.04322, 2025. (abstract only)

Qualifiers cited inside findings: Zhuo et al., *Perceiving accurate CSI phases with commodity WiFi devices*, INFOCOM 2017, DOI 10.1109/INFOCOM.2017.8056964; Wang et al., UbiComp 2016, DOI 10.1145/2971648.2971744; FIMD (Xiao et al., ICPADS 2012) via PADS, not independently read.
