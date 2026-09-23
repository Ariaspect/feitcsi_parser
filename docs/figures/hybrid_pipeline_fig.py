"""Figure: the hybrid presence detector, own-floor default, from CSI frames to accuracy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

W, H = 200, 118
fig = plt.figure(figsize=(22, 13), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

C_IN, C_MOT, C_BR, C_DEC, C_SC, C_PANEL = "#e9eef7", "#fde8d6", "#e6f0fb", "#e9f5e6", "#f6e6f0", "#fbfbf6"

def box(x, y, w, h, title, lines=(), fc="#f4f4f4", ec="#222", lw=1.3, tfs=10.5, lfs=8.6, title_tag=None, radius=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ty = y + h - 2.6
    if title_tag:
        ax.text(x + 1.6, ty, title_tag, fontsize=tfs, fontweight="bold", va="center", ha="left", zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#222", lw=0.8))
        ax.text(x + 1.6 + 0.62 * len(title_tag) + 2.0, ty, title, fontsize=tfs, fontweight="bold", va="center", ha="left", zorder=3)
    else:
        ax.text(x + 1.6, ty, title, fontsize=tfs, fontweight="bold", va="center", ha="left", zorder=3)
    yy = ty - 3.6
    for ln in lines:
        ax.text(x + 1.8, yy, ln, fontsize=lfs, va="center", ha="left", zorder=3, family="DejaVu Sans")
        yy -= 3.05
    return (x, y, w, h)

def arrow(p0, p1, color="#222", lw=1.6, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, color=color, lw=lw, zorder=4,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)

def label(x, y, s, fs=9, **kw):
    ax.text(x, y, s, fontsize=fs, va="center", ha="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), **kw)

# ---------------- panel frames
ax.add_patch(FancyBboxPatch((1.5, 1.5), 45, H - 3, boxstyle="round,pad=0,rounding_size=1.5", fc=C_PANEL, ec="#222", lw=1.8, zorder=1))
ax.add_patch(FancyBboxPatch((49, 1.5), W - 50.5, H - 3, boxstyle="round,pad=0,rounding_size=1.5", fc=C_PANEL, ec="#222", lw=1.8, zorder=1))
ax.text(24, H - 5, "(a)  Hybrid presence detector", fontsize=13, fontweight="bold", ha="center", va="center")
ax.text(52, H - 5, "(b)  Signal processing, own-floor default  (keep 0.65 · 10 s window · lead hold on · bridge off · 2026-09-23)",
        fontsize=13, fontweight="bold", ha="left", va="center")
ax.text(24, H - 8.4, "one 5-min capture → present / empty every second\n→ scored against the camera",
        fontsize=8.6, ha="center", va="center", style="italic", linespacing=1.4)

# ---------------- panel (a)
box(4, 84, 19, 20, "CSI capture (.bin)", ["MT7921 · AP 2 Tx → 1 Rx", "245 live subcarriers", "≈20 Hz (Sept) / 42 Hz (09-21)", "5 min per capture"], fc=C_IN)
box(26, 84, 18, 20, "Camera (1 fps)", ["YOLO person box", "occupied if conf ≥ 0.5 in ROI", "→ ground truth"], fc="#f2f2f2")
box(4, 56, 40, 20, "Hybrid detector (own floor)", ["evidence = motion burst  OR  breathing", "hold 20 s · lead hold 20 s · bridge bursts off", "no empty-room reference: the floor is", "this range's own 20th percentile"], fc="#fff6d6", tfs=10.5)
box(4, 33, 40, 16, "Verdict per second", ["moving · breathing · held · bridged · empty · no data", "present = moving ∪ breathing ∪ held ∪ bridged"], fc=C_DEC)
box(4, 7, 40, 19, "Scoring (truth.py)", ["1 s cells vs camera; empty cells within 5 s of a", "transition are not scored (label margin)", "tp fp fn tn → accuracy · recall · specificity · F1", "reported per capture set (empty / control / …)"], fc=C_SC)
arrow((13.5, 84), (16, 76)); arrow((35, 84), (32, 76)); arrow((24, 56), (24, 49)); arrow((24, 33), (24, 26))
arrow((35, 84), (38, 26), color="#888", lw=1.0, rad=-0.35, ls="--"); label(41.5, 55, "truth", fs=8, color="#666")

# ---------------- panel (b) row A: input + ratio
box(52, 88, 40, 21, "Input", ["CSI frames: complex H[tx, rx, k], k = subcarrier", "keep 2×1 MIMO frames of the AP (source MAC)", "frame times t (μs clock) · camera frames (epoch)"], fc=C_IN, title_tag="M1")
box(96, 88, 100, 21, "CSI ratio & uniform grid  (tiles._presence_grid)",
    ["r_k(t) = H[tx1, rx0, k] / H[tx0, rx0, k]      (two stored planes: |r| in dB, ∠r in rad → complex)",
     "cancels per-packet CFO/phase & Rx gain (same packet, same Rx chain) · frames with no ratio are dropped",
     "resample real & imag to a uniform grid at fs = median frame rate · gaps linearly filled → 'fabricated' mask",
     "dead subcarriers (all-NaN) removed → grid  r[t, k]   (n_samples × n_live)"], fc=C_IN, title_tag="M2", lfs=8.4)
arrow((92, 98.5), (96, 98.5))

# ---------------- panel (b) row B: motion + farsense
box(52, 50, 60, 34, "Motion channel  (hybrid.fractional_motion)",
    ["per frame:   m(t) = median_k  |r_k(t) − r_k(t−1)|  /  |r_k(t)|",
     "per second:  L(s) = median{ m(t) : t ∈ s }        (1 s cells)",
     "unknown(s) = gap fraction in s > 0.5  →  L = NaN",
     "",
     "OWN FLOOR:   F = P20{ L(s) }  over the scored range (no other capture)",
     "threshold:    T = max( 2·F ,  0.10 )",
     "burst(s):      L(s) > T  for ≥ 2 consecutive seconds",
     "",
     "fails when the range has no quiet stretch: F is the occupant"], fc=C_MOT, title_tag="M3")
box(116, 50, 80, 34, "Breathing channel — FarSense  (farsense.prepare / window_step)",
    ["Savitzky–Golay 1.0 s (order 3) on real & imag · high-pass 0 Hz (off)",
     "window 10 s, hop 1 s, band 10–30 rpm → lag range [2.0 s, 6.0 s]",
     "per subcarrier k, per window:",
     "   x_θ(t) = Re{ (r_k(t) − mean) · e^(−jθ) },  θ ∈ 200 steps in [0, π)",
     "   BNR(θ) = max in-band bin / total energy  (FFT 8192, Parseval) → θ* = argmax",
     "   ρ_k(τ) = autocorrelation of x_θ*        keep k with BNR_k ≥ 0.65 · max_k BNR",
     "ρ(τ) = Σ_k BNR_k · ρ_k(τ)  /  Σ_k BNR_k   → first POSITIVE local max in the lag range",
     "output per window (stamped at its centre):  peak p ∈ [−1, 1],  rate = 60·fs/τ* (rpm)"], fc=C_BR, title_tag="M4", lfs=8.4)
arrow((72, 88), (72, 84)); arrow((146, 88), (146, 84))

# ---------------- panel (b) row C: evidence, holds, scoring
box(52, 8, 44, 38, "Breathing evidence  (consistent_breathing)",
    ["window qualifies if  p ≥ 0.2",
     "run: 10 consecutive windows (10 s hop 1 s)",
     "  · ≥ 80 % qualify  (8 of 10)",
     "  · their rates within ± 3 rpm of the",
     "    run's median",
     "→ every second of the run = breathing",
     "",
     "breathing ∧ ¬burst ∧ ¬unknown",
     "",
     "(sparse rule available, off by default)"], fc=C_BR, title_tag="M5")
box(100, 8, 52, 38, "Holds & bridging  (verdict)",
    ["evidence = burst ∪ breathing",
     "trailing hold: 20 s after any evidence",
     "leading hold:  20 s before breathing (lead hold on)",
     "holds that meet from both sides → gap = bridged",
     "bridge bursts = off  (option: run / any fill the stretch",
     "   between two bursts when breathing lies between)",
     "",
     "present = (evidence ∪ holds ∪ bridged) ∧ ¬unknown",
     "state ∈ {moving, breathing, held, bridged, empty, no data}"], fc=C_DEC, title_tag="M6")
box(156, 8, 40, 38, "Scoring  (truth.py)",
    ["camera: occupied frame ⇔ box conf ≥ 0.5",
     "cell truth (1 s): mean occupancy ≥ 0.5",
     "transition = midpoint of a label flip",
     "empty cells within 5 s of a transition",
     "   → excluded (not scored)",
     "",
     "tp fp fn tn over scored cells",
     "accuracy = (tp+tn)/all · recall = tp/(tp+fn)",
     "specificity = tn/(tn+fp) · F1 = 2tp/(2tp+fp+fn)",
     "pooled over a capture set"], fc=C_SC, title_tag="M7")
arrow((156, 50), (140, 46)); arrow((74, 50), (74, 46), color="#c0521e"); label(66, 48, "burst (mask)", fs=8, color="#c0521e")
arrow((90, 50), (112, 46), color="#c0521e", rad=0.2); label(104, 48.3, "burst (evidence)", fs=8, color="#c0521e")
arrow((96, 27), (100, 27)); label(98, 29.5, "breathing", fs=8)
arrow((152, 27), (156, 27)); label(154, 29.5, "present", fs=8)
label(148, 48.5, "p, rate", fs=8, color="#2f5fa8")

# ---------------- footer: parameter strip
ax.text(52, 4.2, "Defaults (own):  keep 0.65 · θ 200 · S-G 1.0 s · high-pass 0 · window 10 s · band 10–30 rpm · peak ≥ 0.2 · run 10 s / 80 % / ±3 rpm · "
        "floor = own P20 · T = max(2F, 0.10) · burst ≥ 2 s · hold 20 s · lead hold on · bridge off · margin 5 s",
        fontsize=8.6, ha="left", va="center", style="italic")
import os
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_pipeline.png"), dpi=150, facecolor="white")
print("saved")
