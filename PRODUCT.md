# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

WiFi sensing researchers running experiments with Intel AX200/AX210 NIC
captures. They analyze CSI data for gesture recognition, breathing
detection, presence sensing, and similar tasks. Their job: inspect signal
quality, compare antennas, validate MIMO behavior, and confirm data is
usable before feeding it into training pipelines or sensing algorithms.

Secondary: students and hobbyists learning CSI-based sensing who need clear
visualization to build intuition about subcarriers, phase wrapping, and
inter-antenna ratios.

## Product Purpose

A realtime heatmap viewer for FeitCSI `.dat` captures that opens any file
at the same speed regardless of size. The researcher inspects amplitude,
phase, and CSI ratio (rx1/rx0) across four linked heatmaps, filters by
MIMO mode and source MAC, and watches live captures grow — all without
re-reading the file on every refresh.

Success: a researcher opens a 200MB capture, zooms into a 2-second window
in under 500ms, confirms the 2x2 frames are where expected, checks the CSI
ratio phase for breathing-like periodicity, and moves on to their
pipeline. The tool is a checkpoint, not a destination.

## Positioning

Bounded-cost exploration. CSIKit's own viewer re-reads and re-decodes the
entire capture on every refresh, so polling a growing 200MB file falls
behind. This tool indexes frames structurally (header scan only, no
payload decode), decodes in vectorised batches, caches decoded blocks in
an LRU, and never returns more cells than the plot has pixels. A 211MB
capture and a 1MB one open at the same speed.

## Operating Context

- Captures are FeitCSI `.dat` files: 272-byte header + csi_length payload,
  self-delimiting, frames appended as the NIC writes them.
- Backend runs locally on the researcher's machine (FastAPI, port 8000).
- Frontend runs locally (Vite dev on 5173, or built into backend's static
  serve on 8000).
- Captures directory is gitignored — files are large and
  environment-specific. Default path: `captures/capture.dat`.
- Real captures have 2-7 source MACs and interleaved 2x1/2x2 MIMO modes.
- The researcher may be watching a live capture grow (realtime mode) or
  exploring a finished one (paused/zoomed).

## Capabilities and Constraints

- Parses `.dat` via CSIKit's `FeitCSIBeamformReader` for header fields,
  pilot interpolation, and RSSI scaling. Does not fork CSIKit — wraps it.
- Four metrics: amplitude (dBm), phase (rad), CSI ratio amplitude (dB),
  CSI ratio phase (rad). Ratio = rx1/rx0 complex division, same tx0.
- Frames with only 1 rx antenna get NaN for ratio (transparent in plot).
- MIMO filter (all / 2x1 / 2x2) and source MAC filter narrow the visible
  data. Filtered-out frames leave NaN holes, not filled from neighbors.
- Tile API: pre-aggregated display-resolution grids. At most 8192 frames
  decoded per request. Stride-sampled ranges marked `X-Tile-Exact: 0`.
- Block decode cache: 4096-frame blocks, 256MB LRU, keyed by file size
  (truncated/rewritten files cannot serve stale blocks).
- Subcarriers arrive centred (index 0 = lowest, N//2 = DC). fftshift must
  NOT be applied — it splits the contiguous spectrum.
- Linked time axis across all four heatmaps. Subcarrier zoom is per-plot.
- Filter change resets view + color scale (CaptureId includes filter).

## Brand Commitments

- Product name: FeitCSI (from feitcsi.kuskosoft.com CSI format spec).
- Hardware: Intel AX200/AX210 NIC.
- No logo, no color palette, no voice constraints beyond the name.

## Evidence on Hand

- `captures/capture.dat` — 2.4MB, 1101 frames, uniform 2x1, 2 source MACs.
- `captures/0811_0812.dat` — 205MB, 92201 frames, interleaved 2x1/2x2
  (971 2x2 frames, 420 geometry transitions), 7 source MACs.
- `captures/0812_0856-.dat` — 141MB, 62976 frames, interleaved 2x1/2x2
  (1153 2x2 frames, 160 transitions), 6 source MACs.
- CSIKit library: `.venv/lib/python3.12/site-packages/CSIKit/`
- FeitCSI format spec: https://feitcsi.kuskosoft.com/csi_format/

## Product Principles

1. **Cost tracks the viewport, not the file.** A 200MB capture and a 1MB
   one open at the same speed. Never re-decode what the user can't see.

2. **Live and explore are the same machinery.** The only difference is
   what triggers a refetch. A frozen view survives any number of polls;
   a live view slides forward preserving duration.

3. **Four metrics, one time axis.** Amplitude, phase, and both CSI ratio
   metrics are read against each other at one instant. Zooming one mirrors
   the time window on all four.

4. **Honest visualization.** Stride-sampled tiles are marked. Real
   dropouts stay NaN. Filtered-out frames leave holes. Silence about
   approximation is a lie.

5. **Wrap CSIKit, don't fork it.** Header parsing, pilot interpolation,
   and RSSI scaling come from upstream. Reproduce its arithmetic
   vectorised across frames, but never diverge from its per-frame results.

## Accessibility & Inclusion

General best practices: contrast, keyboard navigation, screen reader
support where reasonable. Data visualization uses perceptually-uniform
colormaps (viridis for amplitude, twilight for phase) chosen for
color-vision-deficiency safety. No institutional WCAG mandate.
