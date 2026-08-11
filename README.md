# FeitCSI Parser

Parse FeitCSI `.dat` captures (Intel AX200/AX210 NIC) and render realtime
amplitude / phase heatmaps from an actively growing file.

Built on [CSIKit](https://github.com/Gi-z/CSIKit) for parsing and
[Bokeh](https://bokeh.org/) for realtime visualization. Replaces the
`nexmon_csi_parser` pattern (which used `csiread` + Broadcom PCAPs + Streamlit).

## Setup

This project uses `uv` for dependency management.

```bash
uv sync
```

## Captures

Place FeitCSI `.dat` files in `captures/`:

```text
captures/
  capture.dat        # actively written by FeitCSI
```

The default app path is `captures/capture.dat`. Override in the sidebar.

## Realtime Heatmap

```bash
uv run bokeh serve src/feitcsi_parser/bokeh_app.py --show --port 5006
```

Bokeh pushes image updates over WebSocket — no full re-render, no PNG
serialization. Smooth updates at 5-10+ Hz depending on file size and network.

Sidebar controls:

- **FeitCSI .dat file** — path to the growing capture.
- **Trailing window (packets)** — last N packets to display. Older packets
  scroll off the heatmap.
- **Refresh interval (ms)** — how often the file is re-read (default 200 ms).
- **Colormap** — viridis / plasma / inferno / magma / cividis / turbo /
  blues / greens / reds / oranges / purples / greys.
- **Run realtime** — toggle polling on/off (pause to inspect a frame).

The app re-parses the whole file on each tick (CSIKit reads the file in one
pass). For very large files, increase the refresh interval or rotate the
capture.

## Library API

```python
from feitcsi_parser import load_capture, tail_window, plot_heatmap

capture = load_capture("captures/capture.dat")
# capture.amplitude  -> (frames, subcarriers) dBm, fftshifted
# capture.phase      -> (frames, subcarriers) radians [-pi, pi]
# capture.time_seconds -> (frames,) relative seconds

window = tail_window(capture, max_packets=200)
fig = plot_heatmap(window, metric="amplitude", cmap_name="viridis")
fig.savefig("heatmap.png")
```

`plot_heatmap` returns a matplotlib figure (useful for static export or
scripts). For live visualization use the Bokeh app.

## Data Format

FeitCSI `.dat` files are binary: a sequence of
`272-byte header + CSI block` records. Each CSI value is 4 bytes
(signed int16 real + signed int16 imag). Subcarrier count depends on
rate format (HT/VHT/HE) and bandwidth (20/40/80/160 MHz). CSIKit
handles pilot interpolation and subcarrier filtering.

See https://feitcsi.kuskosoft.com/csi_format/ for the on-wire spec.

## Notes

- `fftshift` is applied on the subcarrier axis by default for signed-frequency
  ordering (subcarriers run -N/2 .. +N/2-1). Disable with `fftshift=False` in
  `load_capture` if you need raw order.
- Timestamps are derived from `ftm_clock` (3.125 ns tick counter) with u32
  overflow handling, matching CSIKit's logic. They are relative seconds
  (first packet = 0.0).
- No local tests — FeitCSI hardware required to produce `.dat` files.
