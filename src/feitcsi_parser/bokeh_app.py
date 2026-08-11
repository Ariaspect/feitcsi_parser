"""Realtime FeitCSI heatmap Bokeh server app.

Reads from an actively growing .dat file produced by FeitCSI (Intel AX200/AX210),
re-parses on a periodic callback, and renders amplitude + phase heatmaps over
a trailing packet window. Bokeh pushes updates over WebSocket — no full
re-render, no PNG serialization.

Run:
    uv run bokeh serve src/feitcsi_parser/bokeh_app.py --show

Or with custom args (path is configured in the sidebar at runtime):
    uv run bokeh serve src/feitcsi_parser/bokeh_app.py --show --port 5006
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
    Spinner,
    TextInput,
    Toggle,
    Select,
)
from bokeh.layouts import column, row
from bokeh.palettes import (
    Blues256,
    Cividis256,
    Greens256,
    Greys256,
    Inferno256,
    Magma256,
    Oranges256,
    Plasma256,
    Purples256,
    Reds256,
    Turbo256,
    Viridis256,
)
from bokeh.plotting import figure, curdoc

from feitcsi_parser.parser import FeitCSICapture, load_capture, tail_window


PALETTES = {
    "viridis": Viridis256,
    "plasma": Plasma256,
    "inferno": Inferno256,
    "magma": Magma256,
    "cividis": Cividis256,
    "turbo": Turbo256,
    "blues": Blues256,
    "greens": Greens256,
    "reds": Reds256,
    "oranges": Oranges256,
    "purples": Purples256,
    "greys": Greys256,
}

DEFAULT_DAT_PATH = "captures/capture.dat"
DEFAULT_WINDOW = 200
DEFAULT_REFRESH_MS = 200
DEFAULT_PALETTE = "viridis"


def _empty_image(n_subcarriers: int = 1) -> np.ndarray:
    return np.zeros((n_subcarriers, 1), dtype=float)


def _build_heatmap_figure(
    title: str,
    palette: list[str],
    low: float,
    high: float,
    color_label: str,
) -> tuple[figure, ColumnDataSource, LinearColorMapper]:
    """Build one heatmap figure with image glyph + color mapper."""
    source = ColumnDataSource(data={
        "image": [_empty_image()],
        "x": [0.0],
        "y": [0.0],
        "dw": [1.0],
        "dh": [1.0],
    })
    color_mapper = LinearColorMapper(palette=palette, low=low, high=high)

    p = figure(
        title=title,
        width=1200,
        height=350,
        x_axis_label="Time (s)",
        y_axis_label="Subcarrier bin",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=source,
        color_mapper=color_mapper,
    )
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=8,
        title=color_label,
    )
    p.add_layout(color_bar, "right")
    return p, source, color_mapper


def _update_source(
    source: ColumnDataSource,
    capture: FeitCSICapture,
) -> None:
    """Push capture matrix into the image ColumnDataSource.

    Bokeh image glyph expects image[row, col] where row = y axis (subcarrier),
    col = x axis (time). Our matrix is (frames, subcarriers) → transpose.
    """
    if len(capture) == 0 or capture.num_subcarriers == 0:
        source.data = {
            "image": [_empty_image()],
            "x": [0.0],
            "y": [0.0],
            "dw": [1.0],
            "dh": [1.0],
        }
        return

    matrix = capture.amplitude  # (frames, subcarriers)
    t = capture.time_seconds

    # Bokeh wants image as list-of-2D arrays. Use float32 to cut payload.
    image = matrix.T.astype(np.float32)
    if t.size > 1:
        t_min = float(t[0])
        t_max = float(t[-1])
        dw = max(t_max - t_min, 1e-6)
    else:
        t_min = 0.0
        dw = 1.0

    n_sc = capture.num_subcarriers
    y_low = -n_sc // 2

    source.data = {
        "image": [image],
        "x": [t_min],
        "y": [y_low],
        "dw": [dw],
        "dh": [n_sc],
    }


def _make_update_callback(
    amp_source: ColumnDataSource,
    phase_source: ColumnDataSource,
    amp_mapper: LinearColorMapper,
    phase_mapper: LinearColorMapper,
    status_msg,
    path_input: TextInput,
    window_spinner: Spinner,
    last_refresh_label,
) -> callable:
    """Build the periodic callback closure."""
    import time

    state = {"last_size": 0, "last_refresh": 0.0}

    def update() -> None:
        path = Path(path_input.value)
        if not path.exists():
            status_msg.text = f"<b style='color:red'>File not found: {path}</b>"
            return

        try:
            capture = load_capture(path)
        except Exception as exc:  # noqa: BLE001
            status_msg.text = f"<b style='color:red'>Load error: {exc}</b>"
            return

        window_size = int(window_spinner.value)
        window = tail_window(capture, max_packets=window_size)

        _update_source(amp_source, window)
        _update_source(phase_source, window)

        if len(window) > 0 and window.num_subcarriers > 0:
            phase_image = window.phase.T.astype(np.float32)
            phase_source.data["image"] = [phase_image]

            amp_finite = window.amplitude[np.isfinite(window.amplitude)]
            if amp_finite.size > 0:
                amp_low = float(np.percentile(amp_finite, 2))
                amp_high = float(np.percentile(amp_finite, 98))
                if np.isclose(amp_low, amp_high):
                    amp_low -= 1.0
                    amp_high += 1.0
                amp_mapper.low = amp_low
                amp_mapper.high = amp_high

            phase_finite = window.phase[np.isfinite(window.phase)]
            if phase_finite.size > 0:
                phase_low = float(np.percentile(phase_finite, 2))
                phase_high = float(np.percentile(phase_finite, 98))
                if np.isclose(phase_low, phase_high):
                    phase_low -= 0.1
                    phase_high += 0.1
                phase_mapper.low = phase_low
                phase_mapper.high = phase_high

        state["last_size"] = len(capture)
        state["last_refresh"] = time.time()
        amp_range = ""
        if len(window) > 0 and window.amplitude.size > 0:
            finite = window.amplitude[np.isfinite(window.amplitude)]
            if finite.size:
                amp_range = (
                    f", amp range: [{finite.min():.1f}, {finite.max():.1f}] dBm"
                )
        status_msg.text = (
            f"<b>OK</b> — total packets: {len(capture)}, "
            f"window: {len(window)}, subcarriers: {capture.num_subcarriers}, "
            f"chipset: {capture.chipset}, BW: {capture.bandwidth} MHz"
            f"{amp_range}"
        )

    return update


def main() -> None:
    doc = curdoc()
    doc.title = "FeitCSI Realtime Heatmap"

    # Controls
    path_input = TextInput(value=DEFAULT_DAT_PATH, title="FeitCSI .dat file", width=500)
    window_spinner = Spinner(
        low=1, high=10000, value=DEFAULT_WINDOW, step=50, title="Trailing window (packets)", width=200
    )
    refresh_spinner = Spinner(
        low=50, high=10000, value=DEFAULT_REFRESH_MS, step=50, title="Refresh interval (ms)", width=200
    )
    palette_select = Select(
        value=DEFAULT_PALETTE,
        title="Colormap",
        options=list(PALETTES.keys()),
        width=200,
    )
    run_toggle = Toggle(label="Run realtime", button_type="success", active=False, width=150)

    # Status display
    from bokeh.models import Div
    status_msg = Div(text="<b>Idle</b> — toggle 'Run realtime' to begin.", width=1000)

    # Build two heatmap figures
    amp_fig, amp_source, amp_mapper = _build_heatmap_figure(
        title="FeitCSI — amplitude",
        palette=PALETTES[DEFAULT_PALETTE],
        low=0.0,
        high=60.0,
        color_label="Amplitude (dBm)",
    )
    phase_fig, phase_source, phase_mapper = _build_heatmap_figure(
        title="FeitCSI — phase",
        palette=PALETTES[DEFAULT_PALETTE],
        low=-np.pi,
        high=np.pi,
        color_label="Phase (rad)",
    )

    # Callbacks
    update = _make_update_callback(
        amp_source=amp_source,
        phase_source=phase_source,
        amp_mapper=amp_mapper,
        phase_mapper=phase_mapper,
        status_msg=status_msg,
        path_input=path_input,
        window_spinner=window_spinner,
        last_refresh_label=None,
    )

    # Palette change: update both color mappers in place
    def on_palette_change(attr, old, new):
        palette = PALETTES[new]
        amp_mapper.palette = palette
        phase_mapper.palette = palette

    palette_select.on_change("value", on_palette_change)

    # Refresh interval change: remove old callback, add new with new period
    callback_state = {"cb": None}

    def on_refresh_change(attr, old, new):
        if callback_state["cb"] is not None:
            doc.remove_periodic_callback(callback_state["cb"])
            callback_state["cb"] = None
        if run_toggle.active:
            callback_state["cb"] = doc.add_periodic_callback(update, int(new))

    def on_toggle_change(attr, old, new):
        if new:  # running
            callback_state["cb"] = doc.add_periodic_callback(
                update, int(refresh_spinner.value)
            )
            run_toggle.label = "Pause"
            run_toggle.button_type = "danger"
        else:
            if callback_state["cb"] is not None:
                doc.remove_periodic_callback(callback_state["cb"])
                callback_state["cb"] = None
            run_toggle.label = "Run realtime"
            run_toggle.button_type = "success"

    refresh_spinner.on_change("value", on_refresh_change)
    run_toggle.on_change("active", on_toggle_change)

    # Layout
    controls = row(
        path_input,
        window_spinner,
        refresh_spinner,
        palette_select,
        run_toggle,
        sizing_mode="scale_width",
    )
    doc.add_root(column(controls, status_msg, amp_fig, phase_fig, sizing_mode="scale_width"))


main()
