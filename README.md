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
4. Frontend renders two heatmaps: amplitude (dBm) and phase (rad).

Controls:
- **.dat file** — path to a capture, growing or finished.
- **Refresh (ms)** — polling interval.
- **Run realtime** — toggle polling.

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
- `metric` — `amplitude` or `phase`

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
angle is meaningless). A column that receives no frame borrows the nearest one
within 2x the 95th-percentile inter-frame interval; beyond that it stays NaN, so
a real capture dropout stays visible instead of being painted over.

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
