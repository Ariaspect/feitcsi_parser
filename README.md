# FeitCSI Parser

Realtime heatmap for FeitCSI `.dat` captures (Intel AX200/AX210 NIC).

Stack: FastAPI backend parses `.dat` via [CSIKit](https://github.com/Gi-z/CSIKit),
React + Vite + [uPlot](https://github.com/leeoniya/uPlot) frontend renders
amplitude and phase heatmaps with realtime polling.

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
2. Click **Run realtime**.
3. Backend re-parses `.dat` every `refresh_ms`, returns trailing window of
   `max_packets` packets as JSON.
4. Frontend renders two heatmaps: amplitude (dBm) and phase (rad).

Controls:
- **.dat file** — path to growing capture.
- **Window (packets)** — trailing N packets displayed.
- **Refresh (ms)** — polling interval.
- **Run realtime** — toggle polling.

## API

### `GET /api/snapshot`

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
records. CSIKit handles parsing, pilot interpolation, and subcarrier
filtering. Subcarriers are fftshifted to signed-frequency order
(-N/2 .. +N/2-1).

See https://feitcsi.kuskosoft.com/csi_format/ for the on-wire spec.
