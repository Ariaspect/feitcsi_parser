# FeitCSI Parser

Realtime heatmap for FeitCSI `.dat` captures (Intel AX200/AX210 NIC).

Stack: FastAPI backend parses `.dat` via [CSIKit](https://github.com/Gi-z/CSIKit),
React + Vite + [uPlot](https://github.com/leeoniya/uPlot) frontend renders
amplitude and phase heatmaps with realtime polling.

## Setup

Backend (Python, `uv`):

```bash
uv sync
```

Frontend (Node, `npm`):

```bash
cd frontend && npm install
```

## Run

### Development (single command)

Both backend (`:8000`) and frontend (`:5173`) in parallel:

```bash
cd frontend && npm run dev:all
```

Open http://localhost:5173

### Development (two terminals)

Backend on `:8000`:

```bash
uv run uvicorn backend.app:app --reload --port 8000
```

Frontend on `:5173` (proxies `/api` → `:8000`):

```bash
cd frontend && npm run dev
```

### Production (single port)

Build frontend:

```bash
cd frontend && npm run build
```

Run backend (serves built frontend from `frontend/dist/`):

```bash
uv run uvicorn backend.app:app --port 8000
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
