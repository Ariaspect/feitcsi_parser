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

### On a remote server (backend and frontend, one port)

The way to run this on the collection host. uvicorn serves `/api` **and** the
built frontend from the same origin, so there is nothing to proxy, no CORS, and
no backend port for the client to know about.

```bash
# once, on the server
uv sync                      # backend deps
npm install                  # root: concurrently
npm --prefix frontend install
npm run build                # frontend/dist

# run
HOST=0.0.0.0 npm run serve   # :8000; PORT=9000 to move it
```

Open `http://<host>:8000` — e.g. http://lg:8000. Every feature works over this
one port: capture listing, live polling, tiles, Doppler.

Four things decide whether it stays working:

- **Build before you start, rebuild after frontend changes.** The static mount
  is made at import time only if `frontend/dist` exists, so a server started
  before the first build serves the API and nothing else. `npm run build` then
  restart.
- **One worker.** The frame index, the decoded-block cache (256 MB cap) and the
  capture streams are per-process state. A second uvicorn worker does not share
  any of it: it re-decodes every capture into its own copy, so memory multiplies
  and the cache the panels depend on stops hitting. Leave `--workers` alone.
- **Point it at the captures.** `captures/` in the repo is always readable —
  a symlink into a data mount is the simplest option. To name the directory
  instead, set `FEITCSI_CAPTURE_ROOTS` to an `os.pathsep`-separated list of
  extra roots (`FEITCSI_CAPTURE_ROOTS=/data/csi:/mnt/lab`). Requests for paths
  outside every root are refused. `scripts/csi_live_ship.sh` writes growing
  captures here, and a live view works on them as it does locally.
- **Open the port.** The server binds `0.0.0.0`; the firewall still has to
  allow it, and nothing in the app authenticates — put it on a trusted network,
  or behind something that does.

Under systemd:

```ini
# /etc/systemd/system/feitcsi.service
[Unit]
Description=FeitCSI heatmap
After=network-online.target

[Service]
User=csi
WorkingDirectory=/srv/feitcsi_parser
Environment=HOST=0.0.0.0 PORT=8000
Environment=FEITCSI_CAPTURE_ROOTS=/data/csi
ExecStart=/usr/bin/npm run serve
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Behind nginx or Caddy, pass the client address through
(`uvicorn --proxy-headers --forwarded-allow-ips=127.0.0.1`) and let the proxy
terminate TLS. Tiles are ordinary single responses — a few hundred kB of
float32 — so no streaming or buffering settings are involved.

Tile latency is CPU, not network: a full-extent tile is ~0.4 s on a laptop core
and the eight panels ask at once, so a slower box just refreshes more slowly.
Live polls no longer cancel requests that are still in flight, so a slow server
falls behind gracefully instead of showing nothing.

### From another machine (dev server)

For editing on the box rather than deploying, the Vite dev server also binds
`0.0.0.0`:

```bash
# on the collection host
npm run dev:all
```

Open `http://<host>:5173` — e.g. http://lg:5173. The page fetches only relative
`/api` URLs and Vite forwards that prefix to uvicorn, so the browser sees one
origin here too. uvicorn stays on loopback; nothing but Vite needs to be
reachable.

Two environment variables adjust it:

| Variable | Default | Effect |
| --- | --- | --- |
| `API_TARGET` | `http://localhost:8000` | Where `/api` is forwarded. Point it elsewhere to drive a backend on another box. |
| `VITE_ALLOWED_HOSTS` | unset — any host | Comma-separated hostnames allowed to reach the dev server. A leading dot matches subdomains (`lg,.example.com`). |

Vite rejects requests whose `Host` header it does not recognise, which would
otherwise block every LAN name the machine answers to; unset, the check is off,
which is what a lab network wants. Set `VITE_ALLOWED_HOSTS` on anything less
trusted. `vite preview` (`:4173`) carries the same proxy and binding, so a
production build can be checked over the network the same way.

### Production (single port, local)

```bash
npm run build         # builds frontend into frontend/dist
npm run serve         # uvicorn serves API + static frontend at :8000
```

Open http://localhost:8000. Binds loopback by default; see
[On a remote server](#on-a-remote-server-backend-and-frontend-one-port) to
expose it.

## Usage

1. Place FeitCSI `.dat` file at `captures/capture.dat` (or enter path in UI).
2. To explore a finished capture, just enter its path — no need to start
   polling. To watch one grow, click **Run realtime**.
3. Every `refresh_ms` the frontend polls `/api/meta`, which reads the frame
   index only and never decodes payloads. Pixels come from `/api/tile`, which
   is fetched only when the view actually changes.
4. The **Channel** tab renders six heatmaps, and the **Doppler** tab two more
   (see [Doppler](#doppler)). The channel panels are: amplitude (dBm), phase (rad), CSI ratio
   amplitude and phase, then the swap-corrected CSI ratio pair, then the
   time-unwrapped ratio phase, then the raw channel's impulse response
   (CIR). See [Phase views](#phase-views),
   [Swapped rx streams](#swapped-rx-streams), and
   [Channel impulse response](#channel-impulse-response).

Controls:
- **.dat file** — path to a capture, growing or finished.
- **Refresh (ms)** — polling interval.
- **Run realtime** — toggle polling.
- **Source MAC** — required for the corrected and time-unwrapped views, which
  judge frames against their neighbours. Defaults to a single transmitter.

The four base plots are never modified by any derived view.

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
- `metric` — one of `amplitude`, `phase`, `csi_ratio_amplitude`,
  `csi_ratio_phase`, `phase_unwrapped`, `phase_detrended`,
  `csi_ratio_phase_unwrapped`, `csi_ratio_phase_time_unwrapped`
  (see [Phase views](#phase-views)), `csi_ratio_phase_corrected`,
  `csi_ratio_amplitude_corrected` (see [Swapped rx streams](#swapped-rx-streams)),
  `csi_cir` (see [Channel impulse response](#channel-impulse-response))
- `mimo`, `source_mac` — optional filters, `'all'` or a specific value
- `interpolate` — default `true`; see [Interpolation](#interpolation) below

Returns a bare `(num_subcarriers, columns)` little-endian float32 array,
row-major, row 0 = highest subcarrier. The body stays a buffer the client wraps
in a `Float32Array`; metadata rides in headers:

| Header | Meaning |
|---|---|
| `X-Tile-Width` / `X-Tile-Height` | Grid shape. Width is the **snapped** column count and is never more than requested. |
| `X-Tile-T0` / `X-Tile-T1` | The window this tile actually covers — see [the lattice](#the-lattice). Always contains `[t0, t1]`, rarely equals it. **Draw against these, not against the request.** |
| `X-Tile-DT` / `X-Tile-Level` | Seconds per column, and the lattice level that gave it. |
| `X-Capture-TMin` / `X-Capture-TMax` | The whole file's extent, not this tile's window, so a live view can track growth without a second round trip. |
| `X-Tile-Frames` | Frames decoded. |
| `X-Tile-Total` | Frames in `[t0, t1]` before sampling. |
| `X-Tile-Exact` | `1` if no chunk needed stride sampling. |
| `X-Tile-VMin` / `X-Tile-VMax` | Finite extrema, measured on frames sampled across the capture. |
| `X-Tile-PLow` / `X-Tile-PHigh` | 1st/99th percentiles — the robust scale the amplitude plot locks to. |
| `X-Tile-Filled` | Columns filled from a neighbouring frame across a sampling gap. |

Columns are max-hold for amplitude and nearest-frame for phase (a maximum of an
angle is meaningless). A column that receives no frame is linearly
interpolated between its two bracketing frames when within 2x the
95th-percentile inter-frame interval; beyond that, or with `interpolate=false`,
it stays NaN, so a real capture dropout stays visible instead of being painted
over. See [Interpolation](#interpolation).

#### The lattice

A tile's columns are quantised to a fixed grid: column *c* at level *L* covers
`[c·dt, (c+1)·dt)` with `dt = 1 ms · 2^L`, measured from the capture's own
`t=0`. The server picks the finest level whose columns still fit the requested
`width` — never finer than the capture's median frame spacing — snaps the
window outwards to column boundaries, and reports what it served in
`X-Tile-T0/T1/DT`. The client crops.

The columns are therefore a property of the capture, not of the window that
asked for them. That is the whole point:

- **Panning** shifts columns that keep their values, instead of re-aggregating
  every column over new boundaries. Before the lattice a one-pixel pan changed
  the picture rather than moving it.
- **Live follow** appends columns on the right instead of re-binning the whole
  grid on every poll — the crawl that made a growing capture unreadable.
- **Stride sampling** is anchored to each chunk's own frame range, so a pan no
  longer changes *which* frames a sampled column was built from.
- **Caching works.** Tiles are assembled from 256-column chunks keyed on
  `(capture, metric, level, chunk, filters, frames-in-chunk)`. A request keyed
  on an exact window could never hit; these hit constantly. On a
  64,559-frame capture: full extent 0.394 s cold and 0.008 s warm, a live poll
  0.007 s against 0.375 s before, a pan 0.013 s against 0.036 s.

The cost is that `dt` steps by 2x between levels rather than tracking the pixel
width continuously, and the window drawn is up to one column wider than the one
requested.

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

### `GET /api/doppler`

Subcarrier-averaged Doppler spectrogram. Same binary contract as `/api/tile` —
bare little-endian float32 body, metadata in headers — so the same canvas
renderer draws it. The body is `(win // 2 + 1, n_windows)`, row-major,
**row 0 = highest Doppler frequency**.

Query params:
- `path`, `t0`, `t1` — as `/api/tile`
- `metric` — `amplitude` or `csi_ratio_phase_time_unwrapped`
- `win_seconds` — STFT window length in seconds (default 10, max 600). **Clamped** to what the range holds if longer, so zooming in never blanks the panel; `X-Doppler-WinSeconds` reports what was used
- `overlap` — window overlap fraction (default 0.5)
- `max_gap_fraction` — blank a column once more than this fraction of its window is interpolated across dropouts (default 0.5)
- `mimo`, `source_mac`, `interpolate` — as `/api/tile`

| Header | Meaning |
|---|---|
| `X-Doppler-Width` / `X-Doppler-Height` | Grid shape: windows × frequency bins. |
| `X-Doppler-Fs` | The capture's **own median frame rate** over the frames in range. |
| `X-Doppler-FMax` | Nyquist, `Fs/2`. This file's real ceiling. |
| `X-Doppler-Win` / `X-Doppler-Hop` | Window and hop, in samples. `Win` is always even. |
| `X-Doppler-WinSeconds` | The window actually used, in seconds — may be less than requested. |
| `X-Doppler-Blank` | Columns dropped for being mostly interpolated across dropouts. |
| `X-Doppler-Frames` | Frames that fed the transform. |
| `X-Doppler-ColT0` / `X-Doppler-ColT1` | First and last **column centres** — half a window inside the requested range. |

`X-Doppler-Fs` comes from frame timing, never from a requested width.
Resampling a 5 Hz capture onto a wider grid manufactures peaks above its own
Nyquist; the same mistake decimates a faster capture and truncates the top of
its band. It is measured over the whole (filtered) capture rather than the
frames in view, so the frequency axis does not rescale as you zoom —
`capture.dat`'s 1st-percentile interval is 0.42 ms, and a short slice that
happens to be all burst would otherwise report 1144 Hz against its true
5.1 Hz.

Returns `400` for an unknown metric, or a range too short to hold even a
minimum 8-sample window.

### `GET /api/captures`

Lists capture files under `captures/`, newest first. Takes no parameters.
Populates the file dropdown.

```json
[{"filename": "2026-08/day1/capture.dat",
  "path": "/srv/feitcsi/captures/2026-08/day1/capture.dat",
  "size_bytes": 144104752,
  "mtime": 1755500000.0}]
```

The walk is recursive, so captures may be organised into subdirectories.
`filename` is the path **relative to `captures/`** — a top-level file is a bare
name, a nested one carries its subdirectory, which keeps two files called
`capture.dat` apart in the dropdown. `path` is absolute and is what the client
sends back to `/api/meta` and `/api/tile`; a `filename` works there too — see
[Capture paths](#capture-paths).

Directory symlinks **are** followed, so a large archive can be mounted in with
`ln -s /mnt/data/csi captures/archive`. Symlink cycles terminate, dangling
links are skipped, and the walk stops at 8 levels deep. `.dat` is FeitCSI,
`.bin` is MediaTek — but the suffix only decides what gets *listed*; the parser
is chosen by sniffing the bytes, so a misnamed file still reads correctly.

### `GET /api/health`

Returns `{"status": "ok"}`.

## Doppler

Two panels, in their own tab: an STFT of the amplitude time series, and one of
the time-unwrapped ratio phase. Both are subcarrier-averaged — each subcarrier
is transformed and the magnitude spectrograms are averaged, which lifts SNR
without asking you to pick a tone.

Raw wrapped phase is deliberately **not** offered. Its 2π jumps are broadband
steps that dominate an FFT and read as motion that is not there.

**Doppler here is unsigned.** Amplitude and unwrapped phase are real signals,
so their spectra are conjugate-symmetric and the sign of the shift is not
recoverable: approaching and receding motion are indistinguishable. Signed
Doppler would need the complex CSI. The axis is one-sided, `0 … fs/2`.

### What it can see

Doppler shift is `f_d = 2v/λ`, so at 5 GHz a 1 m/s hand movement sits near
**33 Hz** — above the Nyquist of every capture here. Frame rate is the limit:

| capture | frames | PRF | worst gap | Nyquist |
|---|---|---|---|---|
| `capture.dat` | 1,101 | 5.1 Hz | 401 ms | ±2.5 Hz |
| `csi_20260813_030001.dat` | 65,219 | 11.6 Hz | **22,946 ms** | ±5.8 Hz |
| `20260822_070002.bin` | 60,796 | 17.9 Hz | 116 ms | ±8.9 Hz |

So this is a respiration-and-presence instrument, not a gesture one. Measured
across six slices each: `20260821_170002.bin` (evening) carries a 0.08–0.28 Hz
line at 2.4–4.2× contrast; `20260822_070002.bin` (07:00 the next morning) is
flat at 1.05–1.23×. Occupied versus empty is legible. A flat panel is a real
reading, not a broken one.

Because the interesting band is the bottom few percent of the axis,
**shift + wheel** to zoom the frequency axis is how you actually read these.

### Windows and gaps

The window is set in *seconds*, not frames, because frame rate varies from 5 to
18 Hz across captures — a fixed frame count would mean a different physical
window per file. Longer window, finer frequency resolution, fewer columns.

Frames are resampled onto a uniform grid first, at the capture's median rate.
Dropouts are bridged by interpolation, but only proportionally: a column is
blanked once more than `max_gap_fraction` (default 50%) of its window was
invented.

Bridging *every* gap unconditionally is the tempting simplification and is
wrong here. These captures hold six holes over 10 seconds, one of 22.9 s, and
interpolating those produces a perfectly flat stretch — and flat reads as "no
motion", not "no data", which on a presence panel is the one lie that matters.
Blanking on *any* gap is equally wrong in the other direction: on
`csi_20260813_030001.dat`, 130 gaps of which 66% are under a second (35 s in
total) blanked **32.3%** of columns against 6.3% real dead time. The
proportional rule leaves **3.8%** blank, and the 22.9 s hole still shows as a
hole.

Subcarriers that are non-finite for the whole capture (DC/guard band, dropped
pilots — 11 of 256 on an MTK file) are excluded from the average rather than
poisoning it. Each window is zero-padded before the FFT, which interpolates the
frequency axis for a smoother panel without claiming extra resolution.

## Capture paths

Every endpoint that takes a `path` runs it through one chokepoint,
`resolve_capture_path`, which confines reads to the capture roots. Requests
outside them are `404`, whatever they name.

Accepted spellings:

| Form | Example |
|---|---|
| Root-relative, nested included | `capture.dat`, `2026-08/day1/x.dat` |
| Legacy repo-root-relative | `captures/capture.dat` |
| Absolute, inside a root | `/srv/feitcsi/captures/2026-08/day1/x.dat` |

A `..` component is **rejected outright** rather than normalised, so no request
can climb out of a root. `/etc/passwd`, `../pyproject.toml`, and an absolute
path to anything outside the roots all return `404`.

Symlinks *inside* a root are followed wherever they point, including outside
it. Placing one requires filesystem access to the server, which is an
operator's deliberate act — unlike a path an HTTP caller supplies. So the
normal way to attach a large archive stays:

```bash
ln -s /mnt/data/csi captures/archive
```

### Extra roots

`captures/` is always a root. A deployment that would rather point at a data
mount than symlink it in can name more, separated by `:`:

```bash
FEITCSI_CAPTURE_ROOTS=/mnt/data/csi:/srv/archive npm run serve
```

Only `captures/` is walked by `/api/captures`; extra roots are readable by
path but are not listed.

> **This is confinement, not authentication.** The API still has no login and
> `allow_origins=["*"]`. Anyone who can reach the port can read every capture
> under every root. Keep it on loopback behind an SSH tunnel, or put a reverse
> proxy with auth in front, before exposing it.

## Interpolation

One flag, `interpolate` (default `true`), governs filling gaps in two
different axes, and the frontend's **Interpolate** toolbar button toggles
both together:

- **Subcarrier axis.** Structural nulls — pilots, the DC/guard band — are
  filled by interpolation across neighbouring subcarriers within a frame.
  This is `backend.batch.decode_frames`'/`backend.mtk.decode_frames`'
  `interpolate` parameter; see their docstrings for the null-run and
  MAX_NULL_RUN details.
- **Time axis.** A display column with no decoded frame in it — a gap
  between samples, not a real capture dropout — is filled by linear
  interpolation between the two frames bracketing it, weighted by how far
  the column's centre sits between their timestamps. Only gaps within 2x the
  95th-percentile inter-frame interval are touched; a real dropout is wider
  than that and stays NaN regardless of the flag, so turning interpolation
  off never hides one.

`false` leaves both axes exactly as decoded off the wire — every structural
null and every sampling gap NaN. This is the honest view of what the hardware
actually reported; `true` (the default) is the smoothed one most panels are
easier to read in.

The time-axis fill is a plain weighted average for every metric except the
three wrapped-phase ones (`phase`, `csi_ratio_phase`,
`csi_ratio_phase_corrected`). Averaging a wrapped angle directly is wrong at
the branch cut: a frame at +3.1 rad and its neighbour at -3.1 rad are 0.08 rad
apart on the circle, and a plain average lands near 0 rad — the long way
round. Those three metrics are blended as `exp(i*phase)` and converted back
with `atan2`, which follows the circle instead. Every other metric, including
the `*_unwrapped` and `*_detrended` views, is by construction no longer an
angle on a circle and takes the plain average.

## Phase views

Everything the decoder produces comes out of `np.angle`, so the four base
metrics are **wrapped** to (−π, π]. The ±π banding in those plots is the
branch cut, not structure in the channel. They use a cyclic colormap
(matplotlib's twilight) so a wrap does not paint a false hard edge.

Three derived metrics undo parts of that. They are computed per frame on full
subcarrier vectors, before tile column aggregation — aggregation drops frames,
and a phase sequence with holes cannot be unwrapped.

| Metric | Transform | What it fixes |
|---|---|---|
| `phase_unwrapped` | unwrap along subcarriers | Removes the 2π sawtooth *within* a frame. Does nothing across frames. |
| `phase_detrended` | unwrap + per-frame least-squares line removal | Removes the random per-packet offset (CFO/PLL) and the sampling-time-offset slope. This is what makes raw phase comparable across packets. |
| `csi_ratio_phase_unwrapped` | unwrap along subcarriers | Same sawtooth removal for rx1/rx0. |
| `csi_ratio_phase_time_unwrapped` | unwrap along **time**, on the corrected ratio | Removes the sawtooth as the channel moves, so each subcarrier's trace is continuous accumulated phase. This is the motion view. |

The subcarrier-axis metrics remain available over the API but are no longer
plotted; the UI shows the time-unwrapped ratio instead.

Measured on `captures/capture.dat`, mean across-frame standard deviation per
subcarrier: wrapped 1.81 rad → unwrapped **11.60** rad → detrended 0.68 rad.
Unwrapping alone makes raw phase *worse* across frames, because the per-packet
offset and slope are no longer folded back into (−π, π] — which is why the
detrend is a toggle and not applied silently.

Two things the detrend is deliberately not applied to:

- **The CSI ratio.** rx1/rx0 shares an oscillator and clock between the two
  chains, so the division already cancels the common offset and most of the
  slope. Fitting a line there removes signal, not nuisance. (On a MediaTek
  capture the two halves are transmit chains rather than receive ones, which
  cancels the same offsets but leaves a deliberate ramp behind — see
  [MediaTek captures](#mediatek-captures).)
- **Anything needing absolute time-of-flight.** The fit takes any genuinely
  linear-in-frequency component with it. Standard sanitization in the
  SpotFi/PhaseFi lineage, fine for motion sensing, fatal for ranging.

Unwrapped metrics are not angles on a circle any more, so the frontend gives
them a sequential palette and fits the color scale to the first tile's
1st/99th percentile band, exactly as amplitude does. One caveat inherent to
unwrapping: `np.unwrap` anchors each row on its first subcarrier, so a frame
whose lowest subcarrier sits near the branch cut can shift by a whole 2π
relative to its neighbours, appearing as an isolated column jump. The wrapped
panels above are unaffected — that is part of why they stay.

## Swapped rx streams

Two independent corruptions hit the CSI ratio, and they have different
signatures. Both render as inverted-looking colour on a cyclic colormap,
which is why they are easy to confuse by eye:

| | what happens to the ratio | phase | dB amplitude | looks like |
|---|---|---|---|---|
| **swap** | reciprocal (`rx0/rx1`) | negated | negated | isolated columns |
| **rotation** | multiplied by −1 | shifted by π | unchanged | multi-second blocks |

Together they give four states (`r`, `−r`, `1/r`, `−1/r`), and both are
corrected. Measured on one transmitter over 8000 frames of
`csi_20260813_030001.dat`, rotations occur at 62 of 7999 transitions — rare
events, but each one flips a whole block until the next one flips it back.
One such block spanned 2127.9–2135.9 s; correcting it removed all 79 affected
columns from the tile.

The swap's signature is exact: the complex ratio is inverted, which
**negates the phase and negates the dB amplitude** together. On screen it
reads as an isolated column in mirrored colours — visually distinct from
the transparent columns where no frame exists at all.

Measured on one transmitter over 6000 frames, an affected frame deviates from
the mean of its two neighbours by 1.664 rad where a normal frame deviates by
0.116; negating it gives 0.103, back at baseline. The dB amplitude agrees
(5.034 → 0.625). Only the ratio metrics are affected — rx0's own amplitude and
phase are undisturbed.

`csi_ratio_phase_corrected` and `csi_ratio_amplitude_corrected` put them back.
Detection runs on the ratio phase for both, so the two panels always agree
about which frames were flipped.

Effect at short inter-packet gaps, where the channel cannot physically have
moved and any large step is therefore an artefact:

| gap | transitions > 0.5 rad, before | after |
|---|---|---|
| < 2 ms | 4.55% | **1.14%** |
| 80–150 ms | 19.85% | 13.57% |
| > 150 ms | 30.02% | 24.94% |

The residual at longer gaps is largely genuine channel evolution, not missed
swaps.

### The algorithm

Orientation is not observable from a single frame, so every decision is made
by comparing frames against each other. Three phase passes run in a loop, each
covering the others' blind spot, then two anchors settle which way up the
result sits — against a **reference measured once for the whole capture**:

0. **Reference.** Comparing frames to each other can only ever produce an
   answer that is self-consistent *within the batch being looked at*, and
   there are always two such answers. Which one a view lands on then depends
   on which frames the view contains — so panning or zooming inverted whole
   panels, at a measured 12% of positions at a 200-frame zoom. Both anchors
   below originally derived their reference from the batch in front of them,
   which is what made this structural rather than a tuning problem.

   `build_reference` measures the two quantities once, from a few thousand
   frames drawn evenly across the capture: the median dB band profile and the
   mean phase direction. Both are majority statistics and the corruption is
   the minority (4.1% of frames swapped, 14.2% a π out), so raw frames
   already point the right way in bulk and no prior correction is needed to
   measure them — the chicken-and-egg does not arise. Every tile of that
   capture is then judged against the same absolute orientation, and a
   frame's verdict is the same whichever view asked for it. Cost is one
   decode of ~4096 frames per capture per transmitter (~0.2 s), cached.

   It is per *transmitter*, because the band profile is a property of one
   pair of antennas; blending two senders' profiles anchors to neither. So a
   selected `source_mac` is required, and `reference is not None` is the
   single switch on whether the ratio is corrected at all — a view can never
   be half-corrected, nor claim a correction it did not get. No reference is
   also issued when a sender's band is too flat to correlate against or its
   phase names no clear direction. In every such case the ratio is passed
   through exactly as decoded and the tile reports `X-Tile-Anchored: 0`, which
   the heatmap surfaces as *⚠ uncorrected — select a transmitter*.

1. **Chain.** Every adjacent pair is fitted twice — `phi_i` against
   `phi_prev`, and `-phi_prev` — and the better fit wins. Its *offset* then
   says whether a π rotation came along too. Both decisions accumulate as
   parities, so a run of affected frames needs no special handling.
   *Blind spot:* it propagates. One unreadable transition — a dropout, or a
   stride-sampled view where neighbours sit 400 ms apart instead of 100 —
   and everything downstream stays inverted until another miss undoes it.
2. **Refine.** Each frame is re-decided against the circular mean of its
   neighbours, a consensus no single frame can move, so mistakes stay local.
   *Blind spot:* a large inverted region agrees with itself, and the
   symmetric window straddles a boundary and goes incoherent right where it
   matters.
3. **Merge.** Each candidate split point is judged by comparing the mean of
   the frames *before* it against the mean of those *after*. Averaging many
   frames per side lifts the signal far above what any single pair carries,
   so a boundary no adjacent comparison could resolve becomes obvious.
   Non-maximum suppression keeps one detection per boundary.

Scoring uses the correlation's **magnitude** to choose the orientation and
its **angle** to choose the rotation — magnitude alone is offset-blind and
cannot see a rotation at all.

4. **Amplitude anchor.** Everything above compares frames only to other
   frames, which can place a boundary perfectly and still leave the entire
   region *between* two of them inverted — internally consistent, so no
   phase-based check ever objects. The dB ratio amplitude settles it: a swap
   negates it too, and its shape across the band is fixed by the antennas
   rather than the moving channel. Every 2000-frame chunk of an hourly
   capture correlates +0.955 to +0.999 with the file's median profile, so a
   stretch that anti-correlates is simply wrong.

   With a reference the profile comes from the capture rather than from the
   frames being judged, so there is no risk of confirming a window's own
   inversion, no iteration to a fixed point, and no run-length gate — an
   absolute reference can only ever flip a sign that is already wrong. It
   also drops the ≥400-frame minimum, which is what used to leave every
   zoomed-in view with no absolute orientation at all. Without a reference
   the older behaviour stands: only runs of ≥200 frames are re-oriented, and
   isolated frames stay with the phase passes.

   This pass exists because the phase-only version shipped a regression: it
   removed the real isolated swaps and then inverted a 1400-second block on
   top. Across the 20 hourly captures it took frames sitting in the wrong
   orientation from a mean of 4.1% (44.4% on the worst file) to **0.0%**.

   It needs a profile with real shape — below `MIN_PROFILE_STD` it declines
   rather than acting on noise.

5. **Rotation anchor.** The amplitude cannot settle the *rotation* parity,
   because multiplying the ratio by −1 leaves the dB amplitude exactly where
   it was. And rotations are not rare — on an hourly capture ~24% of
   transitions carry a π offset (the distribution is sharply bimodal: 6174
   transitions below 0.3 rad, 1955 between 2.90 and π, only 28 in between),
   so the parity toggles thousands of times and one miscount flips everything
   after it.

   What anchors it is the phase's own mean direction. Each frame's circular
   mean over subcarriers points somewhere, and that direction is set by the
   fixed offset between the antennas rather than the moving channel: measured
   over an hour it holds at +1.3 rad end to end, while a wrongly-rotated
   stretch sits at −1.8. A full π apart, separable by sign. Across the 20
   captures this took columns sitting a π from the capture mean from 14.2% to
   **0.3%**. As with the amplitude anchor, a reference turns this from a
   comparison with the batch into a single pass against a fixed direction.

6. **Stride-sampled views.** A decimated view is not a frame sequence: its
   rows are seconds apart, so the chain and refine passes have nothing to
   compare against and are skipped entirely. The anchors carry the whole
   decision, judging each frame on its own against the reference — and
   *without smoothing*, because smoothing works by borrowing evidence from
   neighbours that share a state, which sampled rows do not. Measured against
   the native-rate answer, this errs on 5% of frames at every stride from 2
   to 64; averaging 5 neighbours errs on 14–23%, and leaving the frames
   uncorrected errs on 28%. What is lost is the isolated single-frame swaps,
   which at those zooms occupy a fraction of one column and cannot be seen.

Windows are corrected with a 128-frame context margin that is then trimmed
off, so the frames at a tile's or a cache block's edge are decided on the same
neighbours a full-capture pass would have given them. Without it a 200-frame
view disagreed with the capture-scale answer somewhere in 28 of 198 positions;
with it, 4 — and none of them an inversion.

Measured across all 20 hourly captures at the full-file view (the worst case,
where stride sampling puts adjacent columns ~4.5 s apart), inverted column
*transitions* fall from **2514 to 4** — and 3 of those 4 are confirmed genuine
channel rotations at full resolution, not misses.

That transition count is a trap worth flagging, because it was believed for
longer than it should have been: a uniformly inverted block has only **two**
boundaries no matter how wide it is, and transitions touching a NaN gap are
skipped entirely. A count of 4 was therefore perfectly consistent with two
enormous inverted regions. The honest metric is the fraction of *frames* whose
orientation disagrees with the amplitude profile, which is what the anchor
above is measured on. Cost is ~0.5 s per round on
a full 8192-frame tile and ~0.04 s on a typical zoomed view.

### Properties worth knowing

Three properties of the method are worth knowing before relying on it:

- **It needs a single transmitter selected, and does nothing without one.**
  Detection is relative — a frame is judged against its neighbours. On
  `source_mac=all` consecutive frames come from different senders (14% and 7%
  same-sender on the two transmitters of an hourly capture), so `_chain`
  compares two senders 86–93% of the time and the confidence gate declines.
  Measured on the same frames, correcting on one transmitter's own sequence
  leaves **0.3–0.6%** of steps above π/2, where correcting the interleaved
  stream leaves **11.2–11.6%** — against 12.4–13.2% uncorrected. It bought
  almost nothing and reported itself as done, so on `all` the correction is
  now skipped outright and the panel shows the raw ratio.
- **The flag means "opposite orientation to frame 0", not "anomalous".**
  Parity accumulates along the sequence, so roughly half the frames in a long
  batch carry the flag even though individual swaps are rare.
- **Orientation needs the capture, not the window.** Which of the two states
  is "correct" is not observable from a single frame and not observable from
  a window either — see step 0. With a reference the answer is a property of
  the capture; without one it falls back to a majority vote over the batch,
  which is stable only as long as the batch is, and which is wrong exactly
  when the minority assumption fails (a window landing inside a long
  corrupted stretch).
- **A genuine π channel rotation is indistinguishable from an artificial
  one** when it lands between two sampled instants. At heavy zoom-out the
  remaining handful of inverted-looking transitions are real events in the
  room, and correcting them would be destroying data. Zoom in and they
  resolve into ordinary continuous motion.

A note on the fitting metric, because it is a trap worth documenting: the
alignment score `|mean(exp(i(x - y)))|` is **invariant to a constant phase
offset**, so a π-rotated block scores a perfect 1.0 against its neighbours
and reads as "identical". That is exactly why rotations went unnoticed at
first. `_fit` therefore returns the offset alongside the quality, and the
rotation decision reads the offset while the swap decision reads the
quality — scoring with the offset folded in would rate a perfectly-explained
rotated frame as unrelated and hide it completely.

The cause of either is unidentified. All 272 header bytes were scanned and
none separates affected frames from normal ones; the documented `antenna_a`
and `antenna_b` bits in `rate_flags` are constant, and bit 20 — the only bit
that varies — does not correlate (its two groups align at 0.999 *as-is*).
No per-frame property identifies them either. Note that CSIKit parses only
about eight fields out of the 272 header bytes, so most of the header has no
known semantics: "nothing found" is not "nothing there".

## Channel impulse response

`csi_cir` takes the raw channel — `amplitude`/`phase`, i.e. rx0/tx0, not the
rx1/rx0 ratio — and inverse-FFTs it along the subcarrier axis into delay:
`backend.cir.csi_to_cir`. Where every other panel reads the channel in
frequency, this one reads it in time-of-flight — echoes at different path
lengths separate into different delay taps instead of showing up as ripples
across subcarriers.

**Deliberately the raw channel, not the ratio.** An earlier version of this
metric was built on the swap-corrected ratio instead, which cancels the
receiver's CFO/SFO and packet-detection timing offset because both chains
share them — that is exactly why *that* IFFT centred so cleanly on zero
delay. There is no second chain here to cancel anything against, so this CIR
is **not zero-referenced**: what it shows is real propagation delay plus
whatever uncalibrated hardware/timing offset the receiver adds on top of it,
and that combined offset drifts a little between frames as CFO/SFO drift.
Measured on 1500 frames of a single sender on each capture on hand:

```
                  peak offset from centre     frame-to-frame spread (std)
MTK  (256 taps)          13 taps                        3.1 taps
FeitCSI (242 taps)        8 taps                         1.7 taps
```

The offset is real and roughly stable — not noise splashed across the row —
so the panel is still meaningful, just not for absolute time-of-flight. Read
it for **relative** delay between echoes: a second, smaller peak next to the
main one is a reflection arriving that many taps later than the direct path,
regardless of where the main peak itself happens to sit.

Two things have to be undone before the IFFT means anything, both because
every metric in this pipeline is already laid out DC-centred rather than in
raw FFT bin order (see [Data Format](#data-format) below):

- **`ifftshift` before the transform.** `np.fft.ifft` expects index 0 at DC
  with positive frequencies ascending and negative ones wrapped to the top;
  the centred array has DC in the middle. Skipping this does not blur the
  result, it relocates every echo to the wrong delay.
- **`fftshift` after it, for display.** Raw IFFT output puts delay 0 at
  index 0, ascending — the ordinary DSP convention, and what
  `backend.cir.csi_to_cir` returns. `csi_to_cir_centred` (fftshift, delay 0
  where index 0 *would* be, now at the row's middle) is what `csi_cir`
  actually serves. This lets the CIR panel reuse the same centred axis the
  frequency-domain panels already have — the frontend needs a different
  *label* (`Delay tap`, via `Heatmap`'s `axisLabel` prop) but no different
  axis logic — and it still matters here even though the real peak is not
  at centre: a fractional-tap delay splits its energy across the row's two
  edges in the raw layout (tap 0 and the *last* tap), and centring reunites
  that split wherever in the row it lands.

Null subcarriers — the MTK guard band, whatever CSIKit dropped as unusable on
a FeitCSI capture — arrive as NaN and are read as zero energy on that tone
before the transform, which is the standard reading for a punctured
spectrum and is exactly what the MTK hardware's own null-tone encoding
already means. A frame with no primary stream decoded is left NaN rather
than computed as a confident flat zero, which would otherwise be
indistinguishable from "measured, no echoes".

One asymmetry between the two capture formats is worth knowing before
reading fine structure into this panel: MTK's null bins sit at their true
positions in a uniform 256-bin comb, so zero-filling them reconstructs the
transmitted spectrum faithfully. FeitCSI's array has already had its
unusable subcarriers *deleted* by CSIKit rather than zeroed in place, so the
242-wide comb handed to the IFFT there is not perfectly uniform — the result
is still peaked at the true delay but carries extra sidelobe smearing from
the gaps. Good enough to read off relative timing, not to trust to the last
dB, and not comparable dB-for-dB between the two formats in any case — see
[MediaTek captures](#mediatek-captures) for why their raw channels are not
directly comparable to begin with.

`csi_cir` uses max-hold aggregation like the other magnitude metrics
(peak-preserving when a display column spans several native frames), and is
exempt from the tile layer's usual "same cells have data" invariant for
derived metrics: a CIR row is a delay tap, not a subcarrier, so there is no
per-cell correspondence to a subcarrier-indexed base to preserve. What does
still hold, cell for cell, is *frame* coverage — a column the base channel
had no data for gets no CIR either.

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

### MediaTek captures

Captures pulled off the LG webOS board (`/var/iwtools/iw-priv`, read from
`/proc/net/wlan/csi_data`) are a different format entirely and are detected by
sniffing, not by extension. Records are self-delimiting TLVs —
`magic 0xAC | length u16 LE | tag(1) len(2 LE) value ...` — samples are 14-bit
signed rather than `int16`, and subcarriers arrive in raw FFT bin order, so
here `fftshift` **is** required, the exact opposite of the FeitCSI rule above.
A frame is a *group* of up to four records closed by bit 15 of tag 18, never a
run sharing a timestamp: the millisecond clock ticks mid-group.

**The ratio is a transmit pair.** Records are indexed by `tpi` and `rpi`. The
axes are told apart by the transmitter's cyclic shift, which 802.11 applies
per transmit chain and to nothing else: dividing along `tpi` leaves a ramp,
dividing along `rpi` leaves none. So `tpi` indexes the AP's transmit chains,
and it is `tpi` that gets mapped onto the pipeline's rx axis, because
everything downstream reads the ratio off `rx1/rx0`. **A MediaTek capture's
"CSI ratio" therefore compares two antennas at the far end of the link**,
where a FeitCSI capture's compares two on the receiver. Both cancel the
receiver's CFO/SFO — the two halves come out of one packet, one receive chain
and one timing recovery — but they are not the same physical quantity and
should not be pooled or plotted on a shared scale.

`rpi` plane 1 is real signal, not a dead chain (59.43 dB against plane 0's
59.95 dB, smooth across frequency at 0.995), so it is not obvious from the
file alone what it is on a board documented as 1x1. It is not used, because
its ratio is far noisier per frame: at the shortest frame gap the `tpi` ratio
moves 0.158 rad where the `rpi` ratio moves 0.896 rad, already most of the
1.571 rad a uniformly random phase would give.

**The cyclic shift is removed by default.** Because the two halves of the
ratio are two different transmit chains, the fixed per-chain delay the
standard mandates does not cancel; it survives as a pure linear phase ramp.
It measures 396.6–402.7 ns across all six captures on hand — the −400 ns the
standard specifies for a second stream — with no frame of any file dissenting
on the sign. At 80 MHz that wraps the phase about 30 times across the band,
which is enough to make any statistic taken along the subcarrier axis
meaningless: the raw ratio phase of `capture1.bin` has a circular resultant of
0.010, indistinguishable from uniform, where removing the ramp lifts it to
0.842.

The ramp is measured once per file — not per view, for the reason the
orientation reference is also anchored to the file — and subtracted about each
band's own DC bin, so a 20 MHz and an 80 MHz frame land on one phase
reference. Only `csi_ratio_phase` is affected; `ratio_amp` is unchanged
because the correction is a unit-magnitude rotation, and `amplitude`/`phase`
read rx0 and never see it. Pass `deslope=False` to `mtk.decode_frames` for the
ratio as decoded. A capture whose frames disagree on the ramp's sign, or whose
ramp is shallower than 0.05 rad/subcarrier, gets no correction at all rather
than a number not worth trusting.

Two things to watch when using this ratio. It depends on the AP continuing to
send two streams — 59 of `capture1.bin`'s 1290 groups are single-stream and
have no ratio at all — where a genuine receive pair is always present because
it is your own hardware. And no rate word exists in the format (tag 19 reads 0
on every record), so there is no way to confirm from the file whether the AP
ever applies beamforming; if it did, `tpi` would index precoded combinations
rather than antennas. Nothing in the captures here suggests it happens — no
frame-to-frame jump exceeds 0.664 rad — but it is not provable from the data.
