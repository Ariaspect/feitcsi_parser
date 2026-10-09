# Handoff — motion signal work (`lg_csi_experiments`, 2026-09-29)

For a session picking this up. Everything below is committed on
`feat/farsense` (`9c56ebd`) and deployed to `lg:8002` unless marked otherwise.

## Where the work lives

| | |
|---|---|
| Branch / worktree | `feat/farsense` in `/home/cyphy/feitcsi_parser-wt-farsense` |
| Module | `backend/motionsig.py` |
| API | `/api/motion-signal` |
| UI | **Motion signal** tab on `lg:8002` |
| Docs | `docs/motion_signal.md` |
| Refit | `python -m scripts.fit_motionsig --captures captures` |
| Tests | `tests/test_motionsig.py` — 394 pass across the suite |

⚠️ `/home/cyphy/feitcsi_parser` is the **peer's** checkout on their branch.
Do not work in it. Use the worktree.

## The pipeline as it stands

```
RATIO = H_tx1 / H_tx0        complex, 245 subcarriers
  ├─ magnitude → per-subcarrier median → Hampel → high-pass 0.05 Hz
  │              → variance,  window 4 s
  └─ phase     → unwrap in TIME (restart at dropouts) → Hampel
                 → lag-1 autocorrelation, window 15 s   (before the filter)
  both on one 0.5 s grid → median over subcarriers
  → per-capture 5/25 percentile normalisation      (no camera needed)
  → fixed logistic weights, fixed threshold
```

`label-free`: `variance +0.008124  lag1 +0.372754  intercept −1.181105
threshold +0.206971`

Four things in there overturned the original plan, each worth knowing before
changing anything:

1. **The high-pass corner was a bug.** 0.3 Hz under a 2 s window sat *below*
   that window's first non-DC bin, discarding a band the window could not
   resolve anyway. Invariant now pinned by a test: corner < 1/window.
2. **The two features want different windows.** lag-1 rises monotonically to
   15 s (still posture AUC 0.624 → 0.756); variance peaks at 4 s.
3. **lag-1 belongs on the phase**, settled held-out over 200 capture-level
   stratified splits (+1.90 balanced / +3.52 accuracy, 75–85 % of splits).
   The in-corpus comparison said the opposite — magnitude overfits 9.4 points
   against phase's 6.6.
4. **The label-free percentile pair came down twice**: 20/80 → 10/40 → 5/25.

## Numbers, and which set they came from

Always quote the set — the composition moves the absolute numbers several
points while barely moving the comparisons.

| set | what it is |
|---|---|
| corpus 29 | `motionsig.CORPUS`, the weights' training set |
| CSV 48 | log rows whose first column is a heart or blank |
| **61** | CSV 48 + 13 dawn empties (the log's `새벽 15개` aggregate row, 2 have no camera) |

**Presence accuracy (61 captures, each config at its own best):**

| motion channel | breathing | accuracy mean | median | F1 mean | precision | recall |
|---|---|---|---|---|---|---|
| `\|Δr\|/\|r\|` own-floor | ✗ | 67.6 % | 69.2 % | 51.6 % | 63.6 % | 34.1 % |
| `\|Δr\|/\|r\|` own-floor | ✓ | 84.8 % | 92.0 % | 77.5 % | 85.4 % | 71.3 % |
| variance alone | ✗ | **74.8 %** | 84.5 % | 61.2 % | 71.9 % | 54.2 % |
| variance alone | ✓ | **84.9 %** | 89.7 % | 78.8 % | 78.6 % | **81.5 %** |
| variance + lag-1 | ✗ | 69.3 % | 89.3 % | 52.8 % | 77.9 % | 25.7 % |
| variance + lag-1 | ✓ | 83.5 % | **94.4 %** | **79.2 %** | **89.7 %** | 63.0 % |
| both motions OR | ✓ | 84.8 % | 91.4 % | 77.7 % | 85.2 % | 71.7 % |

**Breathing adds 10–17 points of accuracy to every motion channel.** That is
the single largest effect measured, and it is why the motion work does not
move the hybrid.

**As a motion channel** (within-capture AUC, the 31 two-class captures):
`|Δr|/|r|` 0.606 against var+lag1 **0.878**, winning 28 of 31. The gap is
all in small motion — perched occupant 0.460 → 0.962, seated with phone
0.558 → 0.891.

**But swapping it into the hybrid does not help**: best case −0.1 accuracy
on the heart set, because what the new channel is good at is what breathing
already covers. Measured three times (2 s window, 15 s window, 15 s + phase);
the result improved −2.5 → ±0.0 → −0.1 and never crossed zero.

## Open decisions — these need the user

1. **Accuracy or balanced accuracy as the KPI.** Several conclusions invert
   between them. Nothing else can be settled until this is.
2. **Drop lag-1?** Held-out at the current geometry, variance alone beats
   the pair on balanced accuracy (84.0 ± 5.0 against 81.1 ± 8.5, 67 % of
   splits) and ties on accuracy. The original adoption reason — small-motion
   recall 46.9 → 62.9 at fixed specificity — no longer holds: it is now
   66.4 → 65.3. Fixing the high-pass helped variance and not lag-1.
   *Measured, not acted on.*
3. **Refit after the label fix?** `20260915_133849` is in the corpus and
   gained 4 frames (1.3 %) when manual adjudication started being honoured.
   Weights are still the pre-fix ones.

## Today's label fix — read this before trusting older numbers

`truth.frame_occupied` is now the one rule for "the camera says someone is
there". There were three, and none read `occupied`:

- `_camera_truth` gated on `n>0 && max_conf>=0.5`
- the labels endpoint the **UI strip** draws from tested `bool(boxes)` — no
  confidence gate at all, so the picture disagreed with the numbers
- `eval_pipeline.py` had its own copy

Manual adjudication is recorded as `occupied` + `occupied_source` + `manual`
plus a file-level `manual_override`, deliberately leaving `n`/`max_conf` as
the detector reported them so a correction stays visible as a correction.
All of it was being discarded. Four captures affected:

```
20260915_123534   +4 frames   bent out of frame moving a laptop stand
20260915_124619   +1 frame    same
20260915_133849   +4 frames   moving furniture          ← in CORPUS
20260929_130827   +3 frames   stepped out of view       ← today
```

## Captures from today

| capture | state |
|---|---|
| `20260929_130827` | done · 39.8 % occupied after the manual fill · **scenario not recorded** |
| `20260929_132337` | started 13:23:37, 300 s, 43 Hz · **scenario not recorded** |

Both need their `_meta.json` scenario filled in — the operator knows what
was done, the file does not.

## Things that will bite

- **Independent samples are captures, not windows.** A 15 s window at 0.5 s
  hop shares 97 % of its samples with its neighbour. `n = 15,759` windows is
  29 captures. Every window-level confidence interval here is too narrow.
- **9/21 is 39 % of the 61-capture set** (13 dawn + 11 daytime). One day's
  link state is over-represented.
- **Sample rate is confounded with condition** — 42 Hz is 21.6 % occupied,
  43 Hz is 56.2 %, because the dawn empties are all 42 Hz.
- **Thresholds in the comparison tables were chosen on the evaluation set.**
  Differences under ~1 point are not defensible without a held-out split.
- **A capture occupied throughout cannot be normalised from inside itself.**
  Its own 5th percentile is the occupant. Five threshold forms were tried
  (absolute, quantile, robust, max, min); best was +0.6 points, and any gain
  on those captures came straight out of the empty rooms.
- **No non-human motion in the corpus.** Fan, curtain, robot vacuum: zero.
  The case a linear model provably cannot express is the case with no data.

## Deploy recipe (`lg:8002`)

```bash
git -C /home/lg_csi/feitcsi_parser-farsense pull --ff-only
cd /home/lg_csi/feitcsi_parser-farsense/frontend && npm run build   # not pnpm
kill -INT $(ss -tlnp | grep ':8002 ' | grep -o 'pid=[0-9]*' | cut -d= -f2)
tmux kill-session -t farsense; tmux new-session -d -s farsense -c /home/lg_csi/feitcsi_parser-farsense
tmux send-keys -t farsense '.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8002' Enter
```

uvicorn needs more than 6 s to bind — wait on `tmux capture-pane -t farsense -p`
showing "Uvicorn running", not on `ss`. `:8001` is the webcam relay; never bind it.
