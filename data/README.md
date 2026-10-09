# data/

## ground_truth.csv

Training and evaluation labels for the presence models: one row per
**one-minute capture** (camera span 45–75 s), nothing longer. Generated, never
hand-edited — change the table or the rules in `scripts/build_ground_truth.py`
and rerun it on the collection host:

```
python scripts/build_ground_truth.py --captures /home/lg_csi/lg_csi_captures
```

| column | meaning |
|---|---|
| `capture` | stem, `YYYYMMDD_HHMMSS` (KST) |
| `env` | 1 = up to 09/30, 2 = from 10/01 (different room and placement) |
| `label` | `occupied` (a human is present) or `empty`; **the only training target** |
| `activity` | stratum for error analysis: `static`, `small_move`, `large_move`, `standing_walk`, `unknown`; for empties `none`, `robot_vacuum`, `fan_static`, `fan_rotating` |
| `nonhuman` | 1 when a non-human motion source (robot, fan) was running; always `label=empty` |
| `source` | where the label came from: `table` (operator's sheet, env#2), `camera` (env#1), `sidecar` |
| `camera_fraction`, `camera_edited` | fraction of camera frames with a person; 1 if the sidecar was hand-corrected |
| `use` | 1 = use for training/evaluation; 0 = listed for completeness, `reason` says why |

Rules of use: split by environment (or by day inside env#2), never by random
capture — back-to-back one-minute captures of the same session are near
duplicates. Score `nonhuman=1` rows as negatives always; fit the presence
stage without them if the model is a single binary classifier (they are 75 %
of the negatives and would teach "motion ≠ human").
