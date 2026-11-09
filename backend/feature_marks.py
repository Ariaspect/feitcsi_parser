"""Which bank columns do real work, and what for -- the marks the classifier tab draws.

Evidence: ``feature_marks.json``, a sweep of every column over 6 captures per
(room, condition) in env1, env2 and env3 (209 captures, 2026-10-28). For each
matched pair it holds one AUC per room: the chance that a row of the second
group reads higher than a row of the first.

Pairs:
- ``empty|still``: empty room vs a still person.
- ``sfan|+still``: static fan alone vs the same fan with a still person.
- ``rfan|+still``: the same for the rotating fan.
- ``sfan|+small``, ``rfan|+small``: the same with small movement.
- ``robot|+pers``: robot vacuum alone vs with a person (env1, env3 only).
- ``all y0|y1``: every no-person row vs every person row.
- ``empty|mover``: empty vs a fan or robot alone, for the mover output.

A mark's level is read off that evidence, not chosen: two stars when every
room is at least 0.9 (or at most 0.1, "lower"), one star at 0.75 (0.25).
Which columns carry a note at all -- the "relevant or well performing" ones
-- is a judgement, written below with each one's strength and weakness.

Caveats that hold for every mark:
- Six captures per cell, three rooms.
- env1 and env2 captures are in the 11-feature model's training set, so its
  columns there are in-sample; env3 is the held-out room.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA = json.loads((Path(__file__).with_name("feature_marks.json")).read_text())
AUC: dict[str, dict[str, list[float | None]]] = _DATA["auc"]
SOURCE: str = _DATA["source"]

PAIR_LABEL = {
    "empty|still": "presence (empty vs still person)",
    "sfan|+still": "still person under static fan",
    "rfan|+still": "still person under rotating fan",
    "sfan|+small": "small move under static fan",
    "rfan|+small": "small move under rotating fan",
    "robot|+pers": "person beside robot",
    "all y0|y1": "overall person vs no person",
    "empty|mover": "mover alone vs empty (mover output)",
}

# The columns worth reading, with what each is, what it is good at and where it fails.
NOTES: dict[str, dict[str, str]] = {
    "ml_p_person": {
        "explain": "The ML tab's 11-feature model verdict on this unit.",
        "strength": "Best overall in every room (0.96 / 0.97 / 0.97) and the only column that holds for a still person under the rotating fan (0.83 / 1.00 / 0.83).",
        "weakness": "env1 and env2 are in its training set, so only env3 is a fair test. It misses people under the ceiling fan in scenario C, and a fan or robot alone still raises it in env2 and env3 (0.64 / 0.77).",
    },
    "p_occupied": {
        "explain": "The bank's own 7-input logistic, fitted on September data.",
        "strength": "Presence 1.00 in every room; overall 0.87-0.95; a person beside the robot 0.94.",
        "weakness": "A fan or robot alone raises it as well (0.86-0.94), so a machine reads as occupied; a still person under the rotating fan is near chance (0.81 / 0.50 / 0.53).",
    },
    "ml_A_slope3": {
        "explain": "How fast the frame step grows from 0.25 s to 5 s of lag (the model's motion shape).",
        "strength": "A person under the static fan (0.92-0.94) or beside the robot (0.92): a person adds slow drift on top of a machine's steady fast motion. Scale-free.",
        "weakness": "Under the rotating fan it drops to 0.50-0.83. It is an exact mix of the two ratio columns, so read it with them, not as a separate vote.",
    },
    "ml_A_r025_2": {
        "explain": "D(0.25 s) / D(2 s): fast change against slow change.",
        "strength": "Lower with a person under the static fan (0.03-0.19) and beside the robot (0.08); the model's strongest weight.",
        "weakness": "With a still person under the rotating fan it flips between rooms (0.31 / 0.67 / 0.08).",
    },
    "ml_A_p90": {
        "explain": "How much the channel moved: P90 of the 2 s-lag step.",
        "strength": "Presence 0.94-1.00 and mover-vs-empty 0.91-0.98: any motion shows.",
        "weakness": "Any motion shows -- a fan alone looks like a person; no separation of a person under a fan (0.44-0.72).",
    },
    "breath_run20": {
        "explain": "Longest run of 20 s FarSense windows with a breathing peak above 11 rpm.",
        "strength": "Presence 0.90 / 1.00 / 1.00, and it survives the static fan (0.81-0.92).",
        "weakness": "The rotating fan masks breathing (0.31 / 0.47 / 0.93); fidgeting shortens the run.",
    },
    "breath_peak10": {
        "explain": "Highest 10 s FarSense breathing peak in the unit.",
        "strength": "Presence 1.00 in every room; overall 0.80-0.87.",
        "weakness": "A fan or robot alone also raises it (0.86-0.90): machines put lines in the breathing band.",
    },
    "ml_F_pkmed": {
        "explain": "Median FarSense breathing peak over the model's sweep.",
        "strength": "Presence 1.00 in every room; overall 0.76-0.83.",
        "weakness": "Still person under the rotating fan only 0.47-0.69.",
    },
    "ml_X_gate_dc": {
        "explain": "Stationary-object switch times the breathing left after removing the object's two strongest directions.",
        "strength": "Presence 1.00 in every room; a still person under the static fan 1.00 / 0.51 / 1.00.",
        "weakness": "Under the rotating fan it points the wrong way (0.33 / 0.42 / 0.00) -- built for a stationary object.",
    },
    "ml_C3xRun_diff": {
        "explain": "C3 times the breathing run after cancelling the object's repeat period.",
        "strength": "Besides the model's verdict, the only column that leans the right way in every room for a still person under the rotating fan (0.67 / 0.89 / 0.92).",
        "weakness": "Weakest room 0.67; little for small movement under fans (0.42-0.68).",
    },
    "ml_C3_revisit": {
        "explain": "How far the channel returns to an earlier state at 10-30 s lags: the rotating object's tell.",
        "strength": "A person moving under the rotating fan breaks the fan's return (0.06 / 0.00 / 0.08, lower).",
        "weakness": "A still person under the rotating fan is not reliably different (0.33 / 0.06 / 0.69), and the ceiling fan keeps it high with people present -- the scenario C misses.",
    },
    "liu_ms_spread": {
        "explain": "Liu 2: how much psi-hat varies across the unit's 2 s windows.",
        "strength": "Mover alone vs empty 1.00 / 0.97 / 1.00; presence 1.00 / 0.86 / 1.00. A candidate for the mover output.",
        "weakness": "Does not tell a person under a fan from the fan (0.47-0.97 static, 0.33-0.67 rotating).",
    },
    "liu_step_spread": {
        "explain": "Liu 2: spread of the per-second step relative to its median.",
        "strength": "Mover alone vs empty 0.96-0.98; small move under the static fan 0.89-1.00.",
        "weakness": "A still person under either fan is not separated.",
    },
    "liu_coh1": {
        "explain": "Liu 4: correlation of the change between adjacent subcarriers (Liu's proximity feature).",
        "strength": "Presence 1.00 in every room; mover alone vs empty 0.94-1.00: 'something coherent is moving'.",
        "weakness": "Cannot tell a person from a fan (still person under the rotating fan 0.69 / 0.25 / 0.61).",
    },
    "liu_coh1_breath": {
        "explain": "Liu 5: the same correlation on the 0.1-0.7 Hz band.",
        "strength": "Presence 1.00 in every room; small move under the static fan 0.90-1.00; mover 0.96-0.98.",
        "weakness": "A person under the rotating fan flips between rooms (0.56 / 0.42 / 0.31).",
    },
    "delay_spread": {
        "explain": "Spread of the dominant delay of the 2 s change: low when the change repeats its shape.",
        "strength": "Lower for anything moving coherently: mover alone vs empty 0.11 / 0.00 / 0.00, presence 0.00-0.22.",
        "weakness": "Does not say whether the mover is a person; walking is low too.",
    },
    "step_norm": {
        "explain": "Step P90 divided by the unit's own link jitter.",
        "strength": "Small move under the static fan 0.92-1.00; presence 0.86-1.00, on any link state.",
        "weakness": "Under the rotating fan inconsistent (0.50-0.83 still, 0.47-0.94 small).",
    },
    "acf_amp_p90": {
        "explain": "P90 over 2 s windows of psi-hat on the ratio amplitude (WiDetect).",
        "strength": "Still person under the static fan 0.75 / 1.00 / 0.75; small move there 0.89-1.00; presence 0.86-1.00.",
        "weakness": "Under the rotating fan it flips (0.47 / 0.19 / 0.61).",
    },
}


def _level(v: list[float | None]) -> tuple[int, str] | None:
    """(stars, direction) when every room agrees on a side, else None."""
    x = [a for a in v if a is not None]
    if len(x) < 2:
        return None
    if all(a > 0.5 for a in x):
        worst, direction = min(x), "higher"
        sep = worst - 0.5
    elif all(a < 0.5 for a in x):
        worst, direction = max(x), "lower"
        sep = 0.5 - worst
    else:
        return None
    if sep >= 0.4:
        return 2, direction
    if sep >= 0.25:
        return 1, direction
    return None


def mark(key: str) -> dict[str, Any] | None:
    """The tab's mark for one column: its tags (pair, stars, direction, per-room AUC), best stars, and the note; None when unmarked."""
    note = NOTES.get(key)
    if note is None:
        return None
    tags = []
    for pair, v in AUC.get(key, {}).items():
        lv = _level(v)
        if lv:
            tags.append({"pair": pair, "label": PAIR_LABEL[pair], "stars": lv[0], "direction": lv[1], "auc": v})
    tags.sort(key=lambda t: -t["stars"])
    return {"stars": max((t["stars"] for t in tags), default=0), "tags": tags, **note,
            "source": SOURCE, "rooms": _DATA["rooms"]}


def apply(features: list[dict[str, Any]]) -> None:
    """Set ``mark`` on every bank entry (None when unmarked)."""
    for f in features:
        f["mark"] = mark(f["key"])
