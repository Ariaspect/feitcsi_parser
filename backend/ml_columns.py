"""The 11-feature screen model's inputs, and its P(person), as classifier-tab columns -- display only.

The ML tab's model (``af8_env1``, 11 features: AF8's ten plus X_gate_dc) is
computed per unit here so its inputs can be compared with the bank and the
Liu columns in the compare table. Nothing in the bank reads them; the model
itself is unchanged -- these are its own numbers on the unit's window.

The model counts its window in seconds from the capture's first frame,
half-open, and refuses a window with under 30 s (``af8.MIN_SECONDS``) of usable grid,
so a unit shorter than that reads blank. A capture the model cannot decode (a
FeitCSI ``.dat``) reads blank too.
"""

from __future__ import annotations

from pathlib import Path

from backend import af8_env1

KEYS = tuple(f"ml_{k}" for k in af8_env1.FEATURES) + ("ml_p_person",)


def unit_features(path: Path, t0_rel: float, t1_rel: float) -> dict[str, float]:
    """The model's 11 inputs and P(person) on [t0_rel, t1_rel) seconds from the first frame; NaN where it cannot judge."""
    try:
        f = af8_env1.features(path, t0_rel, t1_rel)
        out = {f"ml_{k}": float(f[k]) if f.get(k) is not None else float("nan") for k in af8_env1.FEATURES}
        out["ml_p_person"] = float(af8_env1.probability(f))
        return out
    except Exception:        # too short, undecodable, too few frames: blank, not an error for the whole bank
        return {k: float("nan") for k in KEYS}


_DESC = {
    "A_slope3": ("A slope", [-0.5, 1.0], 3, "Log-log slope of the median frame step D(tau) at 0.25, 2 and 5 s: how fast change grows with lag. An exact combination of the two ratios (R^2 = 1), so its own weight means little."),
    "A_r025_2": ("A D0.25/D2", [0.0, 1.5], 3, "D(0.25 s) / D(2 s): near 1 for noise, small for slow coherent motion. The model's strongest negative weight (-3.72)."),
    "A_r5_2": ("A D5/D2", [0.5, 2.0], 3, "D(5 s) / D(2 s): how much more the channel has wandered at 5 s than at 2 s."),
    "A_p90": ("A P90", [0.0, 0.3], 4, "90th percentile over 1 s blocks of the 2 s-lag step (block mean): how much the channel moved."),
    "F_pkmax": ("F peak max", [0.0, 1.0], 3, "Highest FarSense breathing peak over the window's sweep (31 subcarriers)."),
    "F_pkmed": ("F peak med", [0.0, 1.0], 3, "Median FarSense breathing peak over the sweep. Weight +1.25."),
    "F_run": ("F run", [0.0, 60.0], 0, "Longest run of FarSense windows with peak >= 0.25."),
    "F_rpm_sd": ("F rpm sd", [0.0, 10.0], 2, "Spread of the breathing rates over those windows, rpm."),
    "C3_revisit": ("C3 revisit", [0.0, 1.0], 3, "How far D(tau) falls back below its running maximum at lags of 10-30 s: a rotating fan returning to the same angle. Weight -3.00."),
    "C3xRun_diff": ("C3 x run diff", [0.0, 30.0], 2, "C3 times the FarSense run on r(t) - r(t - T), T the swing period: breathing left after the fan's repetition is cancelled."),
    "X_gate_dc": ("X gate dc", [0.0, 30.0], 2, "The stationary-fan switch (-log10 eigenvector drift x (1 - C3)) times the residual FarSense run after projecting out the fan's top 2 directions. Weight +2.94."),
}

FEATURES: list[dict] = [
    {"key": f"ml_{k}", "label": f"11F {_DESC[k][0]}", "test": "ML 11F", "status": "display only",
     "reference": None, "axis": _DESC[k][1], "decimals": _DESC[k][2],
     "description": f"Input {i + 1} of the ML tab's 11-feature screen model (af8_env1, {af8_env1.TRAINED_ON}), on this unit's window. {_DESC[k][3]}",
     "role": "display only -- an input of the ML tab's model, not of anything on this tab",
     "separates": "see the ML tab's weights and the KPI tables", "not_separates": "--"}
    for i, k in enumerate(af8_env1.FEATURES)
] + [
    {"key": "ml_p_person", "label": "11F P(person)", "test": "ML 11F", "status": "display only",
     "reference": None, "axis": [0.0, 1.0], "decimals": 2,
     "description": f"The 11-feature model's P(person) on this unit's window (person when > {af8_env1.THRESHOLD:g}).",
     "role": "display only -- the ML tab's verdict, recomputed on this unit",
     "separates": "see the ML tab's KPI tables", "not_separates": "--"},
]

assert [f["key"] for f in FEATURES] == list(KEYS)
assert len(af8_env1.FEATURES) == 11 and set(_DESC) == set(af8_env1.FEATURES)
