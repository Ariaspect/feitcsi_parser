"""The test suite's parts: which test files cover which part of the code.

Each test file belongs to exactly one part; ``tests/conftest.py`` marks its tests with the part's
name, so one part runs with ``pytest -m <part>``. ``scripts/test_changed.py`` runs only the tests a
change can reach, and reports them by part.

A new test file must be added here -- ``test_every_test_file_has_a_part`` fails until it is.
"""

from __future__ import annotations

PARTS: dict[str, tuple[str, ...]] = {
    # reading captures: file formats, the frame index, tiles, live growth, the static cache
    "io": ("test_index", "test_mtk", "test_tiles", "test_batch", "test_stream", "test_live_growth",
           "test_lattice", "test_filters", "test_static_cache"),
    # per-frame signal processing and the views drawn from it
    "signal": ("test_ratio", "test_phase", "test_preprocess", "test_agc", "test_cir", "test_doppler",
               "test_doppler_api", "test_spectro", "test_framediff", "test_motion", "test_motionsig"),
    # presence detectors: breathing, FarSense, the hybrid detectors, the LG pipeline, camera truth
    "presence": ("test_presence", "test_presence_api", "test_breathing", "test_farsense", "test_hybrid",
                 "test_hybrid2", "test_hybrid2_calc", "test_lgdetect", "test_lgproc", "test_truth"),
    # the classifier tab: the feature bank, Liu columns, the 11F columns, feature marks
    "classifier": ("test_classifier", "test_liu_features", "test_feature_marks", "test_ml_columns"),
    # the ML tab: AF8 / 11F screen model, the human and anomaly models, ground truth
    "ml": ("test_af8", "test_af8_explain", "test_ml_heads", "test_build_ground_truth"),
    # the app shell: capture listing, capture controls, labels; and this map itself
    "app": ("test_capture_listing", "test_capture_control", "test_labels_api", "test_parts"),
}

PART_OF: dict[str, str] = {t: p for p, ts in PARTS.items() for t in ts}
assert len(PART_OF) == sum(len(ts) for ts in PARTS.values()), "a test file is listed in two parts"
