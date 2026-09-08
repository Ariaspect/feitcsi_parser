"""The vendored parser's processing, run over this project's reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import lgproc
from backend.app import app
from backend.vendor import csi_parse as cp

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
client = TestClient(app)


def _capture() -> Path:
    p = CAPTURES / "20260904_192623.bin"
    if not p.exists():
        pytest.skip("capture not available")
    return p


def test_the_tensor_is_built_to_their_axis_convention() -> None:
    """Their H is (packet, rx, tx, subcarrier) with rx = our rpi.

    Getting this backwards would still produce a plausible tensor and a
    plausible number out the far end, so it is worth an assertion: the rx axis
    must carry the two rpi planes, which is the only thing feature_conj can
    conjugate across.
    """
    t = lgproc.build_tensor(_capture(), max_frames=256)
    H = t["H"]
    assert H.ndim == 4 and H.shape[1] == 2 and H.shape[2] == 2
    assert H.shape[3] == t["nsub"]
    # Both rx rows carry signal; a mis-stacked tensor would leave one empty.
    for rx in range(2):
        assert np.abs(H[:, rx, 0, :]).mean() > 0


def test_their_bin_selection_is_reproduced_and_needs_raw_zeros() -> None:
    """active_bins counts non-zero bins, so interpolating first breaks it.

    Filling pilots and DC hands their rule a capture where those bins are never
    null, and it reports them as carrying data -- 245 bins where the raw stream
    gives 234. The processing here must not pre-fill.
    """
    raw = lgproc.build_tensor(_capture(), max_frames=512, interpolate=False)
    filled = lgproc.build_tensor(_capture(), max_frames=512, interpolate=True)
    n_raw = len(cp.active_bins(raw["H"], raw["tx_valid"]))
    n_filled = len(cp.active_bins(filled["H"], filled["tx_valid"]))
    assert n_raw < n_filled
    assert n_raw == 234


def test_the_two_parsers_disagree_about_the_ratio_axis() -> None:
    """Their feature conjugates across rx; ours divides along tx.

    This is the substantive disagreement, and it is settled per capture by one
    number. If conj_rx ever overtakes conj_tx on a real capture, our axis
    choice needs revisiting -- so assert the direction, not just the values.
    """
    s = lgproc.summarise(_capture(), max_frames=1024)
    c = s["coherence"]
    assert c["raw"] < 0.1              # unusable without either correction
    assert c["conj_rx"] > 0.5          # theirs works
    assert c["conj_tx"] > c["conj_rx"] # ours works better


def test_the_endpoint_reports_the_capture() -> None:
    r = client.get(
        "/api/lgparse", params={"path": str(_capture()), "max_frames": 512}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["nrx"] == 2 and body["ntx"] == 2
    assert body["peer"] and body["peer"].count(":") == 5
    assert len(body["occupancy"]) == body["nsub"]
    assert body["rssiMean"] < 0        # dBm, so the signed read matters here too


def test_their_planes_render_as_tiles() -> None:
    """The point of the tab is a heatmap, so the planes must go through tiles.

    A summary of coherence numbers answers "which parser is better"; a heatmap
    answers "what does this parser see", which is the question a capture gets
    opened to ask.
    """
    for metric in ("lg_amplitude", "lg_conj_phase", "lg_conj_amplitude"):
        r = client.get(
            "/api/tile",
            params={"path": str(_capture()), "t0": 0, "t1": 300,
                    "width": 200, "metric": metric},
        )
        assert r.status_code == 200, metric
        assert len(r.content) > 0


def test_bins_their_occupancy_rule_rejects_are_blank() -> None:
    """Guard bands must be NaN, not plotted.

    Drawn as measurements they would take over the colour scale, which is the
    thing their occupancy rule exists to prevent.
    """
    import numpy as np

    from backend.mtk import MTKIndex
    path = _capture()
    index = MTKIndex(path)
    ids = np.flatnonzero(index.num_rx_arr >= 2)[:64]
    planes = lgproc.decode_block(path, index, ids)
    amp = planes["lg_amplitude"]
    assert np.isnan(amp).any()                    # something is masked
    assert np.isfinite(amp).sum(axis=1).max() <= 234
