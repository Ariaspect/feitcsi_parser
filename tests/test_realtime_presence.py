"""backend.realtime_presence: the 13F / 15F judgment every 5 s on the Realtime stream -- a window fed frame batch by
frame batch equals the one-minute capture's series, the install reference is taken, saved and read back, and the
API starts and cancels an install."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import mtk, realtime, realtime_presence as RP, stream_phys as SP, tfm_phys as TP, tfm_phys_fast as TF
from backend.realtime import RealtimeMonitor

ROOT = Path(__file__).resolve().parents[1]


def _cap() -> Path:
    """A long recording (scenario C's): a one-minute capture is 60 s, short of a complete 61 s stream window."""
    row = next((r for r in TP.META["check"] if r["fold"] == "scnC" and (ROOT / "captures" / f"{r['stem']}.bin").exists()),
               None)
    if row is None:
        pytest.skip("no scenario C recording on this machine")
    return ROOT / "captures" / f"{row['stem']}.bin"


def _fed(path: Path, judge: RP.PresenceJudge, seconds: float) -> mtk.MTKIndex:
    """The first ``seconds`` of ``path`` into ``judge`` in 0.2 s batches, as the monitor reads a growing file."""
    idx = mtk.MTKIndex(path)
    st = np.asarray(idx._stamps)
    n, end = 0, int(np.searchsorted(st, st[0] + int(seconds * 1000), side="right"))
    while n < end:
        m = min(end, int(np.searchsorted(st, st[n] + 200, side="right")))
        judge.feed(path, idx, np.arange(n, m), offset=0.0)
        n = m
    return idx


def test_a_window_fed_in_batches_is_the_captures_minute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _cap()
    monkeypatch.setattr(RP, "INSTALL_MINUTES", 1)
    J = RP.PresenceJudge(ref_path=tmp_path / "ref.npz")
    J.start_install()
    idx = _fed(path, J, SP.WINDOW_S + SP.LAG_MS / 1e3)      # the first window, complete
    st = np.asarray(idx._stamps)
    row = J.step()
    assert row is not None and row["install_minute"] == 1 and row["reference_saved"]
    assert row["first_stamp"] == int(st[0])
    for m in RP.MODELS:
        ref = TP.matrix(TF.series(path, 0.0, SP.WINDOW_S, channels=TP.META["models"][m]["channels"])["S"], m)
        mine = np.load(tmp_path / "ref.npz")["minutes"][0][[RP.CHANNELS.index(c) for c in TP.META["models"][m]["channels"]]]
        assert np.array_equal(np.isnan(mine), np.isnan(ref)) and np.allclose(mine, ref, rtol=1e-9, atol=1e-12, equal_nan=True)
        assert row[m] == pytest.approx(SP.judge(ref, SP.Reference(m, [ref]), RP.FOLD)["p"], abs=1e-5)
    again = RP.PresenceJudge(ref_path=tmp_path / "ref.npz")       # read back after a restart
    assert again.ref_info["minutes"] == 1 and set(again.refs) == set(RP.MODELS)
    assert not J.installing and J.status(0, 3600)["reference"]["minutes"] == 1


def test_no_reference_means_no_verdict(tmp_path: Path) -> None:
    path = _cap()
    J = RP.PresenceJudge(ref_path=tmp_path / "none.npz")
    _fed(path, J, 70.0)
    row = J.step()
    assert row is not None and not any(m in row for m in RP.MODELS)
    assert "no reference" in J.status(0, 3600)["detail"]


def test_the_api_starts_and_cancels_an_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import app

    m = RealtimeMonitor(board=None, capture_dir=tmp_path, local_lock=str(tmp_path / "l.lock"), rt_dir=tmp_path / "rt")
    m.presence = RP.PresenceJudge(ref_path=tmp_path / "ref.npz")
    monkeypatch.setattr(realtime, "_monitor", m)
    client = TestClient(app)
    s = client.get("/api/realtime/status", params={"since": 0, "window_s": 600, "psince": 0}).json()["presence"]
    assert s["models"] == [{"key": "seq13pw", "label": "13F"}, {"key": "seq15pwu", "label": "15F"}]
    assert s["fold"] == "scnC" and s["trained_on"] == ["env1", "env2"] and s["reference"] is None and s["rows"] == []
    assert client.post("/api/realtime/install").json()["presence"]["installing"] is True
    assert client.post("/api/realtime/install/cancel").json()["presence"]["installing"] is False
