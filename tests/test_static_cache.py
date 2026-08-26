"""Cache headers on the built frontend served by the backend.

The asset filenames Vite emits are content-hashed, so they are safe to cache
forever. ``index.html`` is not: it is the only file that names the current
bundle hash, and a browser that caches it keeps loading last week's bundle
long after a deploy. Without an explicit Cache-Control, browsers apply
heuristic caching to the HTML and features silently go missing in the UI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _dist_or_skip() -> Path:
    if not (DIST / "index.html").is_file():
        pytest.skip("frontend/dist not built")
    return DIST


def test_index_html_is_revalidated() -> None:
    """The HTML shell must be revalidated, never served from cache blindly."""
    _dist_or_skip()
    client = TestClient(app)
    for url in ("/", "/index.html"):
        resp = client.get(url)
        assert resp.status_code == 200, url
        assert resp.headers.get("cache-control") == "no-cache", url


def test_hashed_assets_are_immutable() -> None:
    """Content-hashed assets carry a long immutable cache."""
    dist = _dist_or_skip()
    asset = next(iter(sorted((dist / "assets").glob("*.js"))), None)
    if asset is None:
        pytest.skip("no built assets")
    resp = TestClient(app).get(f"/assets/{asset.name}")
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "public, max-age=31536000, immutable"
