"""Shared test fixtures across test modules."""

from __future__ import annotations

import os

# The suite is many small numpy problems, where OpenBLAS's thread pool costs more than it saves:
# tests/test_classifier.py takes 161 s with the default 12 threads and 105 s with one. Set before
# numpy is first imported (the imports below load it).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import struct
from pathlib import Path

import pytest

from backend.index import HEADER_BYTES
from tests.parts import PART_OF, PARTS

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"


@pytest.fixture
def mixed_geometry_file(tmp_path: Path) -> Path:
    """A small .dat with interleaved 2x1 and 2x2 frames.

    Built from the first 6 frames of capture.dat. Frame 2 is rewritten as
    2x2: num_tx patched to 2, csi_length doubled, payload duplicated so the
    first (rx0, tx0) stream is bit-identical to the original 2x1 frame.

    Requires captures/capture.dat to exist.
    """
    if not CAPTURE.is_file():
        pytest.skip("captures/capture.dat not present")
    raw = CAPTURE.read_bytes()

    orig_cl = struct.unpack("I", raw[:4])[0]
    stride = HEADER_BYTES + orig_cl
    boundaries = [i * stride for i in range(7)]  # 6 frames + end

    data = bytearray(raw[: boundaries[6]])
    off = boundaries[2]
    new_cl = orig_cl * 2
    struct.pack_into("I", data, off, new_cl)
    data[off + 47] = 2  # num_tx: 1 -> 2
    payload_start = off + HEADER_BYTES
    payload_end = payload_start + orig_cl
    data[payload_end:payload_end] = data[payload_start:payload_end]

    target = tmp_path / "mixed.dat"
    target.write_bytes(bytes(data))
    return target


@pytest.fixture(scope="session", autouse=True)
def _allow_tmp_capture_root(tmp_path_factory: pytest.TempPathFactory):
    """Let the API read fixtures built under pytest's temp directory.

    ``resolve_capture_path`` confines reads to ``backend.app.capture_roots()``.
    Most fixtures here are synthesised into ``tmp_path`` rather than dropped in
    the real ``captures/``, so the temp root is registered for the test session
    the same way a deployment would register a data mount.
    """
    import os

    from backend.app import ROOTS_ENV_VAR

    base = str(tmp_path_factory.getbasetemp())
    previous = os.environ.get(ROOTS_ENV_VAR)
    os.environ[ROOTS_ENV_VAR] = base if not previous else f"{previous}{os.pathsep}{base}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ROOTS_ENV_VAR, None)
        else:
            os.environ[ROOTS_ENV_VAR] = previous


@pytest.fixture(scope="session")
def bank_capture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """tests.test_hybrid2's 200 s capture (still, a mover from 80 to 140 s), built once for the tests that only read it."""
    from tests.test_hybrid2 import _capture

    return _capture(tmp_path_factory.mktemp("bank"))


@pytest.fixture(scope="session")
def bank_units(bank_capture: Path) -> dict:
    """``classifier.compute_features(bank_capture, 0, 200)`` -- the whole bank, ~10 s -- computed once and
    shared read-only by the classifier, Liu and 11F-column tests. Do not mutate it."""
    from backend import classifier

    return classifier.compute_features(bank_capture, 0.0, 200.0)


def pytest_configure(config: pytest.Config) -> None:
    for part in PARTS:
        config.addinivalue_line("markers", f"{part}: the {part} part of the suite (tests/parts.py)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test with its file's part, so ``pytest -m classifier`` runs one part."""
    for item in items:
        part = PART_OF.get(Path(str(item.fspath)).stem)
        if part:
            item.add_marker(getattr(pytest.mark, part))
