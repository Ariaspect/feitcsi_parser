"""The suite's parts (tests/parts.py) cover every test file, once."""

from __future__ import annotations

from pathlib import Path

from tests.parts import PART_OF, PARTS


def test_every_test_file_has_a_part() -> None:
    files = {p.stem for p in Path(__file__).parent.glob("test_*.py")}
    assert files - set(PART_OF) == set(), "add these to tests/parts.py"
    assert set(PART_OF) - files == set(), "tests/parts.py lists files that do not exist"


def test_parts_do_not_overlap() -> None:
    assert sum(len(v) for v in PARTS.values()) == len(PART_OF)
