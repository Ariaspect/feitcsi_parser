"""Run only the tests a change can reach.

    .venv/bin/python scripts/test_changed.py                 # changes since HEAD (staged, unstaged, untracked)
    .venv/bin/python scripts/test_changed.py --base main     # everything this branch changed against main
    .venv/bin/python scripts/test_changed.py --dry-run       # print the selection, run nothing
    .venv/bin/python scripts/test_changed.py --part ml       # one part of the suite (tests/parts.py)
    .venv/bin/python scripts/test_changed.py --all           # the whole suite

Selection is read from the imports, so it needs no upkeep:

* a changed ``backend/<m>.py`` (or a data file beside it, ``backend/<m>.json``) reaches every
  module that imports ``m``, directly or through others, and every test file that imports any of
  them. ``backend/app.py`` is treated as a leaf: it imports nearly everything, so passing through
  it would select the whole suite for any change; a change to ``app.py`` itself runs every test
  that imports the app.
* a changed test file runs itself, and the test files that import helpers from it -- directly, or
  through a ``tests/conftest.py`` fixture that builds on it (those that request the fixture).
* a changed ``data/`` or ``scripts/`` file runs the tests whose module or test source names it.
* ``tests/conftest.py``, ``tests/parts.py``, ``pyproject.toml`` or the requirements: everything.
* frontend: ``tsc -b`` and ``vitest related`` on the changed sources (everything when the
  package, Vite or TypeScript config changed).

Anything else (docs, memory, sidecars) runs nothing.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
TESTS = ROOT / "tests"
FRONTEND = ROOT / "frontend"
EVERYTHING = {"tests/conftest.py", "tests/parts.py", "pyproject.toml", "requirements.txt", "requirements-dev.txt"}
FRONTEND_ALL = {"frontend/package.json", "frontend/package-lock.json", "frontend/vite.config.ts",
                "frontend/vitest.config.ts", "frontend/tsconfig.json", "frontend/tsconfig.app.json",
                "frontend/tsconfig.node.json"}

sys.path.insert(0, str(ROOT))
from tests.parts import PART_OF, PARTS  # noqa: E402

MODULES = {p.stem for p in BACKEND.glob("*.py") if p.stem != "__init__"}


def imports(path: Path) -> tuple[set[str], set[str]]:
    """(backend modules, test files) that a source file imports."""
    mods, tests = set(), set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom):
            m = n.module or ""
            if n.level and not m:
                mods |= {a.name for a in n.names}
            elif n.level:
                mods.add(m.split(".")[0])
            elif m == "backend":
                mods |= {a.name for a in n.names}
            elif m.startswith("backend."):
                mods.add(m.split(".")[1])
            elif m.startswith("tests."):
                tests.add(m.split(".")[1])
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("backend."):
                    mods.add(a.name.split(".")[1])
    return mods & MODULES, tests


def graph() -> tuple[dict[str, set[str]], dict[str, tuple[set[str], set[str]]]]:
    mods = {m: imports(BACKEND / f"{m}.py")[0] - {m} for m in MODULES}
    tests = {p.stem: imports(p) for p in TESTS.glob("test_*.py")}
    return mods, tests


def reached(changed: set[str], mods: dict[str, set[str]]) -> set[str]:
    """The changed modules and every module that imports one of them, not passing through app."""
    users: dict[str, set[str]] = {m: set() for m in mods}
    for m, deps in mods.items():
        if m == "app":
            continue
        for d in deps:
            users[d].add(m)
    out, todo = set(), list(changed)
    while todo:
        m = todo.pop()
        if m in out:
            continue
        out.add(m)
        todo += list(users.get(m, ()))
    return out


def changed_files(base: str) -> list[str]:
    git = ["git", "-C", str(ROOT)]
    tracked = subprocess.run(git + ["diff", "--name-only", base], capture_output=True, text=True, check=True).stdout
    untracked = subprocess.run(git + ["ls-files", "--others", "--exclude-standard"], capture_output=True, text=True,
                               check=True).stdout
    return sorted({f for f in (tracked + untracked).splitlines() if f})


def mentions(name: str) -> tuple[set[str], set[str]]:
    """Backend modules and test files whose source names *name* (a data or script file)."""
    mods = {m for m in MODULES if name in (BACKEND / f"{m}.py").read_text()}
    tests = {p.stem for p in TESTS.glob("test_*.py") if name in p.read_text()}
    return mods, tests


def conftest_fixtures() -> dict[str, set[str]]:
    """Each tests/conftest.py fixture -> the test files it imports helpers from, followed through the
    fixtures it requests (bank_units requests bank_capture, which builds on test_hybrid2)."""
    fns = {n.name: n for n in ast.walk(ast.parse((TESTS / "conftest.py").read_text()))
           if isinstance(n, ast.FunctionDef) and any("fixture" in ast.unparse(d) for d in n.decorator_list)}
    own = {name: {m.module.split(".")[1] for m in ast.walk(fn)
                  if isinstance(m, ast.ImportFrom) and (m.module or "").startswith("tests.test_")}
           for name, fn in fns.items()}
    asks = {name: {a.arg for a in fn.args.args} & set(fns) for name, fn in fns.items()}
    out = {}
    for name in fns:
        seen, todo, helpers = set(), [name], set()
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            helpers |= own[x]
            todo += list(asks[x])
        out[name] = helpers
    return out


def select(files: list[str]) -> tuple[set[str] | None, list[str] | None, dict[str, list[str]]]:
    """(test files to run or None for all, frontend files for vitest or None for all, why) for the changed files."""
    mods, tests = graph()
    why: dict[str, list[str]] = {}
    changed_mods: set[str] = set()
    direct_tests: set[str] = set()
    run_all = False
    front: list[str] = []
    front_all = False
    for f in files:
        p = Path(f)
        if f in EVERYTHING:
            run_all = True
            why.setdefault("everything", []).append(f)
        elif f in FRONTEND_ALL:
            front_all = True
        elif p.parts[:1] == ("frontend",):
            if p.suffix in (".ts", ".tsx") and p.parts[1:2] == ("src",) and (ROOT / f).exists():
                front.append(str(ROOT / f))
        elif p.parts[:1] == ("backend",) and p.suffix in (".py", ".json") and p.stem in MODULES:
            changed_mods.add(p.stem)
        elif p.parts[:1] == ("backend",):
            m, t = mentions(p.name)
            changed_mods |= m
            direct_tests |= t
        elif p.parts[:1] == ("tests",) and p.stem in tests:
            direct_tests.add(p.stem)
            why.setdefault(p.stem, []).append(f)
        elif p.parts[:1] in (("data",), ("scripts",)):
            m, t = mentions(p.name)
            changed_mods |= m
            direct_tests |= t
            for x in t:
                why.setdefault(x, []).append(f)
    if run_all:
        return None, (None if front_all or front else []), why
    hit = reached(changed_mods, mods)
    selected = set(direct_tests)
    for t, (deps, _) in tests.items():
        reach = deps & hit
        if reach:
            selected.add(t)
            why.setdefault(t, []).extend(sorted(reach))
    # a test file whose helpers changed: the files that import them run too, directly or through a
    # conftest fixture built on them, and so on down the chain (test_mtk -> test_hybrid2 -> bank_units)
    fixtures = conftest_fixtures()
    sources = {t: (TESTS / f"{t}.py").read_text() for t in tests}
    changed_helpers = set(direct_tests)
    while True:
        grown = set()
        for t, (_, helper_of) in tests.items():
            hit = helper_of & changed_helpers
            if hit and t not in changed_helpers:
                grown.add(t)
                why.setdefault(t, []).extend(f"tests/{h}.py" for h in sorted(hit))
        hit_fixtures = {f for f, helpers in fixtures.items() if helpers & changed_helpers}
        for t, src in sources.items():
            used = {f for f in hit_fixtures if f"{f}:" in src or f"{f}," in src or f"{f})" in src}
            if used and t not in changed_helpers:
                grown.add(t)
                why.setdefault(t, []).extend(f"conftest {f}" for f in sorted(used))
        if not grown - changed_helpers:
            break
        changed_helpers |= grown
    selected |= changed_helpers
    return selected, (None if front_all else front), why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="HEAD", help="compare against this ref (default HEAD: uncommitted work)")
    ap.add_argument("--part", choices=sorted(PARTS), help="run one part of the suite")
    ap.add_argument("--all", action="store_true", help="run the whole suite")
    ap.add_argument("--dry-run", action="store_true", help="print the selection only")
    ap.add_argument("--no-frontend", action="store_true", help="skip tsc and vitest")
    ap.add_argument("pytest_args", nargs="*", help="passed to pytest after --")
    a = ap.parse_args()

    if a.all or a.part:
        selected = None if a.all else set(PARTS[a.part])
        front: list[str] | None = None if a.all else []
        why: dict[str, list[str]] = {}
    else:
        files = changed_files(a.base)
        print(f"{len(files)} changed file(s) against {a.base}")
        selected, front, why = select(files)

    if selected is None:
        print("python: the whole suite")
    elif selected:
        by_part: dict[str, list[str]] = {}
        for t in sorted(selected):
            by_part.setdefault(PART_OF.get(t, "?"), []).append(t)
        print(f"python: {len(selected)} test file(s)")
        for part, ts in sorted(by_part.items()):
            for t in ts:
                reason = ", ".join(dict.fromkeys(why.get(t, [])))
                print(f"  [{part}] {t}" + (f"  <- {reason}" if reason else ""))
    else:
        print("python: nothing to run")
    if not a.no_frontend:
        print("frontend: " + ("tsc + every vitest" if front is None else
                              f"tsc + vitest related to {len(front)} file(s)" if front else "nothing to run"))
    if a.dry_run:
        return 0

    rc = 0
    if selected is None or selected:
        targets = [] if selected is None else [str(TESTS / f"{t}.py") for t in sorted(selected)]
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets, *a.pytest_args]
        rc |= subprocess.call(cmd, cwd=ROOT)
    if not a.no_frontend and (front is None or front):
        npx = ["npx"]
        rc |= subprocess.call(npx + ["tsc", "-b"], cwd=FRONTEND)
        vit = npx + ["vitest", "run"] if front is None else npx + ["vitest", "related", "--run", *front]
        rc |= subprocess.call(vit, cwd=FRONTEND)
    return rc


if __name__ == "__main__":
    sys.exit(main())
