"""The README's badges are claims, so they are checked like any other claim.

A hand-maintained "105 tests passing" badge is a number that silently rots the
moment a test is added. This project's whole discipline is that a property you
assert gets checked and a property you merely intend gets violated, so the
badge is asserted here rather than trusted.
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _missing_optional_test_deps() -> list[str]:
    """Which requirements-dev.txt packages are absent from this interpreter.

    The badge counts the whole suite, and the whole suite exists only when the
    optional test dependencies are installed: a module guarded by
    `importorskip` contributes no tests without them.

    This asks the interpreter directly rather than reading pytest's summary.
    The previous version parsed `--collect-only` output for a "skipped" count,
    which that mode never prints -- so the branch it guarded could not fire, and
    a clean checkout got a failed badge check instead of the explanation the
    docstring promised. Distribution name and import name coincide for every
    entry here; a future one where they differ needs a mapping, and will show up
    as a permanent skip rather than silently passing.
    """
    req = ROOT / "requirements-dev.txt"
    names = [re.split(r"[=<>!~\[]", ln, maxsplit=1)[0].strip()
             for ln in req.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    assert names, "requirements-dev.txt lists no packages"
    return [n for n in names if importlib.util.find_spec(n) is None]


def _collected_test_count() -> int:
    """Ask pytest itself, in a subprocess, so this cannot count itself wrong."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, f"could not read a collection count from pytest:\n{out[-500:]}"
    return int(m.group(1))


def test_tests_badge_matches_the_suite():
    """The badge counts the whole suite, so only a whole suite can check it.

    Without the optional test dependencies the `importorskip` modules are never
    collected and the count is legitimately lower. Failing here would hand
    anyone who cloned the repository and installed only requirements.txt a red
    suite for a badge that is perfectly correct -- a false alarm at exactly the
    moment a stranger forms their first impression. So that case skips, and CI,
    which installs requirements-dev.txt, is where a stale badge is caught. A
    guard that cries wolf on a clean checkout teaches people to ignore it, which
    costs more than the badge is worth.
    """
    m = re.search(r"badge/tests-(\d+)%20passing", README)
    assert m, "the tests badge is missing from the README"
    claimed = int(m.group(1))
    missing = _missing_optional_test_deps()
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed, so part of the suite is "
                    f"not collected here; the badge claims {claimed} and is "
                    f"checked in CI, which installs requirements-dev.txt")
    actual = _collected_test_count()
    assert claimed == actual, (
        f"README badge claims {claimed} tests, pytest collects {actual}. "
        f"Update the badge in README.md.")


def test_python_badge_matches_the_ci_matrix():
    badge = re.search(r"badge/python-([\d.]+)-", README)
    assert badge, "the python badge is missing from the README"
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    versions = set(re.findall(r'python-version:\s*"([\d.]+)"', ci))
    assert badge.group(1) in versions, (
        f"README claims python {badge.group(1)}, CI runs {sorted(versions)}")


def test_license_badge_matches_the_license_file():
    assert re.search(r"badge/license-MIT-", README), "license badge missing"
    assert "MIT License" in (ROOT / "LICENSE").read_text()


def test_every_badge_target_exists():
    """A badge that links nowhere is worse than no badge."""
    for label, target in re.findall(r"\[!\[([^\]]*)\]\([^)]+\)\]\(([^)]+)\)", README):
        if target.startswith("http"):
            continue
        assert (ROOT / target.split("#")[0]).exists(), \
            f"badge {label!r} links to {target}, which does not exist"
