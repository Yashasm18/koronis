"""The README's badges are claims, so they are checked like any other claim.

A hand-maintained "105 tests passing" badge is a number that silently rots the
moment a test is added. This project's whole discipline is that a property you
assert gets checked and a property you merely intend gets violated, so the
badge is asserted here rather than trusted.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _collected_test_count() -> tuple[int, str]:
    """Ask pytest itself, in a subprocess, so this cannot count itself wrong.

    Also reports anything skipped at COLLECTION time. A module guarded by
    `importorskip` contributes no tests when its dependency is absent, so the
    count legitimately differs between a machine with the optional test
    dependencies and one without - which is exactly how this check once failed
    in CI while passing locally. requirements-dev.txt exists to keep the two
    environments the same; the note below says so when they diverge.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, f"could not read a collection count from pytest:\n{out[-500:]}"
    skipped = re.search(r"(\d+) skipped", out)
    note = ""
    if skipped:
        note = (f" ({skipped.group(1)} module(s) skipped at collection - install "
                f"requirements-dev.txt so optional tests are collected)")
    return int(m.group(1)), note


def test_tests_badge_matches_the_suite():
    m = re.search(r"badge/tests-(\d+)%20passing", README)
    assert m, "the tests badge is missing from the README"
    claimed = int(m.group(1))
    actual, note = _collected_test_count()
    assert claimed == actual, (
        f"README badge claims {claimed} tests, pytest collects {actual}{note}. "
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
