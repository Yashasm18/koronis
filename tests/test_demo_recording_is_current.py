"""The README's screen recording must show the results the repo currently reports.

Defect 14: the demo GIF sat at the top of the README for four commits after the
experiments behind it were re-run. It showed an action ladder of Rs 21,973 /
10,008 / 3,646 / 1,667 while results/policy.json said 25,331 / 11,519 / 4,150 /
1,768, and a replay panel with no speed control, because the control was added
after the recording was made. Every number in the README was checked by a test;
the picture at the top of it was checked by nobody.

The recording is a binary, so this cannot compare it to results directly. It
compares a stamp committed beside it against the two things a viewer can read
off the frames: the numbers, and the controls that are visible to click.
"""
import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAMP = ROOT / "docs" / "assets" / "koronis-demo.stamp"
INDEX = ROOT / "docs" / "index.html"
RESULTS = ROOT / "results"

REMAKE = (
    "Re-record it:  python site/build.py && python site/record_demo.py"
    "  - which refreshes this stamp as its last step."
)


def _results_digest() -> str:
    """Hash the experiment outputs the recording puts on screen."""
    h = hashlib.sha256()
    for p in sorted(RESULTS.glob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _controls_digest() -> str:
    """Hash the page's control surface: ids and the tab/speed switches.

    Deliberately not the whole page. Copy edits and styling should not force a
    5 MB re-record; a control appearing or disappearing should, because the
    recording then shows a console the reader cannot find on the live site.
    """
    html = INDEX.read_text()
    tokens = sorted(set(re.findall(r'(?:id|data-tab|data-speed)="([^"]+)"', html)))
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def _current() -> dict[str, str]:
    return {"results": _results_digest(), "controls": _controls_digest()}


def _read_stamp() -> dict[str, str]:
    out = {}
    for line in STAMP.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split()
            out[k] = v
    return out


def test_the_recording_shows_the_results_the_repo_reports():
    assert STAMP.exists(), f"{STAMP.name} is missing. {REMAKE}"
    stamp, now = _read_stamp(), _current()

    if stamp.get("results") != now["results"]:
        raise AssertionError(
            "results/ has changed since the demo recording was made, so the "
            "figures in the GIF at the top of the README no longer match the "
            f"figures in its text. {REMAKE}"
        )
    if stamp.get("controls") != now["controls"]:
        raise AssertionError(
            "the demo page's controls have changed since the recording was "
            "made, so the GIF shows a console that differs from the live one. "
            f"{REMAKE}"
        )


def test_the_recording_and_its_poster_are_committed():
    for name in ("koronis-demo.gif", "koronis-demo.mp4", "koronis-demo-poster.png"):
        p = ROOT / "docs" / "assets" / name
        assert p.exists() and p.stat().st_size > 10_000, f"{name} is missing or truncated"


def write_stamp() -> None:
    """Record what the current recording shows. Called by site/record_demo.py."""
    now = _current()
    STAMP.write_text(
        "# Digests of what the README's screen recording shows. Written by\n"
        "#   python site/record_demo.py\n"
        "# Do not hand-edit: refreshing it without remaking the recording is\n"
        "# exactly the check being defeated.\n"
        f"results {now['results']}\n"
        f"controls {now['controls']}\n"
    )
