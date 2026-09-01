"""The built demo page must actually run.

A renamed column in `results/` once left the site's table code looking up a
key that no longer existed. It threw, which killed the whole page script, and
the live demo silently stopped replaying - while every table in the docs stayed
correct and every other test passed. The site is generated from the same
artifacts as the docs, so it can go stale in ways prose cannot, and it is the
first thing anyone actually looks at.

Skipped when Playwright or its browser is unavailable, so the suite still runs
on a machine that has not installed them.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import Error as PWError, sync_playwright  # noqa: E402


@pytest.fixture(scope="module")
def page():
    if not PAGE.exists():
        pytest.skip("docs/index.html not built")
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except PWError as e:
                pytest.skip(f"chromium unavailable: {e}")
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
            pg = ctx.new_page()
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console",
                  lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
            pg.goto(PAGE.as_uri())
            pg.wait_for_timeout(1200)
            yield pg, errors
            ctx.close(); browser.close()
    except PWError as e:                       # pragma: no cover
        pytest.skip(f"playwright unusable: {e}")


def test_page_loads_without_script_errors(page):
    pg, errors = page
    assert not errors, f"the demo page raised: {errors}"


def test_every_table_has_rows(page):
    """An empty table means its result file moved or a column was renamed."""
    pg, _ = page
    pg.click('.tabs button[data-tab="eval"]')
    pg.wait_for_timeout(400)
    for tid in ("t-seeds", "t-mech", "t-policy", "t-front",
                "t-bench", "t-drift", "t-arch", "t-ap"):
        n = pg.eval_on_selector_all(f"#{tid} tr", "els => els.length")
        assert n >= 2, f"#{tid} rendered {n} rows - its data key probably moved"


def test_no_undefined_or_nan_reaches_the_page(page):
    pg, _ = page
    pg.click('.tabs button[data-tab="eval"]')
    pg.wait_for_timeout(300)
    text = pg.eval_on_selector("body", "e => e.innerText")
    for bad in ("undefined", "NaN", "[object", "Infinity"):
        assert bad not in text, f"{bad!r} is visible on the page"


def test_the_replay_is_watchable_not_instant(page):
    """It used to finish the whole stream in under three seconds, which read as
    a dead button. Paced against the wall clock now, so it is also independent
    of the display's refresh rate."""
    pg, _ = page
    pg.click('.tabs button[data-tab="replay"]')
    pg.click("#reset"); pg.wait_for_timeout(200)
    pg.click('.speed button[data-speed="1"]')
    pg.click("#play"); pg.wait_for_timeout(3000)
    seen = int(pg.eval_on_selector("#m-seen", "e => e.textContent.replace(/[^0-9]/g,'')") or 0)
    total = pg.evaluate("JSON.parse(document.getElementById('koronis-data').textContent)"
                        ".replay.events.length")
    assert 0 < seen < total * 0.5, (
        f"after 3 s the replay is at {seen}/{total}; a full pass should take tens of "
        f"seconds at 1x, not finish immediately")
    pg.click("#reset")


def test_speed_control_changes_the_pace(page):
    pg, _ = page
    pg.click('.tabs button[data-tab="replay"]')
    def run(speed, ms):
        pg.click("#reset"); pg.wait_for_timeout(150)
        pg.click(f'.speed button[data-speed="{speed}"]')
        pg.click("#play"); pg.wait_for_timeout(ms)
        n = int(pg.eval_on_selector("#m-seen", "e => e.textContent.replace(/[^0-9]/g,'')") or 0)
        pg.click("#reset")
        return n
    slow, fast = run(1, 1500), run(16, 1500)
    assert fast > slow, f"16x ({fast}) did not outpace 1x ({slow})"


def test_the_replay_actually_advances(page):
    """The failure this file exists for: the page rendered, and did nothing."""
    pg, errors = page
    pg.click('.tabs button[data-tab="replay"]')
    pg.wait_for_timeout(300)
    pg.click("#play")
    pg.wait_for_timeout(2500)
    seen = int(pg.eval_on_selector("#m-seen", "e => e.textContent.replace(/[^0-9]/g,'')") or 0)
    assert seen > 0, f"replay did not advance; page errors: {errors}"
