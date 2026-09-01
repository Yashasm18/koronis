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


def _play_to_end(pg, speed):
    pg.click('.tabs button[data-tab="replay"]')
    pg.click("#reset"); pg.wait_for_timeout(200)
    pg.click(f'.speed button[data-speed="{speed}"]')
    pg.click("#play")
    for _ in range(900):
        done = pg.eval_on_selector("#play", "e => e.textContent") == "Replay incident"
        seen = int(pg.eval_on_selector("#m-seen", "e => e.textContent.replace(/[^0-9]/g,'')") or 0)
        if done and seen:
            return seen
        pg.wait_for_timeout(100)
    raise AssertionError(f"replay did not finish at {speed}x")


def _canvas_ink(pg, cid):
    return pg.evaluate(f"""() => {{const c = document.getElementById('{cid}');
        const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n;}}""")


def test_the_replay_ends_identically_at_every_speed(page):
    """Playback speed is a viewing choice, not an input to the result.

    The evidence graph used to seed each node's position from its index inside
    a 160-event display window, which depends on how many events arrive per
    frame - so the same replay drew a visibly different graph at 1x and 16x
    while every number agreed. A viewer who changed speed saw a different
    picture of the same data.
    """
    pg, _ = page
    fields = ["m-seen", "m-score", "m-alerts", "m-clock", "incbar", "fcast",
              "card-title", "card-score", "dossier"]
    base = None
    for speed in (1, 4, 16):
        _play_to_end(pg, speed)
        pg.eval_on_selector("#dossier-wrap", "e => e.open = true")
        pg.wait_for_timeout(150)
        state = {f: pg.eval_on_selector(f"#{f}", "e => e.textContent.trim()") for f in fields}
        state["graph_ink"] = _canvas_ink(pg, "graph")
        state["chart_ink"] = _canvas_ink(pg, "chart")
        if base is None:
            base, base_speed = state, speed
            continue
        differing = [k for k in base if base[k] != state[k]]
        assert not differing, (
            f"{speed}x ends in a different state from {base_speed}x: "
            + "; ".join(f"{k}: {base[k]!r} vs {state[k]!r}" for k in differing))
    pg.click("#reset")


# The threshold label was drawn first, five pixels above the rule. The threshold
# is 0.9366 and alerting events saturate at 1.0, so that gap is the densest band
# in the chart: the dots landed on top of the words and the rule's own dashes
# struck through them. Nothing failed - the label was *there*, just unreadable.
_LABEL_GEOMETRY = """() => {
  const c = document.getElementById('chart'), x = c.getContext('2d');
  const W = c.width, H = c.height, d = x.getImageData(0, 0, W, H).data;
  const isRed  = i => d[i+3] > 128 && d[i] > 150 && d[i+1] < 120 && d[i+2] < 120;
  const isBlue = i => d[i+3] > 128 && d[i+2] > 150 && d[i] < 120;

  const rowRed = new Array(H).fill(0);
  for (let y = 0; y < H; y++)
    for (let xx = 0; xx < W; xx++) if (isRed((y*W + xx)*4)) rowRed[y]++;
  const ruleY = rowRed.indexOf(Math.max(...rowRed));   // the dashed rule

  let n = 0, minY = 1e9, maxY = -1, minX = 1e9, maxX = -1;
  for (let y = 0; y < H; y++) {
    if (Math.abs(y - ruleY) <= 3) continue;            // skip the rule itself
    for (let xx = 0; xx < W; xx++) if (isRed((y*W + xx)*4)) {
      n++;
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      minX = Math.min(minX, xx); maxX = Math.max(maxX, xx);
    }
  }
  let blue = 0;
  if (n) for (let y = minY; y <= maxY; y++)
    for (let xx = minX; xx <= maxX; xx++) if (isBlue((y*W + xx)*4)) blue++;
  return {ruleY, labelPx: n, labelTop: minY, labelBottom: maxY, blueOverLabel: blue};
}"""


def test_the_frozen_threshold_label_is_legible(page):
    pg, _ = page
    pg.click('.speed button[data-speed="16"]')
    pg.click("#play")
    pg.wait_for_function(
        "() => document.getElementById('play').textContent === 'Replay incident'",
        timeout=60_000)
    pg.wait_for_timeout(400)
    g = pg.evaluate(_LABEL_GEOMETRY)

    assert g["labelPx"] > 80, f"the threshold label is barely drawn at all: {g}"
    assert g["labelTop"] > g["ruleY"], (
        "the threshold label sits above the rule, in the band where alerting "
        f"scores saturate at 1.0: {g}")
    assert g["blueOverLabel"] == 0, (
        f"{g['blueOverLabel']} score points are drawn on top of the threshold "
        f"label, which is what makes it unreadable: {g}")

    pg.click("#reset")
