"""The mark is code, and the three places it appears must agree with that code.

An identity drifts the way a figure drifts. The favicon is a percent-encoded data
URI inlined in the template, the header mark is inlined SVG, and the README points
at a committed file -- three copies of one drawing, none of which a human would
notice going stale. `site/logo_svg.py` is the single source; this asserts the
copies are what it currently emits, so editing the mark by hand fails here rather
than shipping two Koronises.

The last check is the defence-only stance applied to an asset: SVG is a document
format with a script element and remote references, and an icon that can fetch is
a network capability that `test_defence_only.py` would never see, because it walks
Python.
"""
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("koronis_logo", ROOT / "site" / "logo_svg.py")
logo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logo)

MARK = ROOT / "docs" / "assets" / "koronis-mark.svg"
TEMPLATE = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_committed_mark_is_what_the_generator_emits():
    assert MARK.read_text(encoding="utf-8").strip() == logo.mark().strip()


def test_the_site_favicon_is_what_the_generator_emits():
    href = re.search(r'<link rel="icon" href="([^"]*)">', TEMPLATE)
    assert href, "the template has no favicon"
    assert href.group(1) == logo.favicon_data_uri()


def test_the_site_header_carries_the_same_mark():
    assert logo.mark(size=34).replace('role="img"', 'class="mark" role="img"') in TEMPLATE


def test_the_readme_shows_the_mark_beside_the_title():
    first = README.splitlines()[0]
    assert "docs/assets/koronis-mark.svg" in first and "Koronis</h1>" in first, first
    assert MARK.exists()


@pytest.mark.parametrize("svg", [logo.mark(), logo.favicon()])
def test_the_mark_cannot_fetch_or_execute(svg):
    lowered = svg.lower()
    for capability in ("<script", "<foreignobject", "<use", "<image",
                       "xlink:href", "href=", "url(", "javascript:", "onload"):
        assert capability not in lowered, f"the mark carries {capability!r}"
