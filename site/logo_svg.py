"""The Koronis mark, generated rather than drawn.

The name is the argument. The Koronis asteroid family was identified by noticing
that objects scattered across the sky share one origin -- the cluster is only
visible once you stop looking at each rock on its own. That is this project:
attempts that are unremarkable individually, and a campaign once linked.

So the mark is the algorithm. Twelve entities sit on a corona. Five of them are
one incident, joined by a spanning tree rather than a clique, because that is
what the online union-find actually builds. One node is coral: the seed the
consolidation started from. The other seven stay dim -- legitimate traffic, and
most of the ring, which is also true.

    python -m site.logo_svg      # rewrites docs/assets/koronis-mark.svg
"""
from __future__ import annotations

import math
from pathlib import Path

# The site's own tokens, so the mark cannot drift from the app it belongs to.
PLATE, PLATE_EDGE = "#0E141C", "#26303D"
CORONA, DIM = "#293442", "#3A4759"
ACCENT, SEED = "#5B94FF", "#FF6B60"

N_RING = 12
CAMPAIGN = (0, 3, 5, 8, 10)          # spread, not adjacent: that is the point
SEED_NODE = 0
EDGES = ((10, 3), (3, 5), (5, 0), (0, 8))   # a spanning tree, not a clique or a star

C, R_RING = 60.0, 41.0
ASSET = Path(__file__).resolve().parent.parent / "docs" / "assets" / "koronis-mark.svg"


def _pos(i: int) -> tuple[float, float]:
    a = math.radians(-90 + i * (360 / N_RING))
    return round(C + R_RING * math.cos(a), 2), round(C + R_RING * math.sin(a), 2)


def mark(size: int = 120, pad_top: float = 0.0) -> str:
    """The mark as SVG. `pad_top` lifts the plate down the box, for baseline alignment."""
    o, box = pad_top, 120 + pad_top
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 {box:g}"'
         f' width="{size}" height="{size * box / 120:.0f}" role="img"'
         ' aria-label="Koronis"><title>Koronis</title>',
         f'<rect x="1.5" y="{1.5 + o:g}" width="117" height="117" rx="28"'
         f' fill="{PLATE}" stroke="{PLATE_EDGE}" stroke-width="1.5"/>',
         f'<circle cx="{C:g}" cy="{C + o:g}" r="{R_RING:g}" fill="none"'
         f' stroke="{CORONA}" stroke-width="1.6"/>']

    for a, b in EDGES:                                    # edges under the nodes
        (x1, y1), (x2, y2) = _pos(a), _pos(b)
        p.append(f'<path d="M{x1:g} {y1 + o:g} L{x2:g} {y2 + o:g}" stroke="{ACCENT}"'
                 ' stroke-width="2.6" stroke-linecap="round" opacity=".5"/>')

    for i in range(N_RING):
        x, y = _pos(i)
        y += o
        if i not in CAMPAIGN:
            p.append(f'<circle cx="{x:g}" cy="{y:g}" r="3.1" fill="{DIM}"/>')
            continue
        fill = SEED if i == SEED_NODE else ACCENT
        p.append(f'<circle cx="{x:g}" cy="{y:g}" r="11" fill="{fill}" opacity=".14"/>'
                 f'<circle cx="{x:g}" cy="{y:g}" r="6.4" fill="{fill}"/>')
    p.append("</svg>")
    return "".join(p)


def favicon() -> str:
    """The same five nodes at 32 px, with the corona dots dropped.

    A favicon is read at 16 px in a tab strip, where the dim ring turns to mud
    and takes the crown down with it. Everything that survives that size is
    load-bearing, so the ring becomes a hairline and only the incident is drawn.
    """
    k, off = 13.0 / R_RING, 16.0 - C * (13.0 / R_RING)   # ring r=41 -> 13, recentred

    def q(i: int) -> tuple[float, float]:
        x, y = _pos(i)
        return round(x * k + off, 2), round(y * k + off, 2)

    p = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">',
         f'<rect width="32" height="32" rx="7.5" fill="{PLATE}"/>',
         f'<circle cx="16" cy="16" r="13" fill="none" stroke="{CORONA}" stroke-width="1"/>']
    for a, b in EDGES:
        (x1, y1), (x2, y2) = q(a), q(b)
        p.append(f'<path d="M{x1:g} {y1:g} L{x2:g} {y2:g}" stroke="{ACCENT}"'
                 ' stroke-width="1.7" stroke-linecap="round" opacity=".75"/>')
    for i in CAMPAIGN:
        x, y = q(i)
        fill = SEED if i == SEED_NODE else ACCENT
        p.append(f'<circle cx="{x:g}" cy="{y:g}" r="2.7" fill="{fill}"/>')
    p.append("</svg>")
    return "".join(p)


def favicon_data_uri() -> str:
    """Percent-encoded for an inline `href`, so the page stays a single file."""
    out = favicon()
    for ch, esc in (("%", "%25"), ("<", "%3C"), (">", "%3E"), ("#", "%23"), ('"', "'")):
        out = out.replace(ch, esc)
    return "data:image/svg+xml," + out


if __name__ == "__main__":
    ASSET.write_text(mark(pad_top=0.0) + "\n", encoding="utf-8")
    print(f"wrote {ASSET} ({ASSET.stat().st_size} bytes)")
    print(favicon_data_uri())
