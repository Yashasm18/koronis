"""Render results/frontier.csv as a static SVG for the README.

The demo site draws this chart on a canvas; GitHub cannot run that, so the
same data is emitted as an SVG here. Generated, never hand-edited, from the
same artifact — so it cannot drift out of step with the experiment.

    python site/frontier_svg.py   ->  docs/assets/frontier.svg
"""
import csv
from math import log
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "frontier.svg"

W, H = 760, 300
P = dict(l=54, r=132, t=22, b=40)
PW, PH = W - P["l"] - P["r"], H - P["t"] - P["b"]
NS, KS = (130, 2400), (1.35, 330)
# Chosen to read on GitHub's light and dark canvases alike.
MUTE, CRIT, ACC, OK = "#8A94A6", "#D2402F", "#3B7DF0", "#15A06B"
MONO = "font-family='ui-monospace,SFMono-Regular,Menlo,monospace'"


def _x(n): return P["l"] + (log(n) - log(NS[0])) / (log(NS[1]) - log(NS[0])) * PW
def _y(k): return P["t"] + PH - (log(k) - log(KS[0])) / (log(KS[1]) - log(KS[0])) * PH


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results" / "frontier.csv").open()))
    tau = int(rows[0]["tau_binding"])
    ns = sorted({int(r["n"]) for r in rows})
    ks = sorted({int(r["k"]) for r in rows})
    o = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' "
         f"height='{H}' role='img' aria-label='Detectability frontier: 16 measured "
         f"cells against the predicted boundary k = n over tau'>",
         f"<rect width='{W}' height='{H}' fill='none'/>"]

    # blind region + boundary
    y0, y1 = _y(NS[0] / tau), _y(NS[1] / tau)
    o.append(f"<path d='M{_x(NS[0]):.1f},{y0:.1f} L{_x(NS[1]):.1f},{y1:.1f} "
             f"L{_x(NS[1]):.1f},{P['t']} L{_x(NS[0]):.1f},{P['t']} Z' "
             f"fill='{CRIT}' fill-opacity='.09'/>")
    o.append(f"<line x1='{_x(NS[0]):.1f}' y1='{y0:.1f}' x2='{_x(NS[1]):.1f}' "
             f"y2='{y1:.1f}' stroke='{CRIT}' stroke-width='1.6' stroke-dasharray='6 4'/>")

    # axes
    o.append(f"<path d='M{P['l']},{P['t']} L{P['l']},{P['t']+PH} L{P['l']+PW},"
             f"{P['t']+PH}' fill='none' stroke='{MUTE}' stroke-opacity='.5'/>")
    for k in ks:
        o.append(f"<text x='{P['l']-8}' y='{_y(k)+3.5:.1f}' {MONO} font-size='10' "
                 f"fill='{MUTE}' text-anchor='end'>k={k}</text>")
    for n in ns:
        o.append(f"<text x='{_x(n):.1f}' y='{P['t']+PH+16}' {MONO} font-size='10' "
                 f"fill='{MUTE}' text-anchor='middle'>{n}</text>")
    o.append(f"<text x='{P['l']+PW/2:.1f}' y='{P['t']+PH+32}' {MONO} font-size='10' "
             f"fill='{MUTE}' text-anchor='middle'>attempts n</text>")
    # region labels sit in the vertical gaps between point rows, never on one
    blind_y = (_y(ks[-1]) + _y(ks[-2])) / 2
    fires_y = (_y(ks[0]) + _y(ks[1])) / 2
    o.append(f"<text x='{P['l']+6}' y='{blind_y:.1f}' {MONO} font-size='10' "
             f"fill='{MUTE}'>velocity blind — no counter can trip</text>")
    o.append(f"<text x='{P['l']+6}' y='{fires_y:.1f}' {MONO} font-size='10' "
             f"fill='{MUTE}'>velocity fires</text>")

    # measured cells
    for r in rows:
        n, k = int(r["n"]), int(r["k"])
        fires = r["velocity_detected"] == "True"
        pred_blind = r["velocity_blind_predicted"] == "True"
        x, y = _x(n), _y(k)
        col = MUTE if fires else CRIT
        o.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='9' fill='{col}' "
                 f"fill-opacity='.12' stroke='{col}' stroke-width='1.4'/>")
        if r["koronis_detected"] == "True":
            o.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.4' fill='{ACC}'/>")
        if fires != pred_blind:            # measured agrees with the prediction
            o.append(f"<path d='M{x-2.5:.1f},{y+11.5:.1f} l2,2 l3.5,-4' fill='none' "
                     f"stroke='{OK}' stroke-width='1.6' stroke-linecap='round'/>")

    # legend
    lx, ly = P["l"] + PW + 16, P["t"] + PH / 2 - 30
    o.append(f"<line x1='{lx}' y1='{ly-16}' x2='{lx+9}' y2='{ly-16}' stroke='{CRIT}' "
             f"stroke-width='1.6' stroke-dasharray='4 3'/>")
    o.append(f"<text x='{lx+13}' y='{ly-13}' {MONO} font-size='9.5' fill='{CRIT}'>"
             f"k = n / τ,  τ = {tau}</text>")
    o.append(f"<circle cx='{lx+4}' cy='{ly}' r='3.4' fill='{ACC}'/>")
    o.append(f"<text x='{lx+13}' y='{ly+3}' {MONO} font-size='9.5' fill='{MUTE}'>"
             f"koronis detects</text>")
    o.append(f"<path d='M{lx+1},{ly+17} l2,2 l4,-5' fill='none' stroke='{OK}' "
             f"stroke-width='1.6' stroke-linecap='round'/>")
    o.append(f"<text x='{lx+13}' y='{ly+21}' {MONO} font-size='9.5' fill='{MUTE}'>"
             f"matches</text>")
    o.append(f"<text x='{lx+13}' y='{ly+32}' {MONO} font-size='9.5' fill='{MUTE}'>"
             f"prediction</text>")
    o.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(o))
    agree = sum((r["velocity_detected"] == "True") != (r["velocity_blind_predicted"] == "True")
                for r in rows)
    print(f"wrote {OUT}  ({agree}/{len(rows)} cells match the prediction)")


if __name__ == "__main__":
    main()
