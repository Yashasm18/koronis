"""Assemble the demo site from the experiment's own result artifacts.

The site is a *client* for the experiment, not a second simulation. Every
number it shows is read from results/*.json|csv at build time and embedded
verbatim; nothing is transcribed by hand and nothing is recomputed. If an
experiment is re-run, rebuilding the site is the only step needed to update it.

    python site/build.py      ->  docs/index.html
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# GitHub Pages serves /docs on the default branch, so the built page lands
# there and gets a public URL for free. site/ holds the source: template
# plus this script.
OUT = ROOT / "docs" / "index.html"


def _bundle() -> dict:
    return {
        "replay": json.loads((RESULTS / "replay_demo.json").read_text()),
        "bench": json.loads((RESULTS / "benchmark.json").read_text()),
        "policy": json.loads((RESULTS / "policy.json").read_text()),
        "drift": json.loads((RESULTS / "drift.json").read_text()),
        "seeds": list(csv.DictReader((RESULTS / "seeds_summary.csv").open())),
        "mech": list(csv.DictReader((RESULTS / "mechanism.csv").open())),
        "frontier": list(csv.DictReader((RESULTS / "frontier.csv").open())),
        "aperture": list(csv.DictReader((RESULTS / "aperture.csv").open())),
        "arch": list(csv.DictReader((RESULTS / "architecture.csv").open())),
        "latency": list(csv.DictReader((RESULTS / "latency.csv").open())),
    }


def main() -> None:
    # the README needs a static copy of the frontier chart; GitHub cannot run
    # the canvas the site uses, so both are emitted from the same artifact
    import frontier_svg
    frontier_svg.main()

    data = json.dumps(_bundle(), separators=(",", ":"))
    html = (ROOT / "site" / "template.html").read_text()
    OUT.write_text(html.replace("/*__KORONIS_DATA__*/null", data))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
