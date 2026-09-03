"""Every figure in the prose must still exist in `results/`.

There are two other figure guards and neither covers this. `test_docs_match_results.py`
checks that each computed figure *appears* in the docs, which can only detect a number that
has gone missing. `test_doc_tables_match_results.py` re-derives whole tables, which covers
the tables and nothing around them.

Defect 19 lived in exactly that gap: the paragraph under the mechanism table cited
0.452 precision, 464 false positives, recall 0.695, 53 false positives, 0.840 precision and
a 31x reduction, three lines beneath a table reading 0.451, 466, 0.812, 35, 0.903 and 10.
Six wrong figures in one paragraph, in prose no table guard looks at.

So this reads the prose. Any figure-shaped token - a three or four decimal number, a rupee
amount, a digit-grouped integer - has to be a number some artifact actually contains. The
allowlist below is for figures that are deliberately superseded: a defect narrative
describing what a number *used to be* is the one place a stale figure belongs.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# The engineering log's whole purpose is recording what results used to say, so
# every figure in it is potentially historical. It is excluded wholesale rather
# than allowlisted line by line.
SKIP = {"engineering-log.md"}

DOCS = ["README.md", "SECURITY.md", "CONTRIBUTING.md"] + [
    str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs").glob("*.md"))
    if p.name not in SKIP
]

#: Figures that are deliberately not current, with the reason each is kept.
HISTORICAL = {
    ("docs/evaluation.md", "0.015"):
        "the count-min sketch defect: the share a bridged domain measured before "
        "per-relation sketches, against a 0.02 cap",
    ("docs/evaluation.md", "3,120"):
        "cache rows retained for a 60 s window before eviction existed - the "
        "measurement that defined the defect, and not reproducible from results/",
}

#: Figures computed from a `*_raw.csv`, which the value set excludes for density.
#: Each is checked here directly instead, so the exemption is not a free pass.
DERIVED = {
    ("docs/evaluation.md", "0.0020"): (
        "median event-model ECE over the 10 seeds in seeds_raw.csv; ECE is not "
        "carried into seeds_summary.csv"),
    ("README.md", "1,760"): "false-positive cost: 44 x COST_PER_FALSE_BLOCK_INR",
    ("README.md", "1,600"): "false-positive cost: 40 x COST_PER_FALSE_BLOCK_INR",
    ("README.md", "19,040"): "false-positive cost: 476 x COST_PER_FALSE_BLOCK_INR",
    ("README.md", "400"): "false-positive cost: 10 x COST_PER_FALSE_BLOCK_INR",
    ("docs/evaluation.md", "0.0090"): (
        "the gap between the two rows of feature_parity.csv - a difference, so "
        "it appears in no artifact by itself"),
}


def _check_derived():
    """Recompute each DERIVED figure so the allowlist cannot rot."""
    import csv
    import statistics

    rows = [r for r in csv.DictReader((ROOT / "results" / "seeds_raw.csv").open())
            if r["detector"] == "koronis_graph" and r.get("ece")]
    ece = statistics.median(float(r["ece"]) for r in rows)

    parity = {r["baseline_free_mail_list"]: float(r["pr_auc"])
              for r in csv.DictReader((ROOT / "results" / "feature_parity.csv").open())}
    gap = (parity["as_shipped_gmail_yahoo_outlook"]
           - parity["matched_to_detector_gmail_outlook"])

    # The false-positive cost column is count x the declared per-block cost, on
    # the counts as the table rounds them. Recomputed so the column cannot drift
    # from either the medians or the constant.
    from koronis.eval.cost import COST_PER_FALSE_BLOCK_INR

    seeds = {r["detector"]: r
             for r in csv.DictReader((ROOT / "results" / "seeds_summary.csv").open())}
    def fp_cost(detector):
        fp = round(float(seeds[detector]["false_positives_median"]))
        return f"{fp * COST_PER_FALSE_BLOCK_INR:,.0f}"

    out = {
        ("docs/evaluation.md", "0.0020"): f"{ece:.4f}" == "0.0020",
        ("docs/evaluation.md", "0.0090"): f"{gap:.4f}" == "0.0090",
        ("README.md", "1,760"): fp_cost("velocity_tuned") == "1,760",
        ("README.md", "1,600"): fp_cost("shared_entity") == "1,600",
        ("README.md", "19,040"): fp_cost("gbdt_per_txn") == "19,040",
        ("README.md", "400"): fp_cost("koronis_graph") == "400",
    }
    return out


PATTERNS = [
    # a grouped integer, but not the leading part of a decimal like 1,859.5
    (r"\b(\d{1,3},\d{3})(?!\.\d)\b", "grouped integer"),
    (r"₹([\d,]{4,})", "rupee amount"),
    (r"\b(0\.\d{3,4})\b", "decimal"),
    # One-decimal percentages are deliberately NOT checked. They are backed -
    # 95.3% is coverage_upper 0.9529, 33.3% is base_false_flag_rate 0.3333 - but
    # verifying them means expanding every fraction in results/ to a one-decimal
    # percentage, and at that density 63.8% of randomly generated percentages
    # passed. A check that accepts two wrong values in three is worse than none,
    # because it reads as coverage. They are covered by the table guard instead,
    # where a figure is bound to the artifact it came from.
]


def _artifact_values() -> set[str]:
    """Every number in results/, in the shapes the docs write them.

    Two deliberate narrowings, both measured. `*_raw.csv` holds per-trial values -
    thousands of them - while the docs publish medians, so including them inflated
    the accept set to 88,000 entries and made the check nearly vacuous: 30.8% of
    randomly generated three-decimal values passed. And expanding every fraction
    to a one-decimal percentage meant any four-decimal PR-AUC covered some
    percentage, at which point 63.8% of random percentages passed.

    With both removed the false-pass rates are reported by
    `test_the_guard_is_strong_enough_to_be_worth_running`, which fails if they
    drift back up. A guard whose strength is not measured is a guard that can
    quietly stop working - which is the defect class this file exists for.
    """
    out: set[str] = set()
    for path in sorted((ROOT / "results").glob("*")):
        if not path.is_file() or path.name.endswith("_raw.csv"):
            continue
        for tok in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", path.read_text(errors="ignore")):
            try:
                value = abs(float(tok))          # sign is the doc's to choose
            except ValueError:
                continue
            out.add(f"{round(value):,}")
            out.add(str(round(value)))
            for places in (0, 1, 2, 3, 4):
                out.add(f"{value:.{places}f}")
            out.add(f"{value * 100:.0f}")        # shares written as whole percentages
    return out


VALUES = _artifact_values()


def _figures(text: str):
    # A fence holding a COMMAND is not a claim; a fence holding program OUTPUT
    # is. Stripping both let a stale audit dossier sit in docs/evaluation.md
    # with six superseded figures, presented as what koronis.incident.dossier()
    # prints - the one place in the repo a reader would most expect to be
    # looking at real output (defect 27).
    text = re.sub(r"```(?:bash|sh|console|python|py|json|mermaid)\b.*?```", "",
                  text, flags=re.S)
    text = re.sub(r"```\n(?=[^\n]*\$ )" + r".*?```", "", text, flags=re.S)
    for pattern, kind in PATTERNS:
        for m in re.finditer(pattern, text):
            token = m.group(1)
            yield kind, token, text[:m.start()].count("\n") + 1


def test_the_artifact_scan_found_something():
    """Guard the guard: an empty value set would make every case below vacuous."""
    assert len(VALUES) > 500, f"only {len(VALUES)} values read from results/"
    assert "0.997" in VALUES, "the headline PR-AUC is not in the scanned values"


@pytest.mark.parametrize("doc", DOCS)
def test_every_published_figure_still_exists_in_results(doc):
    text = (ROOT / doc).read_text()
    stale = []
    for kind, token, line in _figures(text):
        bare = token.replace(",", "")
        if bare in VALUES or token in VALUES:
            continue
        if (doc, token) in HISTORICAL:
            continue
        if (doc, token) in DERIVED:
            continue
        stale.append(f"{doc}:{line}  {kind} {token!r}")

    assert not stale, (
        "these figures are in the prose but in no artifact under results/, so "
        "either an experiment moved and the text did not, or the figure is "
        "deliberately historical and belongs in HISTORICAL with a reason:\n  "
        + "\n  ".join(stale))


def test_the_derived_figures_still_compute_to_what_is_published():
    """An allowlist entry that is never recomputed is just a suppression.

    Two assertions, and the second is the one that matters. Recomputing alone
    only proves the arithmetic still gives the same answer - it says nothing
    about what the document actually prints. Changing a published cost from
    19,040 to 19,400 passed every other check in this file, because 19400
    happens to collide with a value somewhere in results/. Requiring the
    computed figure to appear in the document closes that: for a derived number,
    membership in a global value set is not evidence of anything.
    """
    for key, holds in _check_derived().items():
        doc, token = key
        assert holds, (
            f"{token!r} in {doc} no longer recomputes to the published value; "
            f"it was allowed on the grounds that {DERIVED[key]}")
        assert token in (ROOT / doc).read_text(), (
            f"{token!r} is the computed value but {doc} no longer prints it - "
            "the figure was edited away from its own arithmetic")


def test_the_historical_allowlist_has_not_gone_stale():
    """An allowlisted figure that is no longer in the document is dead weight."""
    for (doc, token), reason in HISTORICAL.items():
        assert token in (ROOT / doc).read_text(), (
            f"{token!r} is allowlisted for {doc} but no longer appears there; "
            f"drop the entry. Reason given was: {reason}")


def test_the_guard_is_strong_enough_to_be_worth_running():
    """How often does a WRONG figure survive this check?

    Measured rather than assumed, because the first version of this file accepted
    63.8% of randomly generated percentages and would have passed almost any
    typo. Each shape is sampled and the false-pass rate held under a ceiling; if
    results/ grows dense enough to push a rate back up, this fails and the shape
    has to be narrowed or dropped rather than silently believed.
    """
    import random

    random.seed(0)
    shapes = {
        # 22% is what three decimals can do against this many artifacts: it
        # catches roughly four stale figures in five, which is how 0.695 and
        # 0.840 were found. Reported honestly rather than claimed as coverage.
        "0.xxx": ([f"0.{random.randint(100, 999)}" for _ in range(3000)], 0.25),
        "0.xxxx": ([f"0.{random.randint(1000, 9999)}" for _ in range(3000)], 0.10),
        "grouped": ([f"{random.randint(1, 999)},{random.randint(0, 999):03d}"
                     for _ in range(3000)], 0.05),
    }
    weak = []
    for name, (sample, ceiling) in shapes.items():
        rate = sum(1 for t in sample
                   if t.replace(",", "") in VALUES or t in VALUES) / len(sample)
        if rate > ceiling:
            weak.append(f"{name}: {rate:.1%} of wrong values pass (ceiling {ceiling:.0%})")
    assert not weak, (
        "this guard has become too permissive to catch a stale figure:\n  "
        + "\n  ".join(weak))
