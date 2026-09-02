"""Tables published in the docs are re-derived from `results/` and compared cell by cell.

`test_docs_match_results.py` checks the other direction: for each figure computed
from an artifact, that the figure appears somewhere in the docs. That is a real
check, but it is one-directional. It can only notice a *missing* number. A number
that is present but superseded is not missing - it is extra, and nothing was
looking for extras.

Defect 15 is what that gap costs. Three whole tables in docs/evaluation.md - the
per-relation ablation, the online/batch comparison and the merchant-vs-gateway
aperture sweep - still held figures from a run that had been superseded. Every
cell of the relations table was wrong; the aperture table said the gap at sixteen
merchants was 0.073 when results/aperture.csv said 0.070; the online table
reported 43 incidents on stream 0 against an actual 41. The demo site was right
the whole time, because site/build.py reads results/ at build time and cannot go
stale. Only the hand-maintained tables drifted.

So these tables are rendered here from the artifacts and required to appear
verbatim. A re-run that moves any cell fails, and the failure names the row.
"""
import csv
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = (ROOT / "docs" / "evaluation.md").read_text()


def _norm(text: str) -> str:
    """Compare figures, not emphasis. Bold and digit grouping are the doc's
    business; the numbers inside them are the artifact's."""
    return text.replace("**", "").replace(",", "")


EVAL_NORM = _norm(EVAL)


def _rows(name):
    with (ROOT / "results" / name).open() as fh:
        return list(csv.DictReader(fh))


def _f(r, k, d=3):
    return f"{float(r[k]):.{d}f}"


def _i(r, k):
    return str(int(float(r[k])))


def _count(r, k):
    """A count that may be a median across an even number of streams.

    Truncating with int() published 1,859 where the median is 1,859.5, while the
    README rounds 9.5 to 10 three sections earlier. Neither rounding nor
    truncating is right for a table of medians: the repo's existing convention,
    set by the policy table, is to print the fraction.
    """
    v = float(r[k])
    return f"{v:,.1f}".removesuffix(".0")


def relations_rows():
    by = {r["variant"]: r for r in _rows("relations.csv")}
    base = float(by["all"]["pr_auc"])
    labels = [("all", "all relations"), ("no_device_id", "no `device_id`"),
              ("no_ip_id", "no `ip_id`"), ("no_bin_id", "no `bin_id`"),
              ("no_email_domain", "no `email_domain`")]
    for key, label in labels:
        r = by[key]
        change = "—" if key == "all" else f"{float(r['pr_auc']) - base:+.4f}"
        yield (f"| {label} | {_f(r,'pr_auc')} | {_f(r,'precision')} | "
               f"{_f(r,'recall')} | {_i(r,'false_positives')} | {change} |")


def online_rows():
    for r in _rows("online.csv"):
        yield (f"| {r['stream']} | {r['campaign_attempts']} | "
               f"{r['batch_incidents']} / {r['online_incidents']} | "
               f"{_f(r,'batch_purity')} / {_f(r,'online_purity')} | "
               f"{_f(r,'batch_recall')} / {_f(r,'online_recall')} |")


def aperture_rows():
    by = {}
    for r in _rows("aperture.csv"):
        by.setdefault(int(r["n_merchants"]), {})[r["view"]] = r
    for m in sorted(by):
        g, mer = by[m]["gateway"], by[m]["merchant"]
        gap = float(g["pr_auc"]) - float(mer["pr_auc"])
        share = round(float(g["campaign_share_largest_merchant"]) * 100)
        # the sweep's endpoints are bolded in the doc as the control and the extreme
        cell = f"**{gap:.3f}**" if m in (1, 16) else f"{gap:.3f}"
        yield (f"| {m} | {_f(g,'pr_auc')} | {_f(mer,'pr_auc')} | {cell} | {share}% |")


def reliability_rows():
    for r in _rows("incident_reliability.csv"):
        if not r["predicted"]:
            continue                              # empty bin, not published
        yield (f"| {_f(r,'predicted')} | {_f(r,'observed')} | {_i(r,'count')} |")


def resilience_rows():
    for r in _rows("resilience.csv"):
        yield ("| `{}` | {:.0f}% | {} | {} | {} | {} | {} |".format(
            r["fault"], float(r["rate"]) * 100,
            _count(r, "quarantined"), _count(r, "nan_scores"),
            _count(r, "device_links_on_corrupted"),
            _f(r, "campaign_recall"), _count(r, "peak_cache_rows")))


def ceiling_rows():
    for r in _rows("ceiling.csv"):
        # The 256-wide rows are deliberately not published: both families
        # diverge there under the shared budget, which is a training failure
        # rather than a ceiling. They stay in the artifact.
        if r["capacity"].startswith("256"):
            continue
        yield ("| {} | {} | {:,} | {} |".format(
            r["family"], r["capacity"].replace(" x ", " × "),
            int(r["params"]), _f(r, "pr_auc", 4)))


def saturation_rows():
    """Only the four rows the doc publishes: the two spreads either side of k = n."""
    for r in _rows("saturation.csv"):
        if r["k_over_n"] not in ("0.5", "1.0"):
            continue
        yield ("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["n"], r["k"], r["k_over_n"], _f(r, "koronis_recall", 1),
            _f(r, "campaign_isolated", 1), _f(r, "background_isolated", 4),
            _i(r, "entity_values_shared_with_background")))


def policy_rows():
    labels = {"always_allow": "always allow", "always_hold": "always hold",
              "event_thresholding": "event-by-event thresholding",
              "causal_policy": "**causal policy** *(forecast only)*",
              "oracle_policy": "oracle policy *(upper bound)*"}
    for r in _rows("policy_across_streams.csv"):
        label = labels.get(r["policy"])
        if label is None:
            continue
        yield (f"| {label} | {_f(r,'incidents_actioned',1)} | "
               f"{_f(r,'false_incidents',1)} | {_f(r,'analyst_minutes',1)} | "
               f"₹{int(round(float(r['merchant_cost_inr']))):,} |")


TABLES = {
    "per-relation ablation": relations_rows,
    "online vs batch": online_rows,
    "merchant vs gateway aperture": aperture_rows,
    "incident reliability": reliability_rows,
    "response policy": policy_rows,
    "failure behaviour": resilience_rows,
    "per-event ceiling": ceiling_rows,
    "saturation": saturation_rows,
}


@pytest.mark.parametrize("name", sorted(TABLES))
def test_the_published_table_matches_the_artifact_it_came_from(name):
    missing = [row for row in TABLES[name]() if _norm(row) not in EVAL_NORM]
    assert not missing, (
        f"the {name} table in docs/evaluation.md does not match results/. "
        f"These rows, rendered from the artifact, are not in the document:\n  "
        + "\n  ".join(missing)
        + "\nRe-run the experiment and update the table, or the doc is reporting "
          "a superseded run."
    )
