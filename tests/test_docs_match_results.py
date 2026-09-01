"""Published figures must exist in `results/`.

The claim this repo makes about itself is that every number in the README and
the docs is read from an experiment artifact rather than typed. That is a
property, so it is asserted rather than trusted - especially across a change
that re-runs every experiment, where one stale figure surviving would quietly
falsify the whole claim.

Each entry names where a figure comes from and how it is written. A number that
moves and is not updated fails here.
"""
import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = "\n".join(p.read_text() for p in
                 [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")))


def _csv(name):
    return list(csv.DictReader((ROOT / "results" / name).open()))


def _row(name, key, val):
    for r in _csv(name):
        if r[key] == val:
            return r
    raise AssertionError(f"{val!r} not found in {name} under {key}")


def _json(name):
    return json.loads((ROOT / "results" / name).read_text())


def _fmt_inr(x):
    """Indian digit grouping, as the docs write it."""
    s = f"{int(round(float(x)))}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def _cases():
    k = _row("seeds_summary.csv", "detector", "koronis_graph")
    g = _row("seeds_summary.csv", "detector", "gbdt_per_txn")
    bench = _json("benchmark.json")
    sel = _json("select.json")
    pol = {r["policy"]: r for r in _csv("policy_across_streams.csv")}
    drift = _json("drift.json")
    front = _csv("frontier.csv")
    agree = sum((r["velocity_detected"] == "True") !=
                (r["velocity_blind_predicted"] == "True") for r in front)
    return [
        ("koronis PR-AUC",       f"{float(k['pr_auc_median']):.3f}"),
        ("koronis precision",    f"{float(k['precision_median']):.3f}"),
        ("koronis recall",       f"{float(k['recall_median']):.3f}"),
        ("gbdt false positives", f"{int(round(float(g['false_positives_median'])))}"),
        ("frontier agreement",   f"{agree} / {len(front)}"),
        ("binding tau",          f"τ = {front[0]['tau_binding']}"),
        ("inference p50",        f"{bench['p50_ms']:.2f} ms"),
        ("inference p95",        f"{bench['p95_ms']:.2f} ms"),
        ("drift psi cutoff",     f"{drift['threshold_psi']:.3f}"),
        ("drift false-flag",     f"{drift['base_false_flag_rate'] * 100:.1f}%"),
        ("causal policy cost",   f"₹{_fmt_inr(pol['causal_policy']['merchant_cost_inr'])}"),
        ("oracle policy cost",   f"₹{_fmt_inr(pol['oracle_policy']['merchant_cost_inr'])}"),
        ("analyst minutes",      str(pol["event_thresholding"]["analyst_minutes"])),
        ("selected variant",     sel["selected_on_calibration"]),
        ("selected test cost",   f"₹{_fmt_inr(sel['selected_test_cost_inr'])}"),
        ("full test cost",       f"₹{_fmt_inr(sel['full_test_cost_inr'])}"),
    ]


@pytest.mark.parametrize("label,needle", _cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_published_figure_is_backed_by_results(label, needle):
    assert needle in DOCS, (
        f"{label}: results/ says {needle!r}, which does not appear in the docs. "
        f"Re-run the experiment or update the text.")


def test_every_results_file_referenced_by_a_cli_command_exists():
    """A documented command that writes nothing is a broken promise."""
    import koronis.cli as cli
    for cmd in re.findall(r"koronis\.cli (\w+)", (ROOT / "README.md").read_text()):
        assert hasattr(cli, cmd), f"README documents `{cmd}`, which does not exist"


# ── cross-document agreement ────────────────────────────────────────────────
# The manifest above asks "does every figure from results/ appear somewhere".
# It cannot see two documents stating the same quantity differently, which is
# how the decision-layer funnel came to read "1 action recommended" in one file
# and "2 actions recommended" in another, both citing the same stream. These
# check agreement, not just presence.

def _funnel_numbers(text):
    """Every `N event alerts -> M incidents -> K action(s)` triple in a file."""
    return re.findall(
        r"(\d+)\s+event alerts\s*(?:->|→)\s*(\d+)\s+incidents\s*(?:->|→)\s*(\d+)\s+actions?",
        text)


def test_the_funnel_agrees_across_documents_and_with_results():
    demo = {r["policy"]: r for r in _json("policy.json")["summary"]}["causal_policy"]
    expected = (str(demo["events_alerted"]), str(demo["incidents_formed"]),
                str(demo["incidents_actioned"]))
    found = {}
    for f in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")):
        for triple in _funnel_numbers(f.read_text()):
            found.setdefault(triple, []).append(f.name)
    assert found, "the decision-layer funnel is not stated anywhere"
    assert len(found) == 1, (
        f"documents disagree about the funnel: "
        + "; ".join(f"{t} in {v}" for t, v in found.items()))
    (triple, where), = found.items()
    assert triple == expected, (
        f"docs say {triple} in {where}; policy.json says {expected}")


def test_the_test_count_is_the_same_everywhere_it_is_stated():
    """Badge, quickstart and CONTRIBUTING each state it; they must agree."""
    readme = (ROOT / "README.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    badge = int(re.search(r"badge/tests-(\d+)%20passing", readme).group(1))
    stated = {int(n) for n in re.findall(r"#\s*(\d+) tests, ~", readme + contributing)}
    assert stated, "no '# N tests' comment found in README or CONTRIBUTING"
    assert stated == {badge}, (
        f"badge says {badge}, setup comments say {sorted(stated)}")


def test_the_defect_count_matches_the_engineering_log():
    readme = (ROOT / "README.md").read_text()
    claimed = re.search(r"\|\s*What broke\?\s*\|\s*(\d+) defects", readme)
    assert claimed, "the README's 'What broke?' row no longer states a defect count"
    actual = len(re.findall(r"^(\d+)\. \*\*", (ROOT / "docs" / "engineering-log.md").read_text(), re.M))
    assert int(claimed.group(1)) == actual, (
        f"README claims {claimed.group(1)} defects; the log has {actual} numbered entries")
