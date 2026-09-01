"""Published claims must reconcile with `results/` - the other direction.

`test_docs_match_results.py` asserts that figures derived from `results/`
*appear* in the docs. That is one-directional: it cannot see a number that is
written in the docs, backed by nothing, and simply stale. Four such figures
survived a full re-run of every experiment precisely because nothing looked
this way, which is defect (1) of the engineering log wearing a new costume -
the property was intended, not asserted.

A blanket scan was tried and rejected on evidence, not taste: run across the
docs it raised 26 flags of which nearly all were false - `1918` (Hirayama),
`3600` (the window in seconds), and a run of regex artifacts where `₹3,405`
matched as `405`. A check with that signal-to-noise is suppressed within a
week, and a suppressed check is worse than none, because it still reads as
coverage.

So this is a manifest instead: each entry names one *published claim*, where it
is written, and the artifact it must agree with. Small, unambiguous, and it
fails loudly for a real reason.

**What this does not promise.** It closes one blind spot, not the general one.
A figure with no entry here is unguarded, and adding a claim to the docs
without adding a check remains possible - the manifest only makes that omission
visible in review. Some claims are not mechanically checkable at all: the
*scope* of a number ("median across eight streams") is prose about which
population produced it, and no assertion here can tell a correct caption from a
wrong one. This is a narrower guarantee than "the docs cannot go stale", and it
should be cited as the narrower one.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
LOG = (ROOT / "docs" / "engineering-log.md").read_text()


def _csv(name):
    return list(csv.DictReader((ROOT / "results" / name).open()))


def _json(name):
    return json.loads((ROOT / "results" / name).read_text())


def test_readme_defect_count_matches_the_engineering_log():
    """The README summarises a count the log actually enumerates."""
    claimed = re.search(r"\|\s*(\d+) defects,", README)
    assert claimed, "the README's 'What broke?' row no longer states a defect count"
    actual = len(re.findall(r"^\d+\. \*\*", LOG, re.M))
    assert int(claimed.group(1)) == actual, (
        f"README claims {claimed.group(1)} defects, the engineering log "
        f"enumerates {actual}.")


def test_readme_skip_note_matches_the_site_test_module():
    """The 'N tests skip without dev deps' note is arithmetic, so check it."""
    m = re.search(r"the (\w+) tests that drive the demo page\s*\n?\s*skip"
                  r".*?(\d+) are collected rather than (\d+)", README, re.S)
    assert m, "the README's skip note has changed shape"
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    claimed = words.get(m.group(1).lower())
    assert claimed, f"could not read a number from {m.group(1)!r}"
    without, full = int(m.group(2)), int(m.group(3))
    actual = len(re.findall(
        r"^def test_", (ROOT / "tests" / "test_site_renders.py").read_text(), re.M))
    assert claimed == actual, (
        f"README says {claimed} site tests skip, the module defines {actual}.")
    assert full - without == actual, (
        f"README says {full} - {without} = {full - without} tests are lost "
        f"without the dev dependencies, but the site module defines {actual}.")


def test_log_shared_entity_figure_matches_the_seed_sweep():
    """The retraction in defect (6) quotes a PR-AUC. It must be the measured one."""
    m = re.search(r"collapses to (0\.\d+) once legitimate", LOG)
    assert m, "defect (6) no longer quotes a collapsed PR-AUC"
    row = next(r for r in _csv("seeds_summary.csv")
               if r["detector"] == "shared_entity")
    expected = f"{float(row['pr_auc_median']):.3f}"
    assert m.group(1) == expected, (
        f"the log says raw co-occurrence collapses to {m.group(1)}, "
        f"seeds_summary.csv reports {expected}.")


def test_log_idle_benchmark_matches_the_benchmark_artifact():
    """The discarded-benchmark note quotes the idle re-run; that is a result."""
    m = re.search(r"The idle run is ([\d.]+) ms", LOG)
    assert m, "the discarded-benchmark note no longer quotes an idle figure"
    expected = f"{_json('benchmark.json')['p50_ms']:.2f}"
    assert m.group(1) == expected, (
        f"the log says the idle run is {m.group(1)} ms, "
        f"benchmark.json reports {expected} ms.")


def test_log_gate_verdict_is_qualified_by_the_depth_it_was_measured_at():
    """Defect (8)'s superseded figures must stay marked as superseded.

    The `24 to 17` false-positive swing is real but historical: it was measured
    at two layers, and calibration later selected three, where the same
    ablation says the gate is noise. `docs/evaluation.md` carries that
    correction. The log said the old thing flatly and so contradicted it - the
    two must not drift apart again, because the log is what a reader checks
    when they want to know whether a retraction was honest.
    """
    para = re.search(r"^8\. \*\*.*?(?=^9\. \*\*)", LOG, re.S | re.M)
    assert para, "defect (8) not found in the engineering log"
    para = para.group(0)
    if "24 to 17" in para:
        assert "two-layer" in para or "two layers" in para, (
            "defect (8) quotes the two-layer false-positive swing without "
            "saying it was measured at two layers - at the selected depth of "
            "three the same ablation reports noise.")
        assert "noise" in para, (
            "defect (8) states the superseded verdict without the correction "
            "that docs/evaluation.md carries.")


def test_log_gate_noise_figure_matches_the_architecture_sweep():
    """The `+0.0005` narrowing is a current measurement, so check it."""
    m = re.search(r"median of \+([\d.]+) PR-AUC at full", LOG)
    assert m, "defect (8) no longer quotes the gate's current effect size"
    row = next(r for r in _csv("architecture_delta.csv") if r["camouflage"] == "1.0")
    expected = f"{abs(float(row['delta_add_gate'])):.4f}"
    assert m.group(1) == expected, (
        f"the log says adding the gate back is worth +{m.group(1)} PR-AUC, "
        f"architecture_delta.csv reports {expected}.")


def test_readme_incident_funnel_is_one_consistent_scope():
    """Alerts, incidents and actions in the funnel must come from one stream."""
    m = re.search(r"(\d+) event alerts\s*->\s*(\d+) incidents"
                  r"\s*->\s*(\d+) actions? recommended", README)
    assert m, "the README's incident funnel has changed shape"
    alerts, incidents, actions = (int(g) for g in m.groups())
    pol = _json("policy.json")
    causal = next(r for r in pol["summary"] if r["policy"] == "causal_policy")
    assert alerts == causal["events_alerted"], (
        f"funnel says {alerts} alerts, policy.json says {causal['events_alerted']}")
    assert incidents == causal["incidents_formed"], (
        f"funnel says {incidents} incidents, policy.json says "
        f"{causal['incidents_formed']}")
    assert actions == causal["incidents_actioned"], (
        f"funnel says {actions} actions, policy.json says "
        f"{causal['incidents_actioned']} for the same stream - the funnel must "
        f"not mix a single stream with the across-stream median.")
