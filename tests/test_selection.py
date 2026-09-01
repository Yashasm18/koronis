"""Model selection must not be able to see the test split.

The point of `koronis.cli select` is that a candidate is chosen using only
calibration data. These pin the properties that make that claim true, since a
selection protocol that quietly reads test would produce exactly the same
tables and be worthless.
"""
import inspect
import json
import re
from pathlib import Path

import koronis.cli as cli

ROOT = Path(__file__).resolve().parents[1]
SRC = inspect.getsource(cli.select)


def test_candidates_cover_every_component_measured_as_net_negative():
    """device, email and the gate were each measured as removable; the
    selection has to actually offer removing them."""
    names = set(cli.SELECT_CANDIDATES)
    assert "full" in names
    assert any("no_device" in n for n in names)
    assert any("no_email" in n for n in names)
    assert any("no_gate" in n or n == "lean" for n in names)
    for rels, gate in cli.SELECT_CANDIDATES.values():
        assert rels, "a candidate must keep at least one relation"
        assert isinstance(gate, bool)


def test_the_winner_is_picked_on_calibration_not_on_test():
    """The selection statement must reference the calibration column only."""
    picks = re.findall(r'idxmin\(\)[^\n]*', SRC) + re.findall(r'\["(\w*cost\w*)"\]\.idxmin', SRC)
    assert picks, "no argmin over a cost column found in select()"
    assert 'med["select_cost_inr"].idxmin()' in SRC, \
        "the winner must be chosen on select_cost_inr (calibration)"
    assert 'test_cost_inr"].idxmin' not in SRC, \
        "select() must never argmin over a test column"


def test_threshold_and_selection_use_different_calibration_draws():
    """Fitting the threshold and scoring the candidate on the same events
    flatters whichever variant suits that draw."""
    seeds = re.findall(r"_calibration_set\(seed \* 10 \+ (\d+)\)", SRC)
    assert len(seeds) == 2, f"expected two calibration draws, found {seeds}"
    assert seeds[0] != seeds[1], "both calibration sets use the same seed"


def test_recorded_verdict_is_internally_consistent():
    v = json.loads((ROOT / "results" / "select.json").read_text())
    assert v["selected_on_calibration"] in cli.SELECT_CANDIDATES
    assert v["held_up_on_test"] == (
        v["selected_test_cost_inr"] < v["full_test_cost_inr"])


def test_relations_are_restored_after_each_candidate():
    """select() patches a module-level list; leaking that would corrupt every
    later experiment in the same process."""
    from koronis.data import schema
    before = list(schema.RELATIONS)
    assert "finally:" in SRC and "schema.RELATIONS[:] = original" in SRC
    assert list(schema.RELATIONS) == before
