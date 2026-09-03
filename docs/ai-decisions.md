# What AI is in here, and what was left out

Every model choice in Koronis was made on the calibration split and is recorded with the
measurement that decided it — including the ones that were rejected, and the one that was
kept despite earning nothing. The point of this file is that none of these are arguments.

> An argument for a design decision is not evidence for it — the ablation you did not run
> is the claim you cannot make.

## What is in the model

A relational message-passing network, **32 wide × 3 deep, 9,171 parameters**, written from
scratch in [`layers.py`](../koronis/models/layers.py) with no graph library. Six per-event
features; three relations (`device_id`, `ip_id`, `bin_id`) carried as graph structure.

It was **selected, not chosen**. `koronis.cli capacity` sweeps a width × depth grid and
picks on calibration cost, never touching test. The record of that run says
`"default_is_the_winner": false` — the sweep changed the answer, which is the evidence it
was not decorative. Depth is the piece that earns its keep: two layers instead of three
costs 0.0010 PR-AUC at full camouflage.

## Measured, and removed

**The heterophily gate.** Argued for on the grounds that rings camouflage into legitimate
traffic, which predicts it should help *more* as camouflage rises. It did the opposite:
measurably net-negative at two layers, and at the three layers finally selected the
difference is noise (−0.0005).
The argument was good and the measurement disagreed, so the component went and the claim
was retracted.

**`email_domain` as a model relation.** Removing it *improves* PR-AUC (0.995 → 0.998) and
cuts false positives from 12 to 9. It is noise to the detector — and still perfectly good
evidence for whether two alerts belong to the same incident, which is why
[`schema.py`](../koronis/data/schema.py) keeps `RELATIONS` and `MODEL_RELATIONS` apart.

## Measured, kept, and earning nothing

**The learned relation attention.** Mixing relations by a learned softmax instead of
uniformly at 1/R changes PR-AUC by **0.0000 at every camouflage level**. It is recorded here
rather than quietly retained: on this project's own standard it is a candidate for removal,
and the only reason it is still in the default is that taking it out means re-running every
published number. It costs no parameters — the ablation bypasses the weighting rather than
deleting it — so the honest statement is that it is free and useless, not that it helps.

## Never added, and the number that decided it

**A language model reading the transaction.** This is the obvious objection to every
comparison in this repo — that the baselines were simply too small. It is testable, so it
was tested: `koronis.cli ceiling` holds the per-event feature set fixed and scales capacity
across two unrelated families. Both plateau far below the graph, and more capacity makes
the tabular learner *worse*, not better: its best result in the sweep is its **smallest**
model, 1,550 parameters at **0.2891**, falling to **0.2233** at 1,020,000. The per-event
network tops out at **0.3279**. The graph reaches **0.9915 with 9,171 parameters** — fewer
than either. The ceiling is a property of what a single authorisation contains, not of who
is reading it, and a transformer reading one transaction is another per-event model.
[Full table](evaluation.md#the-per-event-ceiling).

**A language model reading the neighbourhood.** Feed it the linked attempts and it is no
longer a per-event model — it is doing the graph's job. The budget it would have to do it
in is measured: **0.909 ms p50, 1,095 events/sec** per worker, and the nine-worker sizing
in the README is derived from those constants. What a hosted model call would cost instead
is not measured here, so no multiplier is claimed — only that the budget the rest of the
design is sized against is sub-millisecond, and nothing in this repo can make a network
call to find out.

**An agent loop over the response.** There are four rungs, their costs are declared
constants, and the choice among them is an argmin over expected cost with a conformal
forecast as its only uncertain input. That decision has a closed form; wrapping it in a
planner would add nondeterminism to an argmin and make the audit dossier harder to defend,
which is the opposite of what this track asks for. The gap left to close is small and
measured: **₹9,282 against an oracle's ₹3,145** across eight streams.

**Per-entity embedding tables.** They would fit the entities in the training data, and
every entity in production is one the model has never seen. Inductiveness is the reason the
hold-out uses unseen entities at all.

## Where a language model would actually fit

Turning an incident into an analyst-readable narrative is a genuine language task, and the
[audit dossier](evaluation.md#decision-layer) is currently a template. It is not built,
for two reasons stated plainly: the package has **no network capability** and that is
asserted per module ([`test_defence_only.py`](../tests/test_defence_only.py)), so it would
sit outside the trust boundary in the analyst's own tooling; and its benefit is a human
judgement this project has no way to measure. An unmeasured component is exactly what the
gate above was removed for.
