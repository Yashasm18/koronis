# Engineering log

Repository map, and every defect that changed a published result.

[← back to the README](../README.md)


| path | responsibility |
|---|---|
| [`koronis/data/`](../koronis/data/) | Event schema, background loader, campaign injector with `(n, k, camouflage)` control |
| [`koronis/graph/build.py`](../koronis/graph/build.py) | Windowed entity-sharing graph, backwards-in-time edges, degree cap |
| [`koronis/models/layers.py`](../koronis/models/layers.py) | From-scratch relational message passing |
| [`koronis/models/koronis.py`](../koronis/models/koronis.py) | Inductive detector |
| [`koronis/models/loss.py`](../koronis/models/loss.py) | Expected-rupee-cost objective |
| [`koronis/models/velocity.py`](../koronis/models/velocity.py) | FP-budget-tuned multi-entity baseline |
| [`koronis/eval/`](../koronis/eval/) | Cost model, latency harness, calibration, frontier sweep |

### What broke

Every defect below was surfaced by running experiments, not by reading code.

1. **Entity-ID leakage.** Campaign entity ids were named per campaign index, so datasets
   built with different seeds reused the same device fingerprints — train and test shared
   entity identity. Every test passed and the metrics looked excellent, but the model
   could have keyed on entity names rather than structure, making the inductive claim
   false while appearing perfect. It surfaced only because a test *asserted* the
   hold-out's entities were disjoint.

2. **A segfault from import order.** LightGBM and PyTorch each ship an OpenMP runtime; on
   macOS, loading them in the wrong order crashes the interpreter. The package now fixes
   the order before either is reachable.

3. **Train 0.998 / test 0.000.** Trained on a single loud campaign, the model memorised
   micro-amounts instead of coordination. The training distribution now spans both spread
   and camouflage, so coordination is the only invariant available to learn.

4. **The frontier disagreed with its own prediction on 25% of the grid.** Devices and IPs
   were assigned round-robin while BINs used random sampling, and the multinomial maximum
   reached 18 attempts where the mean was 8, tripping a threshold of 9 for reasons
   unrelated to the hypothesis. With uniform spread across every entity, agreement went to
   100%.

5. **A rescaling that quietly changed a published conclusion.** Calibration and test
   scores were each divided by their own maximum before thresholding. No test labels were
   touched, but it let each split redefine what a score of `1.0` meant, so a "frozen"
   threshold referred to a different absolute quantity on each — and the conclusion drawn
   from it was wrong, in the direction that flattered this project.

6. **A simulation that made the problem easy, and three bugs behind it.** The background
   ran at 9 events/hour across thirty days, so an injected campaign was 97% of the traffic
   in its own window. Fixing the density to 1,600 events/hour uncovered three further
   defects the thin traffic had masked: every entity type shared one Zipf draw with a
   clamp, piling the tail onto one id that carried 42% of all traffic; the frontier used
   `max(τ)` as its binding constraint when a multi-entity engine needs `k ≥ n / min(τ)`;
   and the calibration split carried a 37% positive rate under which "alert on every
   event" is genuinely cost-optimal. It also reversed a headline: raw co-occurrence
   counting, previously reported at 0.894 PR-AUC, collapses to 0.055 once legitimate
   traffic is dense enough to co-occur constantly.

7. **Two concurrent rings became one blob.** The incident layer merged two independent
   campaigns into a single 597-attempt incident, bridged entirely by email domain — five
   distinct values across 607 alerts. Restricting links to values covering under 2% of the
   whole stream then over-fragmented the campaign into sixty disjoint cliques, which
   exposed the real cause: the generator assigned devices, IPs and BIN ranges in the same
   round-robin order, so all three partitioned a campaign identically and the relations
   never cross-cut. With independent assignment the campaign forms one connected component
   and two concurrent rings stay two clean incidents. That fix moved a headline — action
   regret against the oracle had been ₹0 on 11/11 incidents, and now it is not.

8. **An architectural claim that was never tested, and did not hold.** The mechanism and
   relation ablations measured which *data sources* mattered; the heterophily gate and the
   learned relation attention were design decisions in the model itself, argued for in
   prose and never removed. Ablating them says the gate is **net-negative** — at full
   camouflage, taking it out improves PR-AUC and cuts false positives from 24 to 17 — and
   that the harm grows with camouflage, the inverse of the stated rationale. The component
   that does earn its place is the second relational layer, and its benefit shows exactly
   the conditional shape the gate was supposed to have. The claim is retracted in place;
   the gate is not removed, because selecting architecture on test results is the leakage
   this project refuses elsewhere.

9. **A baseline that became a copy of itself.** After model selection turned the
   heterophily gate off by default, the architecture ablation kept comparing `full` — a
   variant defined as "no departures from the default" — against `no_gate`. Those are the
   same model once the default changes, and the sweep dutifully reported a difference of
   exactly 0.0000 at every camouflage level rather than failing. The variants are now
   expressed as departures *from the selected architecture* (`add_gate` rather than
   `no_gate`), and a test asserts no variant's settings match the defaults it is supposed
   to depart from. A null result that arrives as a clean zero is worth more suspicion than
   a noisy one.

An inference benchmark reporting p50 1.78 ms was also discarded: it was measured while a
training job ran on the same machine. The idle run is 0.99 ms.

Three lessons, none about neural networks. From (1) and (4): a property you assert in a
test gets checked; a property you merely intend gets silently violated the moment an
unrelated helper changes. From (5): a preprocessing step that seems neutral can decide
your conclusion, and the dangerous ones are those whose bias points your way. From (6):
when a result looks too clean, suspect the simulation before congratulating the model. And
from (8): *an argument for a design decision is not evidence for it — the ablation you did
not run is the claim you cannot make.*
