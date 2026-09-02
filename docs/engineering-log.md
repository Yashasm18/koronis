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
   counting, previously reported at 0.894 PR-AUC, collapses to 0.051 once legitimate
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

8. **An architectural claim that was never tested, and did not hold — then held less
   than that.** The mechanism and relation ablations measured which *data sources*
   mattered; the heterophily gate and the learned relation attention were design decisions
   in the model itself, argued for in prose and never removed. Ablating them at the
   two-layer depth the model then used said the gate was **net-negative** — better without
   it on 12 of 15 seed × camouflage cells, false positives falling 24 to 17 — and that the
   harm grew with camouflage, the inverse of the stated rationale. The claim was retracted
   in place; the gate was not removed, because selecting architecture on test results is
   the leakage this project refuses elsewhere.

   It was removed later, by [calibration-based selection](evaluation.md#closing-the-loop-selecting-an-architecture-without-touching-test),
   which also chose three layers — and at three layers the ablation no longer says what it
   said at two. Adding the gate back is now **noise**: a median of +0.0005 PR-AUC at full
   camouflage against a seed-to-seed spread of 0.992 to 0.999, winning on 2 of 5 seeds.
   So the retraction itself needed narrowing. The honest statement is that the gate was
   harmful at the depth it was first measured at, is neither here nor there at the depth
   finally selected, and was removed on cost by selection rather than by that table.
   [The current numbers.](evaluation.md#does-the-architecture-earn-its-place) The component
   that does earn its place is depth itself.

9. **A baseline that became a copy of itself.** After model selection turned the
   heterophily gate off by default, the architecture ablation kept comparing `full` — a
   variant defined as "no departures from the default" — against `no_gate`. Those are the
   same model once the default changes, and the sweep dutifully reported a difference of
   exactly 0.0000 at every camouflage level rather than failing. The variants are now
   expressed as departures *from the selected architecture* (`add_gate` rather than
   `no_gate`), and a test asserts no variant's settings match the defaults it is supposed
   to depart from. A null result that arrives as a clean zero is worth more suspicion than
   a noisy one.

10. **The demo site rendered perfectly and did nothing.** Renaming an architecture
   variant in `results/` left the site's table code looking up a key that no longer
   existed. Reading `.pr` off `undefined` threw, which killed the entire page script — so
   the console loaded, drew every static panel, and then silently refused to replay.
   Every doc figure was correct, every other test passed, and the live demo was broken.
   The site is generated from the same artifacts as the prose but can fail in ways prose
   cannot, so `tests/test_site_renders.py` now loads the built page, asserts no script
   error, asserts every table has rows, and asserts the replay actually advances.

11. **A replay so fast it read as a broken button.** The loop advanced the stream clock
   by a fixed amount per animation frame, which put the whole 4,500-second replay through
   in **2.9 seconds**. Clicking *Replay incident* appeared to do nothing: by the time you
   looked, it had finished and the button had reset. Every automated check passed, because
   every one of them asked whether the replay advanced rather than whether a person could
   watch it. Pacing is now against the wall clock — which also removes a second defect
   nobody had noticed, that a 120 Hz display replayed twice as fast as a 60 Hz one — with
   a 1× / 4× / 16× control for readers who do not want to wait. Tests assert both the pace
   and that the speed control changes it.

12. **A streaming guarantee that was true by accident.** The replay cached only layer-1
   outputs, computing layer 2 from neighbours' cached layer 1. That reproduced batch
   scores exactly — for a two-layer model. When a calibration sweep selected three layers,
   the parity test failed, because layer 3 needs neighbours' layer 2 and nothing was
   keeping it. The property had been presented as a consequence of the backwards-in-time
   edge rule, which it is; the implementation only supported it at one depth. Every
   intermediate layer is cached now, and parity is asserted at one, two, three and four
   layers rather than at whatever depth happens to be current.

13. **The same replay drew a different graph depending on how fast you watched it.**
   The evidence graph seeded each node's position from its index inside a 160-event
   display window. That index depends on how many events arrive per animation frame, which
   depends on playback speed — so at 1× the graph drew 1,315 lit pixels and at 16× it drew
   5,032, from identical data. Every number agreed at every speed; only the picture moved,
   which is why the existing checks passed. Position is now a pure function of the event's
   index in the append-only `seen` list, and a test plays the stream to completion at 1×,
   4× and 16× and asserts the end states are identical down to the canvas pixel count. It
   reproduces the old behaviour when reverted.

14. **The picture at the top of the README was the one thing no test read.** The demo
   recording sat unchanged through four commits that re-ran the experiments behind it. It
   showed an action ladder of ₹21,973 / ₹10,008 / ₹3,646 / ₹1,667 and a frozen threshold
   of 0.6785 while `results/policy.json` had moved to ₹25,331 / ₹11,519 / ₹4,150 / ₹1,768
   at threshold 0.9366 — so the first thing a reader saw contradicted the text beside it.
   It also showed a replay panel with no speed control, because the control was added
   afterwards. Every *number* in the README was covered by a test; the image asserting
   those numbers was covered by nothing, because it is a binary. A stamp committed beside
   the recording now hashes the two things a viewer can read off the frames — the contents
   of `results/`, and the page's control surface — and a test fails when either moves.
   Re-recording refreshes the stamp as its last step, so the two cannot drift apart.

   The recorder had rotted the same way. It slowed the replay by throttling
   `requestAnimationFrame`, which worked only while the replay advanced a fixed amount per
   frame; defect 11 moved pacing onto the wall clock, after which throttling cut the frame
   rate and nothing else. It is now in the repository as `site/record_demo.py` rather than
   in an ignored scratch directory, uses the page's own 4× control, and waits for the
   stream to drain instead of sleeping for a guessed duration.

15. **Three published tables were reporting a run that had been superseded.** The demo
   site and the documents disagreed, and the site was right: `site/build.py` reads
   `results/` at build time and cannot go stale, while the tables in
   [`evaluation.md`](evaluation.md) are written by hand. The per-relation ablation was
   wrong in every cell — it reported PR-AUC 0.989 for the full model against an actual
   0.995, and 24 false positives against 12. The aperture sweep reported a gap of 0.073 at
   sixteen merchants against an actual 0.070. The online/batch table reported 43 incidents
   on stream 0 against 41. Prose carried the same rot: the relations paragraph claimed
   recall fell "from 0.968 to 0.813" when the figures were 0.988 to 0.868, and claimed
   dropping `device_id` cut false positives when it raises them from 12 to 14; the policy
   paragraph said 213 analyst minutes beside a table saying 211; the incident-reliability
   paragraph cited four figures from an older fit directly beneath the table that
   contradicted them.

   The doc guard was green throughout, and the reason is structural. It asserted, for each
   figure computed from an artifact, that the figure *appears somewhere in the docs* —
   results → docs. That direction can only detect a number that is **missing**. A
   superseded number is not missing; it is extra, and nothing was looking for extras.
   `test_doc_tables_match_results.py` now renders each of these tables from its artifact
   and requires it verbatim, so a re-run that moves any cell fails and the message names
   the row.

   The demo console had a related defect of its own: it reported "*N* linked alerts
   ... connected by 6 device, 5 IP and 6 BIN relations", stapling the window's alert count
   to one event's neighbour counts. Those counts sum to that event's `linked` field — 17,
   not 395 — which is checkable on all 6,400 replay events and holds on every one.

   The stamp added with defect 14 would not have caught the corrected sentence. It hashed
   the page's *control surface* on the argument that a copy edit should not force a
   re-record; but the sentence is built in JavaScript, no control changed, and the
   recording would have kept displaying a retracted claim. It hashes the whole built page
   now. The earlier scoping was an argument, and the ablation for it was never run.

16. **The threshold label was drawn into the busiest band of the chart.** "frozen
   threshold" was printed five pixels above the dashed rule, and drawn *before* the score
   points. The threshold is 0.9366 and every alerting event scores 1.0, so those five
   pixels are the densest part of the chart: the dots landed on top of the words and the
   rule's own dashes struck through them. Nothing failed — the label was present, correctly
   positioned relative to the line it annotates, and unreadable. The rule and its label are
   drawn last now, with the text below the line where the plot is empty, on a chip of the
   canvas background.

   The test reads the rendered canvas rather than the drawing code: it locates the dashed
   rule as the row with the most red, treats the remaining red as the label, and asserts
   the label lies below the rule with **zero** score points drawn over it. On the previous
   drawing it reports the label at rows 13–19 against a rule at row 25, with 196 blue
   pixels on top of it. A count of red pixels alone would not have done — that only moves
   from 440 to 481, which is too small a margin to assert on.

17. **Three ways the stream failed without saying so.** Fault injection against the live
   `push` path — not a code review — turned up three defects, and all three were silent:
   the stream kept running and kept returning answers.

   A non-finite feature produced a NaN score. `NaN >= threshold` is `False` in IEEE
   arithmetic, so the event reported itself as **"no alert"**: a missing `amount` or a null
   `approved` made the detector quietly stop detecting. A missing entity became an entity —
   values were interned with `str(value)`, so a null turned into the key `"None"` and every
   event without a device fingerprint linked to every other one, which is a ring
   manufactured out of absent data on traffic where null fingerprints are ordinary. And
   nothing was ever evicted: 3,120 events through a 60-second window left 3,120 rows in
   every layer cache while 12 events were in scope, and the entity index kept a key for
   every distinct value ever seen — the structure a campaign minting a fresh entity per
   attempt inflates fastest.

   The rule adopted is that an event which cannot be scored honestly is **escalated, not
   scored anyway**: quarantined, counted, and returned with a reason. Losing an event
   loudly is recoverable; a fraud detector that says "no alert" when it means "I could not
   read this" is not. Absent entity values are treated one-sidedly, the same argument the
   sketch uses — refusing to link on an absent value can only fragment an incident, which
   an analyst can still see, while linking on it invents coordination that was never there.
   `koronis.cli resilience` now injects each fault and measures the consequence, and
   `placeholder_device` is the control that makes the point: it is the *same* missing data
   with a constant substituted upstream, and it draws **958 device links at a 1% rate and
   19,792 at 10%** where the null draws none. At 10% the link-share cap already refuses to
   link on a value that common, so recall barely moves — but those links still reach the
   audit dossier and still tell an analyst that unrelated attempts share a device.

   Two claims had to be withdrawn and re-earned. The README's "memory is bounded by the
   window rather than by traffic history" was false for the streaming scorer, and
   SECURITY.md repeated it in a section written one commit earlier. Both now cite the
   measurement — 1,859 peak rows against 6,310 events — and a test fails if the caches go
   back to tracking total traffic.

18. **Was the gap a modelling gap or an information gap?** Every headline comparison here
   is against per-transaction models, so the obvious objection is that the baselines were
   simply too small. `koronis.cli ceiling` tests it instead of arguing: the per-event
   feature set is held fixed while capacity is scaled across two families with different
   inductive biases, on the same 60-epoch budget as every published number. The first run
   used the 40-epoch default and was discarded before it was written down — an unfair
   training budget would have flattered the conclusion in exactly the direction the
   conclusion points.

An inference benchmark reporting p50 1.78 ms was also discarded: it was measured while a
training job ran on the same machine. The idle run is 0.91 ms.

Three lessons, none about neural networks. From (1) and (4): a property you assert in a
test gets checked; a property you merely intend gets silently violated the moment an
unrelated helper changes. From (5): a preprocessing step that seems neutral can decide
your conclusion, and the dangerous ones are those whose bias points your way. From (6):
when a result looks too clean, suspect the simulation before congratulating the model. And
from (8): *an argument for a design decision is not evidence for it — the ablation you did
not run is the claim you cannot make.*
