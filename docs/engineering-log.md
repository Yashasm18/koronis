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

19. **A paragraph that contradicted the table three lines above it.** The prose under the
   mechanism ablation reported 0.452 precision, 464 false positives, recall falling to
   0.695, false positives falling to 53, precision rising to 0.840, and a 31x reduction.
   The table directly above it said 0.451, 466, 0.812, 35, 0.903 and 10 - a 47x reduction.
   Six wrong figures in one paragraph, all of them survivors of a re-run that regenerated
   the table beneath which they sat.

   This is defect 15 again, and the guard written for defect 15 could not see it:
   `test_doc_tables_match_results.py` re-derives whole tables and nothing around them.
   `test_published_figures_are_current.py` now reads the prose, requiring every
   figure-shaped token to be a number some artifact contains.

   That guard was then measured rather than trusted, which changed it twice. Including the
   per-trial `*_raw.csv` files inflated the accept set to 88,000 values, at which point
   **30.8% of randomly generated three-decimal figures passed**. Expanding every fraction
   to a one-decimal percentage was worse: **63.8%** of random percentages passed, so
   percentages are now deliberately **not** checked rather than checked uselessly - a
   guard that accepts two wrong values in three reads as coverage while providing none.
   The false-pass rate of each remaining shape is asserted, so the check fails if it ever
   drifts back toward vacuous.

   The same sweep found six more stale figures the table guard had never covered:
   `τ_bin = 9` where `frontier.csv` says **236**; forecast coverage **96.6%** where
   `policy.json` says 95.3%; event-model ECE **0.0025** against a measured 0.0020; the
   per-event complexity bound still written as `R = 4` relations and `L = 2` layers when
   the selected model uses 3 and 3; the README and the demo site disagreeing over whether
   consolidation saves **seventeen** or **eighteen** times the triage, from the same
   17.58; and `peak_cache_rows` published as 1,859 by truncating a median of 1,859.5, in a
   repo that rounds 9.5 to 10 three sections earlier.

20. **Two documents said the detector links on email domain.** It does not: calibration
   dropped `email_domain` from `MODEL_RELATIONS`, and the distinction is the whole reason
   `schema.py` keeps two lists. The README and `architecture.md` both described the graph
   as linking on all four relations, and `architecture.md` additionally claimed the learned
   relation attention "discovers which entity type carries the signal" - measured at
   **0.0000 delta at every camouflage level**, which is the opposite of discovering
   anything.

   Auditing the same claim turned up a real asymmetry nobody had noticed: the baseline's
   free-mail feature covers gmail/yahoo/outlook while the detector's covers gmail/outlook,
   and yahoo is about 15% of generated traffic, in two feature sets a code comment called
   "identical in spirit". Measuring it produced one more lesson. An ad-hoc run said the
   extra domain *cost* the baseline 0.02 PR-AUC; run on the actual test protocol as
   `koronis.cli feature_parity`, the sign **flipped** - it helps the baseline by 0.0090,
   which is the conservative direction. The first number was already written into two
   files before the second was measured. Published figures come from `results/` for
   exactly this reason.

21. **The experiment that would have flattered the project, and did not survive its own
   diagnostic.** The frontier draws the *baseline's* failure boundary and shows Koronis
   detecting everywhere on the grid, which is half a characterisation. So spread was pushed
   to `k = n`, where every attempt carries its own device, IP and BIN and there is no
   campaign subgraph left at all. The model returned **recall 1.0 and PR-AUC 0.9913**.

   That looked like a headline and is an artifact. At `k = n` every campaign event has
   degree **zero** while background events average about 46 and 0.02% are isolated, because
   the generator draws campaign entities from a pool disjoint from the background's. "Has no
   neighbours at all" is then a perfect label proxy, free, and it is what the model reads —
   coordination is by construction not there to read. Real traffic is full of first-time
   customers on a fresh device, IP and BIN; a background where essentially no legitimate
   event is isolated cannot test this.

   The sweep is published as an **invalid measurement**, with the diagnostic columns that
   invalidate it, rather than as a favourable result. The limitation it was built to probe
   stands unmeasured, exactly as the README already said. This is defect 6 again — a
   simulation that makes the problem easy — caught this time before the number reached a
   document.

22. **A prediction that could not have failed.** The frontier's headline was 16 of 16 cells
   agreeing with `k ≥ n/τ`, presented as arithmetic derived on paper and then tested. The
   generator spreads attempts *uniformly*, so each entity carries exactly `n/k` and a counter
   trips precisely when `k ≤ n/τ`: under uniform spread the agreement is **exact by
   construction**. It is an implementation check, which is worth having and is now called
   one, not a risky prediction that survived. Defect 4 had already recorded that agreement
   only reached 100% once spread was made uniform; the README kept selling the 100%.

   Under realistic non-uniform spread the busiest entity carries more than `n/k`, so a
   counter trips at a *higher* `k` than `n/τ` predicts and the genuinely blind region is
   smaller than the dashed line — which means the frontier as drawn is generous to this
   project, not to the baseline.

   **The first attempt at this retraction did not take.** The corrected paragraph went in
   while the sentence it retracted survived two lines below it — "the dashed line is
   arithmetic, drawn before any run; every measured cell falls on the side it predicts" —
   along with the section heading, the caption on the live demo site, an entry in
   `architecture.md`, the thesis sentence in the opening, and two command comments. Seven
   places, of which a reviewer spotted three. A reader would have seen the correction and
   then the un-retracted claim beneath it, and concluded the correction was cosmetic. A
   retraction that leaves the original standing is worse than none, because it looks like
   one has been made. The check now is a grep for the retracted phrasing across every
   surface, including the generated site.

   Two documents also had the cost model charging this merchant for chargebacks on cards the
   attack validated. Those land on whichever merchant the card is later spent at, usually
   somebody else. The constant is unchanged and still a declared assumption; the reasoning
   behind it no longer claims a loss the merchant does not bear.

23. **A loader that could not run, described as one that could.** `background.py` advertised
   an IEEE-CIS path against `train_transaction.csv`. `DeviceInfo` is not in that file — it is
   in `train_identity.csv` — and `pd.read_csv(usecols=...)` drops a column it cannot find
   without complaining, so the read succeeded and the frame lookup raised a bare `KeyError`.
   `limitations.md` had recorded where `DeviceInfo` actually lives, in a different bullet,
   without anyone connecting it to the code that reads it.

   The path is now explicitly unfinished: it raises `NotImplementedError` naming the missing
   column, the file it belongs to, the ~24% row coverage a join would give a relation the
   model depends on, and `path=None` as the option every published number uses. The
   limitations bullet and the demo site both say "unfinished" instead of implying a working
   alternative that was merely declined.

   Not migrating to IEEE-CIS remains the right call and is unchanged: its native density of
   ~210 events/hour against the simulator's ~1,500 is the thin-traffic regime of defect 6,
   and the dataset carries no authorisation outcome at all, so `approved` would have to be
   synthesised from its fraud flag — the mechanism the detector leans on. The defect was
   never the decision. It was claiming a capability the code did not have.

An inference benchmark reporting p50 1.78 ms was also discarded: it was measured while a
training job ran on the same machine. The idle run is 0.91 ms.

## Claims withdrawn

Defects are one thing; a published claim that turned out to be wrong is another. These are
the six, listed so the count in the README is checkable rather than remembered.

| # | the claim | what replaced it |
|---|---|---|
| 1 | raw co-occurrence counting is a strong baseline at **0.894 PR-AUC** | 0.051 once background traffic runs at a realistic density (defect 6) |
| 2 | the heterophily gate helps, and helps more as camouflage rises | measurably net-negative at two layers; within noise at the three calibration selected (defect 8) |
| 3 | the per-relation **attention weights** show which relation carries the signal | attention says where a model looked, not what it needed; the per-relation ablation replaced it |
| 4 | per-event inference **p50 1.78 ms** | measured while a training job shared the machine; the idle run is 0.91 ms |
| 5 | incident reliability separates cleanly **at both ends** | the bottom bin is under-confident by a factor of three, on 77 of 93 incidents (defect 15) |
| 6 | memory is bounded by the window rather than by traffic history | true of the sketch, false of the streaming scorer, which retained a row per event ever seen (defect 17) |

| 7 | the frontier's **16/16** agreement is a prediction that was tested | under uniform spread it is exact by construction — an implementation check, not a risky prediction (defect 22) |

| 8 | the loader **has** an IEEE-CIS path, declined on measured grounds | the path could not run at all: `DeviceInfo` is in a file it never opened (defect 23) |

Six of the eight were found by running an experiment rather than by reading code, and every
one of them was a claim that flattered the project.


Three lessons, none about neural networks. From (1) and (4): a property you assert in a
test gets checked; a property you merely intend gets silently violated the moment an
unrelated helper changes. From (5): a preprocessing step that seems neutral can decide
your conclusion, and the dangerous ones are those whose bias points your way. From (6):
when a result looks too clean, suspect the simulation before congratulating the model. And
from (8): *an argument for a design decision is not evidence for it — the ablation you did
not run is the claim you cannot make.*
