# Contributing to Koronis

Thanks for taking a look. This is a small research codebase with a few firm conventions;
following them keeps the results trustworthy.

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m playwright install chromium     # for the site-render tests
.venv/bin/python site/build.py                      # the site tests need docs/index.html
.venv/bin/python -m pytest tests/ -q                # 268 tests, ~2 min
```

`requirements-dev.txt` is test-only. The suite runs without it — the eight tests that
drive the demo page skip — but then fewer tests are collected than CI collects, and the
test-count badge check will say so.

Python 3.14 is the tested version (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## How the project stays honest

Every headline claim is backed by a **test that asserts the property**, not code that
merely intends it. This is the whole convention, and most of it was learned by getting it
wrong: [`docs/engineering-log.md`](docs/engineering-log.md) records twenty-nine defects and
what each one cost. If you add or change a claim, add the assertion that would fail if it
broke — and check that it *does* fail, by breaking it on purpose.

That last step is not optional. Several guards here were written, verified green, and only
then discovered to be checking nothing.

Four families of check exist, and they fail in different ways:

| Guard | Catches |
|---|---|
| Defence-only (`test_defence_only.py`) | the package acquiring a way to reach outside its own process |
| Model invariants (`test_pseudonymisation_is_lossless.py`, `test_stream.py`) | the model acquiring a dependence it should not have — identifier values, a node's future |
| Figures (`test_docs_match_results.py`, `test_doc_tables_match_results.py`, `test_published_figures_are_current.py`) | published numbers drifting from `results/` — in the tables, and in the prose around them |
| The demo page (`test_site_renders.py`) | the page throwing, or rendering something the numbers do not say |
| The recording (`test_demo_recording_is_current.py`) | the README's GIF showing a console that no longer exists |

**A note on direction.** `test_docs_match_results.py` checks that each computed figure
*appears* in the docs — that can only detect a **missing** number. A superseded number is
not missing, it is extra. Three whole tables rotted in that blind spot, and then a
paragraph rotted in the gap the table guard left. If you add a figure check, ask which of
the two directions it covers.

**One check no test here performs.** Every guard in this repo compares the working tree
against itself — the docs against `results/`, the recording against the page. None of them
can tell you whether something you are describing as *pre-existing* actually predates your
change. `git log -S"<symbol>"` and `git show <ref>:<file>` answer that, and defect 24 is what
it costs when nobody asks: a feature added in response to review was recorded as already
present, on the strength of a source comment that was written in the same change. If you are
about to write "already done", read the history first.

**And measure how strong it is.** `test_published_figures_are_current.py` asserts its own
false-pass rate by sampling wrong values, because its first version accepted 63.8% of
random percentages. A check nobody has tried to fool reads as coverage while providing
none.

## Regenerating everything

**All numbers in the README and on the demo site come from `results/`.** Nothing is
transcribed by hand. The experiments that write those files are listed under
[Reproducing everything](README.md#reproducing-everything) — that list is the single copy,
deliberately not duplicated here.

The demo site and the README's screen recording are generated too, never hand-edited:

```bash
python site/build.py                              # results/ -> docs/index.html
python site/build.py && python site/record_demo.py   # + the GIF/MP4/poster; needs ffmpeg
```

`record_demo.py` refreshes `docs/assets/koronis-demo.stamp` as its last step. Do not
hand-edit the stamp — updating it without remaking the recording defeats the only check on
the image.

If a change moves a metric, re-run the affected experiment(s), rebuild the site, and update
the matching numbers **in the same PR**. If it changes the demo page at all, re-record.

## Design constraints

- **No graph libraries.** Relational message passing is written from scratch in
  [`koronis/models/layers.py`](koronis/models/layers.py) on purpose. Do not add DGL,
  PyTorch Geometric, or similar.
- **Defence-only.** No network capability, no live payment integration, no real card or
  BIN data. The `grep` check in [`SECURITY.md`](SECURITY.md) must stay clean.
- **Backwards-in-time edges only.** A node may aggregate from its own past, never its
  future — this is what makes the streaming and latency numbers meaningful, and streaming
  parity is asserted at one, two, three and four layers.
- **Identifiers are compared, never interpreted.** `device_id`, `ip_id` and `bin_id` are
  used only for equality, which is what lets an integrator send tokens instead of raw
  values and get bit-identical scores. Do not add an ordering, a hash bucket, or a learned
  embedding on an identifier. `email_domain` is the one documented exception; both the rule
  and the exception are asserted.
- **No per-entity embedding tables.** The model must stay inductive: production entities
  are ones it has never seen.

## Style

- Match the surrounding code: naming, comment density, and idiom.
- Comments explain *why*, and especially why something is not the obvious thing. A comment
  that restates the code is noise.
- Keep functions small and single-purpose; large files usually mean tangled
  responsibilities.
- Run `pytest` before opening a PR. CI runs the same suite on Python 3.14.

## Commits and PRs

- Small, focused commits with a clear subject line saying what broke, not what was touched.
- Describe how you verified the fix — the same standard the engineering log holds itself
  to. "Verified by reverting; it reports X" is the most useful sentence in a bug-fix PR.
