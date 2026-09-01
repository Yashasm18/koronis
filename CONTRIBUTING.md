# Contributing to Koronis

Thanks for taking a look. This is a small research codebase with a few firm conventions;
following them keeps the results trustworthy.

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m playwright install chromium     # for the site-render tests
.venv/bin/python site/build.py                      # the site tests need docs/index.html
.venv/bin/python -m pytest tests/ -q                # 155 tests, ~1–2 min
```

`requirements-dev.txt` is test-only. The suite runs without it — the tests that
drive the demo page skip — but then fewer tests are collected than CI collects,
and the test-count badge check will say so.

Python 3.14 is the tested version (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## How the project stays honest

Every headline claim is backed by a **test that asserts the property**, not code that
merely intends it. Several past defects (entity-ID leakage, the frontier disagreeing with
itself) were caught only because a test checked the invariant directly. If you add or
change a claim, add the assertion that would fail if it broke.

**All numbers in the README and on the demo site come from `results/`.** The experiments
write those files:

```bash
.venv/bin/python -m koronis.cli ablation      # headline detector comparison
.venv/bin/python -m koronis.cli seeds         # 10-trial intervals
.venv/bin/python -m koronis.cli mechanism     # mechanism ablation
.venv/bin/python -m koronis.cli relations     # per-relation ablation
.venv/bin/python -m koronis.cli incidents     # incidents -> forecast -> policy
.venv/bin/python -m koronis.cli drift         # traffic-profile stress test
.venv/bin/python -m koronis.cli frontier      # predicted vs measured boundary
.venv/bin/python -m koronis.cli latency
.venv/bin/python -m koronis.cli replay
.venv/bin/python -m koronis.cli benchmark
```

The demo site is regenerated, never hand-edited:

```bash
python site/build.py       # results/  ->  docs/index.html
```

If a change moves a metric, re-run the affected experiment(s), rebuild the site, and
update the matching numbers in the README **in the same PR**.

## Design constraints

- **No graph libraries.** Relational message passing is written from scratch in
  [`koronis/models/layers.py`](koronis/models/layers.py) on purpose. Do not add DGL,
  PyTorch Geometric, or similar.
- **Defence-only.** No network capability, no live payment integration, no real card or
  BIN data. The `grep` check in [`SECURITY.md`](SECURITY.md) must stay clean.
- **Backwards-in-time edges only.** A node may aggregate from its own past, never its
  future — this is what makes the streaming and latency numbers meaningful.

## Style

- Match the surrounding code: naming, comment density, and idiom.
- Keep functions small and single-purpose; large files usually mean tangled
  responsibilities.
- Run `pytest` before opening a PR. CI runs the same suite on Python 3.14.

## Commits and PRs

- Small, focused commits with a clear subject line.
- Describe what broke and how you verified the fix — the same standard the README's
  "Engineering notes" section holds itself to.
