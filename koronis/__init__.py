"""Koronis — early detection of distributed card-testing campaigns.

Named for the Koronis asteroid family: fragments of one shattered parent body,
scattered and unremarkable across the sky, but unmistakably clustered once you
plot them in orbital-element space. Hirayama found them in 1918 by choosing the
right coordinates — which is what this project does to a card-testing campaign.
"""

import os

# LightGBM and PyTorch each ship their own OpenMP runtime. On macOS, loading
# both into one process is fragile in two distinct ways, and both bite:
#
#   1. Letting each spawn a full thread pool deadlocks — the suite hangs
#      indefinitely rather than failing.
#   2. Loading them in the *wrong order* segfaults. If torch initialises its
#      OpenMP runtime first, a later LightGBM training call crashes the
#      interpreter outright.
#
# Pinning the pool fixes (1). Importing them here, in a fixed order, fixes (2):
# whichever submodule the caller reaches for first, the runtimes are already
# initialised the same way every time. Our graphs are small enough that the
# single-thread cost is negligible.
#
# Set OMP_NUM_THREADS yourself to override the pinning.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as _lightgbm  # noqa: F401  (import order is the point)  # noqa: E402,F401  (must precede torch)
import torch as _torch  # noqa: F401  (import order is the point)  # noqa: E402,F401
