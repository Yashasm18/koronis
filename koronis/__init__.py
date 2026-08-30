"""Koronis — early detection of distributed card-testing campaigns."""

import os

# LightGBM and PyTorch each ship their own OpenMP runtime. On macOS, loading
# both into one process and letting them each spawn a full thread pool
# deadlocks: the test suite hangs indefinitely rather than failing. Pinning the
# pool to a single thread before either library loads avoids the contention.
# Our graphs are small enough that the throughput cost is negligible.
# Set OMP_NUM_THREADS yourself to override.
os.environ.setdefault("OMP_NUM_THREADS", "1")
