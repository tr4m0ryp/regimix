import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from regimix import make_regime_episodes  # noqa: E402


@pytest.fixture(scope="session")
def episodes():
    """Persistent two-regime random walk, spreads 0.5 and 2.0, 3000 episodes."""
    return make_regime_episodes(3000, scales=(0.5, 2.0), stay=0.95, random_state=7)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
