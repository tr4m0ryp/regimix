"""Synthetic episodes with a known answer: a random walk whose spread switches between persistent regimes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state

from ..encode import check_edges, segments_from_cumulative

__all__ = ["RegimeEpisodes", "make_regime_episodes"]


@dataclass
class RegimeEpisodes:
    """Z (n, S) segments oldest first, y (n,) labels, distance (n,), horizon (n,), regime (n,) at the decision
    time, and the generating scales and transition matrix."""

    Z: np.ndarray
    y: np.ndarray
    distance: np.ndarray
    horizon: np.ndarray
    regime: np.ndarray
    scales: np.ndarray
    transmat: np.ndarray


def make_regime_episodes(n_episodes=4000, edges=(0, 1, 3, 7, 15, 31), horizon=10, scales=(0.5, 2.0),
                         stay=0.95, distance_sd=1.0, round_to=None, random_state=None) -> RegimeEpisodes:
    """Simulate independent episodes of a regime-switching Gaussian random walk on a unit time grid.

    Each episode runs ``edges[-1]`` steps of lookback, then ``horizon`` more steps. The regime is a Markov
    chain that keeps its state with probability ``stay`` per step and otherwise jumps uniformly to another one;
    a step's increment is ``N(0, scales[regime]^2)``. The threshold sits ``distance`` below the level at the
    decision time, with ``distance ~ N(0, (distance_sd * scale_at_decision * sqrt(horizon))^2)`` so that many
    episodes are contested. ``y = 1`` when the level after the horizon is above the threshold.
    ``round_to`` rounds every increment to a grid (exact zeros become common, the variance-collapse case).
    ``horizon`` may be an int or an array of n ints.
    """
    rng = check_random_state(random_state)
    e = check_edges(edges)
    if not np.allclose(e, np.round(e)):
        raise ValueError("edges must be whole steps")
    sc = np.asarray(scales, dtype=np.float64)
    K, L = sc.size, int(e[-1])
    hz = np.broadcast_to(np.asarray(horizon, dtype=np.int64), (n_episodes,))
    T = L + int(hz.max())
    A = np.full((K, K), (1 - stay) / max(K - 1, 1))
    np.fill_diagonal(A, stay if K > 1 else 1.0)
    state = np.empty((n_episodes, T), dtype=np.int64)
    state[:, 0] = rng.randint(K, size=n_episodes)
    for t in range(1, T):
        u = rng.random_sample(n_episodes)
        jump = rng.randint(max(K - 1, 1), size=n_episodes)
        moved = (state[:, t - 1] + 1 + jump) % K if K > 1 else state[:, t - 1]
        state[:, t] = np.where(u < stay, state[:, t - 1], moved)
    inc = rng.standard_normal((n_episodes, T)) * sc[state]
    if round_to:
        inc = np.round(inc / round_to) * round_to
    level = np.concatenate([np.zeros((n_episodes, 1)), np.cumsum(inc, axis=1)], axis=1)
    now = level[:, L]
    lags = e[1:].astype(np.int64)
    cum = now[:, None] - level[:, L - lags]
    Z = segments_from_cumulative(cum, lags)
    regime = state[:, L - 1]
    d = rng.standard_normal(n_episodes) * distance_sd * sc[regime] * np.sqrt(hz)
    end = level[np.arange(n_episodes), L + hz]
    y = (end > now - d).astype(int)
    return RegimeEpisodes(Z, y, d, hz.astype(np.float64), regime, sc, A)
