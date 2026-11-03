"""Filtered regime probabilities at the decision time, for many short sequences at once.

The forward recursion (Hamilton 1989) gives ``alpha_S(j) = P(state_S = j | z_1..z_S)``. hmmlearn computes it
one sequence at a time; here every episode advances together, one matrix product per segment. Each step is
renormalised, and because the forward recursion is linear in alpha the per-step normaliser cancels from the
final normalised vector, so the result equals hmmlearn's ``predict_proba`` at the last position of every
sequence (lemma L1, checked to machine precision in the tests).
"""
from __future__ import annotations

import numpy as np

from .fit import state_params

__all__ = ["emission_loglik", "filtered_posterior", "regime_weights"]

_LOG2PI = np.log(2.0 * np.pi)


def emission_loglik(hmm, Z) -> np.ndarray:
    """(n, S, K) Gaussian log-density of every segment under every state."""
    Z = np.asarray(Z, dtype=np.float64)
    mean, var = state_params(hmm)
    diff = Z[:, :, None] - mean[None, None, :]
    return -0.5 * (_LOG2PI + np.log(var)[None, None, :] + diff ** 2 / var[None, None, :])


def filtered_posterior(hmm, Z) -> np.ndarray:
    """(n, K) probability of each regime at the last segment given all segments of the episode."""
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    logb = emission_loglik(hmm, Z)
    b = np.exp(logb - logb.max(axis=2, keepdims=True))
    a = np.asarray(hmm.startprob_, dtype=np.float64)[None, :] * b[:, 0]
    a /= np.maximum(a.sum(axis=1, keepdims=True), 1e-300)
    A = np.asarray(hmm.transmat_, dtype=np.float64)
    for t in range(1, Z.shape[1]):
        a = (a @ A) * b[:, t]
        a /= np.maximum(a.sum(axis=1, keepdims=True), 1e-300)
    return a


def regime_weights(hmm, Z, next_step: bool = True) -> np.ndarray:
    """Mixture weights w: the filtered posterior, optionally advanced one step by the transition matrix."""
    w = filtered_posterior(hmm, Z)
    return w @ np.asarray(hmm.transmat_, dtype=np.float64) if next_step else w
