"""Baum-Welch for many short sequences of equal length, vectorised across sequences.

hmmlearn runs its forward-backward pass one sequence at a time from Python, which costs about half a
millisecond per sequence per iteration: for 10^5 short episodes an EM iteration takes close to a minute.
Here every sequence advances together, one (n, K) array operation per position, and the sufficient statistics
are handed to hmmlearn's own ``_init`` and ``_do_mstep``. The parameters after each iteration therefore equal
stock hmmlearn's up to floating-point rounding (checked in the tests), including any subclass M-step such as
:class:`regimix.hmm.FlooredGaussianHMM`. Only one-feature Gaussian HMMs are supported.
"""
from __future__ import annotations

import numpy as np

__all__ = ["batched_estep", "fit_batched"]


def batched_estep(hmm, Z: np.ndarray) -> tuple[dict, float]:
    """Sufficient statistics and total log-likelihood of the (n, S) sequences under ``hmm``.

    The scaled forward-backward recursion, with each position's emission likelihood divided by its maximum
    over states (a per-row constant that cancels from the posteriors and is added back to the log-likelihood).
    """
    n, S = Z.shape
    K = hmm.n_components
    logb = hmm._compute_log_likelihood(Z.reshape(-1, 1)).reshape(n, S, K)
    shift = logb.max(axis=2, keepdims=True)
    b = np.exp(logb - shift)
    pi = np.asarray(hmm.startprob_, dtype=np.float64)
    A = np.asarray(hmm.transmat_, dtype=np.float64)
    alpha = np.empty((n, S, K))
    c = np.empty((n, S))
    a = pi[None, :] * b[:, 0]
    for t in range(S):
        if t:
            a = (alpha[:, t - 1] @ A) * b[:, t]
        c[:, t] = np.maximum(a.sum(axis=1), 1e-300)
        alpha[:, t] = a / c[:, t, None]
    beta = np.ones((n, S, K))
    for t in range(S - 2, -1, -1):
        beta[:, t] = ((b[:, t + 1] * beta[:, t + 1]) @ A.T) / c[:, t + 1, None]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    trans = np.zeros((K, K))
    for t in range(S - 1):
        right = b[:, t + 1] * beta[:, t + 1] / c[:, t + 1, None]
        trans += alpha[:, t].T @ right
    trans *= A
    logprob = float(np.log(c).sum() + shift.sum())
    g = gamma.reshape(-1, K)
    x = Z.reshape(-1, 1)
    stats = hmm._initialize_sufficient_statistics()
    stats["nobs"] = n
    stats["start"] = gamma[:, 0].sum(axis=0)
    if S > 1:
        stats["trans"] = trans
    stats["post"] = g.sum(axis=0)
    stats["obs"] = g.T @ x
    stats["obs**2"] = g.T @ x ** 2
    if "obs*obs.T" in stats:
        stats["obs*obs.T"] = stats["obs**2"][:, :, None].copy()
    return stats, logprob


def fit_batched(hmm, Z) -> object:
    """Fit ``hmm`` (an unfitted one-feature GaussianHMM or subclass) on the rows of Z, mirroring ``hmm.fit``."""
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    n, S = Z.shape
    X = Z.reshape(-1, 1)
    hmm._init(X, np.full(n, S))
    hmm._check()
    if hmm.n_features != 1:
        raise ValueError("the batched fitter supports one-feature HMMs only")
    hmm.monitor_._reset()
    for _ in range(hmm.n_iter):
        stats, logprob = batched_estep(hmm, Z)
        lower = hmm._compute_lower_bound(logprob)
        hmm._do_mstep(stats)
        hmm.monitor_.report(lower)
        if hmm.monitor_.converged:
            break
    return hmm
