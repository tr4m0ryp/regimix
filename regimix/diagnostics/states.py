"""Diagnostics of a fitted segment HMM: does it hold persistent, distinct regimes or has it collapsed?"""
from __future__ import annotations

import numpy as np

from ..hmm import regime_weights, state_params

__all__ = ["state_table", "posterior_cardinality", "zero_share"]


def state_table(hmm, min_covar: float) -> dict:
    """Per-state mean, variance, stationary share and expected dwell, plus the collapse and spread ratios.

    ``collapse_ratio = min variance / min_covar``: below 1 the documented floor was breached (stock hmmlearn).
    ``spread_ratio = max spread / min spread``: regimes that differ by less than ~2x carry little information.
    ``dwell = 1 / (1 - A_jj)`` in segments: a dwell near 1 means the "regime" is closer to per-segment noise.
    """
    mean, var = state_params(hmm)
    A = np.asarray(hmm.transmat_, dtype=np.float64)
    vals, vecs = np.linalg.eig(A.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pi / pi.sum()
    stay = np.clip(np.diag(A), 0, 1 - 1e-12)
    sd = np.sqrt(var)
    return {"mean": mean, "var": var, "stationary": pi, "dwell": 1.0 / (1.0 - stay),
            "collapse_ratio": float(var.min() / min_covar) if min_covar > 0 else np.inf,
            "spread_ratio": float(sd.max() / max(sd.min(), 1e-300))}


def posterior_cardinality(w: np.ndarray, decimals: int = 6, sure: float = 0.99) -> dict:
    """How many distinct regime weight vectors occur, and the share of episodes with a near-certain regime.

    A healthy posterior is continuous (distinct share near 1); a posterior that takes only a handful of values,
    or is almost always near-certain, acts as a binary gate rather than a graded weight.
    """
    w = np.asarray(w, dtype=np.float64)
    distinct = np.unique(np.round(w, decimals), axis=0).shape[0]
    return {"distinct": int(distinct), "distinct_share": distinct / max(len(w), 1),
            "sure_share": float(np.mean(w.max(axis=1) > sure))}


def zero_share(Z, atol: float = 1e-12) -> np.ndarray:
    """(S,) share of episodes whose segment is exactly flat (|z| <= atol), per segment."""
    return np.mean(np.abs(np.asarray(Z, dtype=np.float64)) <= atol, axis=0)


def weights_of(hmm, Z, next_step: bool) -> np.ndarray:
    return regime_weights(hmm, Z, next_step=next_step)
