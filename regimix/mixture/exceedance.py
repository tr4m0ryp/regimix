"""Regime-mixture probability that the process ends above a threshold.

With signed distance ``d = X(t) - k`` to the threshold k, remaining horizon ``tau`` and regime weights w,

    p_raw = sum_j w_j * Phi((d + mu_j * tau) / (max(sigma_j, sigma_min) * sqrt(tau))),

where sigma_j^2 is regime j's variance per unit time and mu_j its drift per unit time (0 unless drift is on).
If the regime stayed fixed over the horizon and X were Brownian with that drift and variance, each term would
be the exact exceedance probability; switching during the horizon is ignored (a known approximation).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..hmm.fit import state_params

__all__ = ["state_scale_drift", "mixture_probability", "mixture_logit", "EPS"]

EPS = 1e-4


def state_scale_drift(hmm) -> tuple[np.ndarray, np.ndarray]:
    """Per-state spread (sqrt of the variance per unit time) and drift (mean per unit time)."""
    mean, var = state_params(hmm)
    return np.sqrt(var), mean


def _check(distance, horizon, n: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.broadcast_to(np.asarray(distance, dtype=np.float64), (n,)).astype(np.float64)
    h = np.broadcast_to(np.asarray(horizon, dtype=np.float64), (n,)).astype(np.float64)
    if not (np.isfinite(d).all() and np.isfinite(h).all()):
        raise ValueError("distance and horizon must be finite")
    if np.any(h <= 0):
        raise ValueError("horizon must be strictly positive")
    return d, h


def mixture_probability(weights, distance, horizon, scale, drift=None, sigma_min: float = 0.0) -> np.ndarray:
    """(n,) mixture probability from explicit weights (n, K), per-state scale (K,) and drift (K,) or None."""
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    d, h = _check(distance, horizon, len(w))
    sd = np.maximum(np.asarray(scale, dtype=np.float64), sigma_min)
    if np.any(sd <= 0):
        raise ValueError("a state spread is 0; set sigma_min > 0")
    mu = np.zeros_like(sd) if drift is None else np.asarray(drift, dtype=np.float64)
    pk = ndtr((d[:, None] + mu[None, :] * h[:, None]) / (sd[None, :] * np.sqrt(h)[:, None]))
    return (w * pk).sum(axis=1)


def mixture_logit(hmm, weights, distance, horizon, drift: bool = False, sigma_min: float = 0.05) -> np.ndarray:
    """(n,) logit of the uncalibrated mixture probability, clipped to [EPS, 1 - EPS] before the logit."""
    sd, mu = state_scale_drift(hmm)
    p = mixture_probability(weights, distance, horizon, sd, mu if drift else None, sigma_min)
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))
