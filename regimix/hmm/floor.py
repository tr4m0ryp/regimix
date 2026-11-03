"""A Gaussian HMM whose M-step honours ``min_covar``.

hmmlearn documents ``min_covar`` as a floor on the covariance diagonal but, up to at least 0.3.3, applies it
only when the covariances are initialised. On data with many exact zeros (rounded or discretely ticked
increments) one state then collapses onto a spike at zero with a variance orders of magnitude below the
floor, and its filtered posterior turns into a near-binary gate. ``FlooredGaussianHMM`` applies the floor after
every M-step, the same rule as the upstream patch shipped in ``patches/hmmlearn``.

For "spherical" and "diag" the M-step objective is separable and unimodal in each variance, so clipping the
unconstrained maximiser at the floor is the exact constrained maximiser and EM stays monotone on the
constrained parameter set. For "full" and "tied" the constrained maximiser of the Gaussian log-likelihood
over ``{S : S >= floor * I}`` is the eigenvalue-clipped sample covariance, applied only when an eigenvalue is
below the floor, so fits that never reach it are bit-identical to the stock class.
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

__all__ = ["FlooredGaussianHMM", "floor_covars"]


def _floor_matrix(c: np.ndarray, floor: float) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    if w.min() >= floor:
        return c
    return (v * np.maximum(w, floor)) @ v.T


def floor_covars(covars: np.ndarray, covariance_type: str, floor: float) -> np.ndarray:
    """Return ``covars`` (in hmmlearn's internal ``_covars_`` layout) floored at ``floor``."""
    covars = np.asarray(covars, dtype=np.float64)
    if floor <= 0:
        return covars
    if covariance_type in ("spherical", "diag"):
        return np.maximum(covars, floor)
    if covariance_type == "tied":
        return _floor_matrix(covars, floor)
    if covariance_type == "full":
        return np.stack([_floor_matrix(c, floor) for c in covars])
    raise ValueError(f"unknown covariance_type {covariance_type!r}")


class FlooredGaussianHMM(GaussianHMM):
    """``hmmlearn.hmm.GaussianHMM`` with ``min_covar`` enforced after every M-step."""

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if "c" in self.params:
            self._covars_ = floor_covars(self._covars_, self.covariance_type, self.min_covar)
