"""Fit the label-free regime model: one Gaussian HMM over every episode's segment sequence."""
from __future__ import annotations

import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .batched import fit_batched
from .floor import FlooredGaussianHMM

__all__ = ["fit_segment_hmm", "state_params"]


def fit_segment_hmm(Z, n_states: int = 2, covariance_type: str = "diag", n_iter: int = 60,
                    tol: float = 1e-2, min_covar: float = 5e-2, enforce_min_covar: bool = True,
                    init_params: str = "stmc", random_state=None, backend: str = "numpy",
                    quiet: bool = True):
    """Fit a K-state Gaussian HMM by Baum-Welch on the rows of Z, each row one sequence.

    Parameters
    ----------
    Z : array of shape (n, S), segments oldest first (see :mod:`regimix.encode`). Labels are never used.
    n_states : number of hidden regimes K.
    covariance_type, n_iter, tol, min_covar, init_params, random_state :
        passed to ``hmmlearn.hmm.GaussianHMM``.
    enforce_min_covar : apply ``min_covar`` after every M-step (:class:`FlooredGaussianHMM`); False gives the
        stock hmmlearn behaviour, which may let a variance collapse far below ``min_covar``.
    backend : "numpy" (default) runs the E-step vectorised across episodes (:mod:`regimix.hmm.batched`),
        many times faster than "hmmlearn", which calls ``hmm.fit`` (same estimates up to rounding).
    quiet : silence hmmlearn's convergence warnings.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] < 1:
        raise ValueError("Z must be a 2-D array of shape (n_episodes, n_segments)")
    if not np.isfinite(Z).all():
        raise ValueError("Z contains NaN or inf; drop or impute those episodes first")
    n, S = Z.shape
    cls = FlooredGaussianHMM if enforce_min_covar else GaussianHMM
    hmm = cls(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, tol=tol,
              min_covar=min_covar, init_params=init_params, params="stmc", random_state=random_state)
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        if backend == "numpy":
            fit_batched(hmm, Z)
        elif backend == "hmmlearn":
            hmm.fit(Z.reshape(-1, 1), lengths=np.full(n, S))
        else:
            raise ValueError("backend must be 'numpy' or 'hmmlearn'")
    return hmm


def state_params(hmm) -> tuple[np.ndarray, np.ndarray]:
    """Per-state (mean, variance) of a one-feature Gaussian HMM, whatever its covariance type."""
    K = hmm.n_components
    mean = np.asarray(hmm.means_, dtype=np.float64).reshape(K, -1)[:, 0]
    cov = np.asarray(hmm.covars_, dtype=np.float64)  # (K, 1, 1) for a single feature
    if cov.ndim != 3 or cov.shape[0] < K or cov.shape[1] != 1:
        raise ValueError("expected a one-feature Gaussian HMM")
    return mean, cov[:K, 0, 0].copy()
