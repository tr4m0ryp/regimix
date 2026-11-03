"""The regime-mixture arms: one set of HMM fits per (fit configuration, seed), every readout variant on top.

For a seed, the HMM is fitted on each calibration fold's training part (out-of-fold raw outputs for the
calibrator) and once on all training episodes (held-out outputs). The HMM never reads labels, so mixture and
calibrator variants are read from the same fits; only the fit configurations (K, the variance floor) refit.
"""
from __future__ import annotations

import numpy as np

from regimix.calibration import apply_calibrator, fit_calibrator
from regimix.estimator.crossfit import fit_hmm, raw_outputs
from regimix.hmm import regime_weights
from regimix.mixture import mixture_logit

DEFAULT_FIT = dict(n_states=2, covariance_type="diag", n_iter=60, tol=1e-2, min_covar=5e-2,
                   enforce_min_covar=True)
DEFAULT_MIX = dict(next_step=True, drift=False, sigma_min=0.05)


class BlockedKFold:
    """k contiguous folds of whole time blocks (a scikit-learn splitter; ``groups`` are the block ids)."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        u = np.unique(groups)
        for chunk in np.array_split(u, self.n_splits):
            va = np.isin(groups, chunk)
            yield np.flatnonzero(~va), np.flatnonzero(va)


class Fits:
    """HMMs of one (fit configuration, seed): one per calibration fold plus the final one."""

    def __init__(self, Ztr, splits, hmm_kw, seed):
        self.splits = splits
        self.folds = [fit_hmm(Ztr[tr], hmm_kw, seed) for tr, _ in splits]
        self.final = fit_hmm(Ztr, hmm_kw, seed)


def _oof(fits, fn, n):
    """Collect fn(hmm, train_idx, val_idx) -> (z, w) over the calibration folds."""
    z, w = np.full(n, np.nan), None
    for hmm, (tr, va) in zip(fits.folds, fits.splits):
        zv, wv = fn(hmm, tr, va)
        if w is None:
            w = np.full((n, wv.shape[1]), np.nan)
        z[va], w[va] = zv, wv
    return z, w


def readout(fits, Ztr, dtr, htr, ytr, Zte, dte, hte, mix=None, calibration="platt", C=1.0):
    """Calibrated held-out logit plus the raw outputs (out-of-fold on train, final on test), usable as extra inputs."""
    mix = {**DEFAULT_MIX, **(mix or {})}
    z_tr, w_tr = _oof(fits, lambda m, tr, va: raw_outputs(m, Ztr[va], dtr[va], htr[va], **mix), len(ytr))
    z_te, w_te = raw_outputs(fits.final, Zte, dte, hte, **mix)
    cal = fit_calibrator(calibration, z_tr, w_tr, ytr, C)
    return dict(logit=apply_calibrator(calibration, cal, z_te, w_te), z_tr=z_tr, w_tr=w_tr, z_te=z_te, w_te=w_te)


def _gate(hmm, Zfit, Zout):
    """One-hot "last segment is large": the high-spread state whenever |z_S| exceeds the threshold whose
    exceedance share on the fitting rows equals that state's stationary probability."""
    K = hmm.n_components
    A = hmm.transmat_
    vals, vecs = np.linalg.eig(A.T)
    pi = np.real(vecs[:, np.argmin(abs(vals - 1))])
    pi = pi / pi.sum()
    eps = np.quantile(abs(Zfit[:, -1]), 1 - pi[-1])
    w = np.zeros((len(Zout), K))
    hi = abs(Zout[:, -1]) > eps
    w[hi, -1] = 1.0
    w[~hi, 0] = 1.0
    return w


def gate_readout(fits, Ztr, dtr, htr, ytr, Zte, dte, hte, C=1.0):
    """Gate ablation: the regime weights replaced by a one-hot of the last segment's size."""
    mix = DEFAULT_MIX

    def fn(m, tr, va):
        w = _gate(m, Ztr[tr], Ztr[va])
        return mixture_logit(m, w, dtr[va], htr[va], drift=mix["drift"], sigma_min=mix["sigma_min"]), w

    z_tr, w_tr = _oof(fits, fn, len(ytr))
    w_te = _gate(fits.final, Ztr, Zte)
    z_te = mixture_logit(fits.final, w_te, dte, hte, drift=mix["drift"], sigma_min=mix["sigma_min"])
    cal = fit_calibrator("platt", z_tr, w_tr, ytr, C)
    return dict(logit=apply_calibrator("platt", cal, z_te, w_te))


def weights_test(fits, Zte, next_step=True):
    return regime_weights(fits.final, Zte, next_step=next_step)
