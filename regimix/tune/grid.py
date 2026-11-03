"""Cheap tuning: one HMM fit per (fit configuration, fold, seed); every mixture and calibrator variant is
scored on top of that fit for free, because the HMM is label-free and the variants only change what is read
from it. Inside a fold the calibrator is fitted on the training part's raw outputs.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import expit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state

from ..calibration import KINDS, apply_calibrator, fit_calibrator
from ..estimator.crossfit import fit_hmm, raw_outputs
from ..estimator.inputs import encode_labels, resolve_inputs

__all__ = ["grid_search", "DEFAULT_FIT"]

DEFAULT_FIT = dict(n_states=2, covariance_type="diag", n_iter=60, tol=1e-2, min_covar=5e-2,
                   enforce_min_covar=True)


def _logloss(p, y) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grid_search(Z, y, distance, horizon, fit_grid=None, next_step=(True, False), drift=(False, True),
                sigma_min=(0.05,), calibrations=KINDS, cv=None, groups=None, n_seeds=1, random_state=0,
                C: float = 1.0) -> list[dict]:
    """Score every (fit config x next_step x drift x sigma_min x calibration) by cross-validated log-loss.

    Parameters
    ----------
    fit_grid : list of dicts overriding :data:`DEFAULT_FIT` (HMM settings); default ``[{}]``.
    cv : splitter (default ``TimeSeriesSplit(5)``); episodes outside every test fold are not scored.

    Returns
    -------
    List of ``{"params": classifier keyword arguments, "logloss": mean fold log-loss}``, best first. The
    params can be passed straight to :class:`regimix.RegimeExceedanceClassifier`.
    """
    Z, d, h = resolve_inputs(Z, distance, horizon)
    _, y = encode_labels(y)
    splits = list((cv or TimeSeriesSplit(5)).split(Z, y, groups))
    seeds = check_random_state(random_state).randint(0, 2 ** 31 - 1, size=n_seeds)
    mixes = list(itertools.product(next_step, drift, sigma_min))
    results = []
    for fc in fit_grid or [{}]:
        hmm_kw = {**DEFAULT_FIT, **fc}
        L = {(m, k): np.zeros((n_seeds, len(y))) for m in mixes for k in calibrations}
        for si, seed in enumerate(seeds):
            for tr, va in splits:
                hmm = fit_hmm(Z[tr], hmm_kw, int(seed))
                for m in mixes:
                    mk = dict(next_step=m[0], drift=m[1], sigma_min=m[2])
                    ztr, wtr = raw_outputs(hmm, Z[tr], d[tr], h[tr], **mk)
                    zva, wva = raw_outputs(hmm, Z[va], d[va], h[va], **mk)
                    for k in calibrations:
                        cal = fit_calibrator(k, ztr, wtr, y[tr], C)
                        L[(m, k)][si, va] = apply_calibrator(k, cal, zva, wva)
        for (m, k), per in L.items():
            p = expit(per.mean(axis=0))
            score = float(np.mean([_logloss(p[va], y[va]) for _, va in splits]))
            params = {**hmm_kw, "next_step": m[0], "drift": m[1], "sigma_min": m[2], "calibration": k,
                      "C": C, "n_seeds": n_seeds}
            results.append({"params": params, "logloss": score})
    return sorted(results, key=lambda r: r["logloss"])
