"""Non-regime arms: volatility probits and flexible learners on the same inputs.

- realised-vol probit: sigma^2 = realised variance per unit time over the whole lookback (fine increments).
- HAR-style probit: log realised variance over the horizon regressed (OLS, training rows) on log realised
  variance over each lookback window (Corsi 2009 at the segment lags); sigma from the prediction.
Both go through the same Platt calibrator, fitted on training rows.
- logistic regression and histogram gradient boosting on (Z, |Z|, d, tau, d / sqrt(tau)).
- boosting plus regime outputs: the same boosting with the regime posterior and the mixture's raw logit appended (out-of-fold on
  training rows, so the learner never sees in-sample regime outputs).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regimix.calibration import apply_calibrator, fit_calibrator

PCLIP = 1e-4


def _logit_phi(d, tau, sigma):
    p = np.clip(ndtr(d / (sigma * np.sqrt(tau))), PCLIP, 1 - PCLIP)
    return np.log(p / (1 - p))


def _platt(z_tr, y_tr, z_te):
    cal = fit_calibrator("platt", z_tr, None, y_tr)
    return apply_calibrator("platt", cal, z_te)


def rv_probit(task, tr, te, sigma_min=0.05):
    sig = np.maximum(np.sqrt(task.rv), sigma_min)
    z = _logit_phi(task.d, task.tau, sig)
    return _platt(z[tr], task.y[tr], z[te])


def har_sigma(task, tr, sigma_min=0.05):
    """Per-episode spread from the HAR regression fitted on training rows (lognormal mean of the variance)."""
    X = np.column_stack([task.har_X, np.log(task.tau)])
    reg = LinearRegression().fit(X[tr], task.har_y[tr])
    resid = task.har_y[tr] - reg.predict(X[tr])
    return np.maximum(np.sqrt(np.exp(reg.predict(X) + resid.var() / 2)), sigma_min)


def har_probit(task, tr, te, sigma_min=0.05):
    sig = har_sigma(task, tr, sigma_min)
    z = _logit_phi(task.d, task.tau, sig)
    return _platt(z[tr], task.y[tr], z[te])


def features(task):
    return np.column_stack([task.Z, abs(task.Z), task.d, task.tau, task.d / np.sqrt(task.tau)])


def logreg(task, tr, te):
    X = features(task)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], task.y[tr])
    return m.decision_function(X[te])


def hgb(X_tr, y_tr, X_te, seed):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50,
                                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                       n_iter_no_change=30, random_state=seed)
    m.fit(X_tr, y_tr)
    p = np.clip(m.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def hgb_plain(task, tr, te, seed):
    X = features(task)
    return hgb(X[tr], task.y[tr], X[te], seed)


def hgb_stack(task, tr, te, seed, ours):
    """``ours``: a regime readout dict (raw logit and weights, out-of-fold on train, final on test)."""
    X = features(task)
    Xtr = np.column_stack([X[tr], ours["z_tr"], ours["w_tr"]])
    Xte = np.column_stack([X[te], ours["z_te"], ours["w_te"]])
    return hgb(Xtr, task.y[tr], Xte, seed)
