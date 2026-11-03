"""``RegimeExceedanceClassifier``: the scikit-learn style estimator."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ..calibration import KINDS, apply_calibrator, fit_calibrator
from ..hmm import regime_weights
from .crossfit import fit_hmm, oof_raw, raw_outputs
from .inputs import encode_labels, resolve_inputs

__all__ = ["RegimeExceedanceClassifier"]


class RegimeExceedanceClassifier(ClassifierMixin, BaseEstimator):
    """Probability that a process ends above a threshold, mixed over label-free activity regimes.

    Each episode is a row of lookback segments Z (oldest first, unit-time scaled, see
    :func:`regimix.segments_from_path`), a signed distance ``d = X(t) - threshold`` and a remaining horizon
    ``tau > 0``. A K-state Gaussian HMM is fitted on the segment sequences without labels; the filtered regime
    at the decision time (optionally advanced one step) weights K random-walk exceedance probabilities; the
    resulting logit is calibrated on out-of-fold outputs from ``cv``.

    Parameters
    ----------
    n_states : number of regimes K. ``n_states=1`` is the one-regime (pooled spread) probit.
    covariance_type, n_iter, tol, min_covar : passed to ``hmmlearn.hmm.GaussianHMM``.
    enforce_min_covar : apply ``min_covar`` after every M-step (stock hmmlearn does not; see
        :class:`regimix.hmm.FlooredGaussianHMM`).
    next_step : weight the regimes by the one-step-ahead prediction ``alpha_S A`` instead of ``alpha_S``.
    drift : use each regime's mean as a drift per unit time; off by default (driftless).
    sigma_min : floor on every regime's spread, in the units of Z.
    calibration : one of ``"none"``, ``"platt"``, ``"platt_state"``, ``"isotonic"``.
    C : inverse regularisation of the logistic calibrators.
    cv : a scikit-learn splitter, an int (number of ``TimeSeriesSplit`` folds) or None (``TimeSeriesSplit(5)``).
        Use a grouped or time-blocked splitter when episodes share a path.
    n_seeds : number of HMM restarts; their calibrated logits are averaged.
    random_state : int, RandomState or None.
    distance_col, horizon_col, segment_cols : column names when Z is a DataFrame.

    Attributes
    ----------
    hmms_ : fitted HMMs, one per seed, states sorted by increasing variance.
    calibrators_ : one calibrator per seed (None when ``calibration="none"``).
    oof_logit_ : (n_seeds, n) out-of-fold raw logits used to fit the calibrators (NaN where uncovered).
    classes_ : the two class labels; ``classes_[1]`` means "ended above the threshold".
    """

    def __init__(self, n_states=2, covariance_type="diag", n_iter=60, tol=1e-2, min_covar=5e-2,
                 enforce_min_covar=True, next_step=True, drift=False, sigma_min=0.05, calibration="platt",
                 C=1.0, cv=None, n_seeds=1, random_state=0, distance_col=None, horizon_col=None,
                 segment_cols=None):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.enforce_min_covar = enforce_min_covar
        self.next_step = next_step
        self.drift = drift
        self.sigma_min = sigma_min
        self.calibration = calibration
        self.C = C
        self.cv = cv
        self.n_seeds = n_seeds
        self.random_state = random_state
        self.distance_col = distance_col
        self.horizon_col = horizon_col
        self.segment_cols = segment_cols

    def _hmm_kw(self) -> dict:
        return dict(n_states=int(self.n_states), covariance_type=self.covariance_type, n_iter=self.n_iter,
                    tol=self.tol, min_covar=self.min_covar, enforce_min_covar=self.enforce_min_covar)

    def _mix_kw(self) -> dict:
        return dict(next_step=bool(self.next_step), drift=bool(self.drift), sigma_min=float(self.sigma_min))

    def _splitter(self):
        if self.cv is None:
            return TimeSeriesSplit(5)
        if isinstance(self.cv, (int, np.integer)):
            return TimeSeriesSplit(int(self.cv))
        return self.cv

    def _inputs(self, Z, distance, horizon):
        return resolve_inputs(Z, distance, horizon, self.distance_col, self.horizon_col, self.segment_cols)

    def fit(self, Z, y, distance=None, horizon=None, groups=None):
        """Fit the HMMs (label-free) and the out-of-fold calibrators."""
        if self.calibration not in KINDS:
            raise ValueError(f"calibration must be one of {KINDS}")
        if int(self.n_states) < 1 or int(self.n_seeds) < 1:
            raise ValueError("n_states and n_seeds must be >= 1")
        Z, d, h = self._inputs(Z, distance, horizon)
        self.classes_, y01 = encode_labels(y)
        if len(y01) != len(Z):
            raise ValueError("Z and y have different lengths")
        rng = check_random_state(self.random_state)
        self.seeds_ = [int(s) for s in rng.randint(0, 2 ** 31 - 1, size=int(self.n_seeds))]
        hmm_kw, mix_kw = self._hmm_kw(), self._mix_kw()
        splits = None
        if self.calibration != "none":
            splits = list(self._splitter().split(Z, y01, groups))
        self.hmms_, self.calibrators_, oof = [], [], []
        for seed in self.seeds_:
            cal = None
            if splits is not None:
                z, w, cov = oof_raw(Z, d, h, splits, hmm_kw, mix_kw, seed)
                if cov.sum() < 2:
                    raise ValueError("the splitter left fewer than 2 out-of-fold episodes")
                cal = fit_calibrator(self.calibration, z[cov], w[cov], y01[cov], self.C)
                oof.append(z)
            self.hmms_.append(fit_hmm(Z, hmm_kw, seed))
            self.calibrators_.append(cal)
        self.oof_logit_ = np.vstack(oof) if oof else None
        self.n_segments_ = Z.shape[1]
        return self

    def _check(self, Z):
        check_is_fitted(self, "hmms_")
        if Z.shape[1] != self.n_segments_:
            raise ValueError(f"expected {self.n_segments_} segments, got {Z.shape[1]}")

    def _per_seed(self, Z, d, h, calibrated: bool) -> np.ndarray:
        out = []
        for hmm, cal in zip(self.hmms_, self.calibrators_):
            z, w = raw_outputs(hmm, Z, d, h, **self._mix_kw())
            out.append(apply_calibrator(self.calibration, cal, z, w) if calibrated else z)
        return np.vstack(out)

    def decision_function(self, Z, distance=None, horizon=None) -> np.ndarray:
        """(n,) calibrated logit, averaged over seeds."""
        Z, d, h = self._inputs(Z, distance, horizon)
        self._check(Z)
        return self._per_seed(Z, d, h, calibrated=True).mean(axis=0)

    def predict_proba(self, Z, distance=None, horizon=None) -> np.ndarray:
        """(n, 2) class probabilities; column 1 is P(ends above the threshold)."""
        p = expit(self.decision_function(Z, distance, horizon))
        return np.column_stack([1.0 - p, p])

    def predict(self, Z, distance=None, horizon=None) -> np.ndarray:
        return self.classes_[(self.decision_function(Z, distance, horizon) > 0).astype(int)]

    def raw_logit(self, Z, distance=None, horizon=None) -> np.ndarray:
        """(n,) uncalibrated mixture logit, averaged over seeds."""
        Z, d, h = self._inputs(Z, distance, horizon)
        self._check(Z)
        return self._per_seed(Z, d, h, calibrated=False).mean(axis=0)

    def regime_posterior(self, Z, seed: int = 0, next_step: bool | None = None) -> np.ndarray:
        """(n, K) regime weights of one seed's HMM (states sorted by variance), usable as features elsewhere."""
        Z = resolve_inputs(Z, 0.0, 1.0, self.distance_col, self.horizon_col, self.segment_cols)[0]
        self._check(Z)
        ns = self.next_step if next_step is None else next_step
        return regime_weights(self.hmms_[seed], Z, next_step=bool(ns))
