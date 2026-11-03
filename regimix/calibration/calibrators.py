"""Calibrators fitted on out-of-fold raw logits.

``none`` keeps the raw logit, ``platt`` fits ``sigmoid(a + b z)``, ``platt_state`` gives every regime but the
last its own extra slope on z (weighted by its posterior), ``isotonic`` fits a monotone map of the raw
probability. Raw logits are clipped to [-9, 9] first, so a saturated mixture cannot dominate the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = ["KINDS", "fit_calibrator", "apply_calibrator"]

KINDS = ("none", "platt", "platt_state", "isotonic")
_ZMAX = 9.0
_ISO_BOUNDS = (0.01, 0.99)


def _design(kind: str, z: np.ndarray, post: np.ndarray | None) -> np.ndarray:
    if kind == "platt_state":
        if post is None:
            raise ValueError("platt_state needs the regime posterior")
        return np.column_stack([z, *[z * post[:, k] for k in range(post.shape[1] - 1)]])
    return z[:, None]


def fit_calibrator(kind: str, z, post, y, C: float = 1.0):
    """Fit a calibrator of the given kind; returns None for ``none``."""
    if kind not in KINDS:
        raise ValueError(f"calibration must be one of {KINDS}, got {kind!r}")
    if kind == "none":
        return None
    z = np.clip(np.asarray(z, dtype=np.float64), -_ZMAX, _ZMAX)
    y = np.asarray(y).astype(int)
    if np.unique(y).size < 2:
        raise ValueError("calibration needs both classes in the out-of-fold labels")
    if kind == "isotonic":
        lo, hi = _ISO_BOUNDS
        return IsotonicRegression(out_of_bounds="clip", y_min=lo, y_max=hi).fit(expit(z), y)
    return LogisticRegression(C=C, max_iter=1000).fit(_design(kind, z, post), y)


def apply_calibrator(kind: str, cal, z, post=None) -> np.ndarray:
    """Calibrated logit of the positive class."""
    z = np.clip(np.asarray(z, dtype=np.float64), -_ZMAX, _ZMAX)
    if kind == "none":
        return z
    if kind == "isotonic":
        p = np.clip(cal.predict(expit(z)), *_ISO_BOUNDS)
        return np.log(p / (1 - p))
    return cal.decision_function(_design(kind, z, post))
