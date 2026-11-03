"""Per-segment permutation importance of a fitted classifier."""
from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["segment_importance"]


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def segment_importance(clf, Z, distance, horizon, y=None, n_repeats: int = 3, random_state=0) -> np.ndarray:
    """(S,) importance of each segment, oldest first.

    With labels: the mean increase in log-loss when that segment's column is shuffled across episodes.
    Without labels: the mean absolute change of the first seed's regime weights under the same shuffle.
    """
    rng = check_random_state(random_state)
    Z = np.asarray(Z, dtype=np.float64)
    if y is not None:
        y01 = (np.asarray(y) == clf.classes_[1]).astype(int)
        base = _logloss(clf.predict_proba(Z, distance, horizon)[:, 1], y01)
    else:
        w0 = clf.regime_posterior(Z)
    out = np.zeros(Z.shape[1])
    for s in range(Z.shape[1]):
        for _ in range(n_repeats):
            Zp = Z.copy()
            Zp[:, s] = Zp[rng.permutation(len(Z)), s]
            if y is not None:
                out[s] += _logloss(clf.predict_proba(Zp, distance, horizon)[:, 1], y01) - base
            else:
                out[s] += np.abs(clf.regime_posterior(Zp) - w0).sum(axis=1).mean()
        out[s] /= n_repeats
    return out
