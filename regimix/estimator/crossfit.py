"""Cross-fitting: out-of-fold raw logits whose HMM never saw the evaluated fold (lemma L2).

The HMM is label-free, so the labels enter only through the calibrator, which is fitted on these out-of-fold
outputs. The final HMM is then refitted on all training episodes; that leaks no label because it reads none.
"""
from __future__ import annotations

import numpy as np

from ..hmm import fit_segment_hmm, regime_weights, sort_states
from ..mixture import mixture_logit

__all__ = ["fit_hmm", "raw_outputs", "oof_raw"]


def fit_hmm(Z, hmm_kw: dict, seed: int):
    """Fit and canonically order one segment HMM."""
    return sort_states(fit_segment_hmm(Z, random_state=seed, **hmm_kw))


def raw_outputs(hmm, Z, d, h, next_step: bool, drift: bool, sigma_min: float):
    """(raw logit (n,), regime weights (n, K)) of one fitted HMM."""
    w = regime_weights(hmm, Z, next_step=next_step)
    return mixture_logit(hmm, w, d, h, drift=drift, sigma_min=sigma_min), w


def oof_raw(Z, d, h, splits, hmm_kw: dict, mix_kw: dict, seed: int):
    """Out-of-fold (z, w, covered) over the given (train, test) index pairs.

    Rows in several test folds get the mean of their outputs; rows in none are left NaN with covered False.
    """
    n, K = len(Z), hmm_kw["n_states"]
    z, w, cnt = np.zeros(n), np.zeros((n, K)), np.zeros(n)
    for tr, va in splits:
        hmm = fit_hmm(Z[tr], hmm_kw, seed)
        zv, wv = raw_outputs(hmm, Z[va], d[va], h[va], **mix_kw)
        z[va] += zv
        w[va] += wv
        cnt[va] += 1
    covered = cnt > 0
    z[covered] /= cnt[covered]
    w[covered] /= cnt[covered, None]
    z[~covered], w[~covered] = np.nan, np.nan
    return z, w, covered
