"""Canonical state order, so regime 0 is always the calmest one."""
from __future__ import annotations

import numpy as np

from .fit import state_params

__all__ = ["sort_states"]


def sort_states(hmm):
    """Relabel the states of a fitted one-feature Gaussian HMM in order of increasing variance (in place).

    The likelihood is invariant to the relabelling; it only makes posteriors comparable across seeds and
    folds. Ties are broken by the state mean.
    """
    mean, var = state_params(hmm)
    order = np.lexsort((mean, var))
    if np.array_equal(order, np.arange(len(order))):
        return hmm
    hmm.startprob_ = np.asarray(hmm.startprob_)[order]
    hmm.transmat_ = np.asarray(hmm.transmat_)[np.ix_(order, order)]
    hmm.means_ = np.asarray(hmm.means_)[order]
    if hmm.covariance_type != "tied":
        hmm._covars_ = np.asarray(hmm._covars_)[order]
    return hmm
