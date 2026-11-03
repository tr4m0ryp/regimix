"""regimix: calibrated threshold-exceedance probabilities from a partial path, mixed over label-free
hidden Markov activity regimes.

Typical use::

    from regimix import RegimeExceedanceClassifier, segments_from_path
    Z = segments_from_path(values, None, decision_idx, edges=[0, 1, 3, 7, 15])
    clf = RegimeExceedanceClassifier(n_states=2).fit(Z, y, distance=d, horizon=tau)
    p = clf.predict_proba(Z_new, distance=d_new, horizon=tau_new)[:, 1]
"""
from . import calibration, datasets, diagnostics, encode, hmm, mixture, tune
from .calibration import KINDS as CALIBRATIONS
from .datasets import make_regime_episodes
from .encode import segments_from_cumulative, segments_from_path
from .estimator import RegimeExceedanceClassifier
from .hmm import FlooredGaussianHMM, filtered_posterior, fit_segment_hmm, regime_weights
from .mixture import mixture_probability
from .tune import grid_search

__version__ = "0.1.0"

__all__ = ["CALIBRATIONS", "FlooredGaussianHMM", "RegimeExceedanceClassifier", "calibration", "datasets",
           "diagnostics", "encode", "filtered_posterior", "fit_segment_hmm", "grid_search", "hmm",
           "make_regime_episodes", "mixture", "mixture_probability", "regime_weights",
           "segments_from_cumulative", "segments_from_path", "tune"]
