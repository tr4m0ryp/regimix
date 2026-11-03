"""The batched Baum-Welch reproduces hmmlearn's fit (same init, same M-step) up to rounding."""
import numpy as np
import pytest

from regimix.hmm import fit_segment_hmm


@pytest.mark.parametrize("cov", ["diag", "spherical", "full", "tied"])
@pytest.mark.parametrize("floor", [False, True])
def test_matches_hmmlearn_fit(episodes, cov, floor):
    Z = episodes.Z[:300]
    kw = dict(n_states=2, covariance_type=cov, n_iter=12, tol=0.0, enforce_min_covar=floor, random_state=3)
    a = fit_segment_hmm(Z, backend="numpy", **kw)
    b = fit_segment_hmm(Z, backend="hmmlearn", **kw)
    assert a.monitor_.iter == b.monitor_.iter
    for name in ("startprob_", "transmat_", "means_", "covars_"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(list(a.monitor_.history), list(b.monitor_.history), rtol=1e-12)


def test_same_convergence_stop(episodes):
    Z = episodes.Z[:300]
    a = fit_segment_hmm(Z, 2, n_iter=200, tol=1e-3, backend="numpy", random_state=0)
    b = fit_segment_hmm(Z, 2, n_iter=200, tol=1e-3, backend="hmmlearn", random_state=0)
    assert a.monitor_.iter == b.monitor_.iter < 200


def test_single_segment_sequences(episodes):
    Z = episodes.Z[:200, -1:]
    a = fit_segment_hmm(Z, 2, n_iter=5, tol=0, backend="numpy", random_state=0)
    b = fit_segment_hmm(Z, 2, n_iter=5, tol=0, backend="hmmlearn", random_state=0)
    np.testing.assert_allclose(a.means_, b.means_, rtol=1e-9)
    np.testing.assert_allclose(a.transmat_, b.transmat_, rtol=1e-9)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_segment_hmm(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        fit_segment_hmm(np.zeros((10, 3)) + np.arange(3), backend="nope")
