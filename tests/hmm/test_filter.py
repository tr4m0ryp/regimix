"""Lemma L1: the vectorised forward pass equals hmmlearn's per-sequence posterior at the last position."""
import numpy as np
import pytest

from regimix.hmm import filtered_posterior, fit_segment_hmm, regime_weights


@pytest.mark.parametrize("cov", ["diag", "spherical", "full", "tied"])
@pytest.mark.parametrize("K", [1, 2, 3])
def test_equals_hmmlearn_last_position(episodes, cov, K):
    Z = episodes.Z[:400]
    hmm = fit_segment_hmm(Z, K, cov, n_iter=10, random_state=0)
    ours = filtered_posterior(hmm, Z)
    ref = np.array([hmm.predict_proba(z[:, None])[-1] for z in Z])
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-12)


def test_start_at_prior_when_emissions_are_uninformative(episodes):
    hmm = fit_segment_hmm(episodes.Z[:300], 3, n_iter=5, random_state=0)
    hmm.means_ = np.zeros((3, 1))
    hmm.covars_ = np.ones((3, 1))
    Z = episodes.Z[:50]
    A = hmm.transmat_
    prior = hmm.startprob_ @ np.linalg.matrix_power(A, Z.shape[1] - 1)
    np.testing.assert_allclose(filtered_posterior(hmm, Z), np.tile(prior, (50, 1)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(regime_weights(hmm, Z), np.tile(prior @ A, (50, 1)), rtol=0, atol=1e-15)


def test_rows_are_distributions_and_independent(episodes):
    hmm = fit_segment_hmm(episodes.Z[:500], 2, n_iter=10, random_state=0)
    w = filtered_posterior(hmm, episodes.Z[:200])
    np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(filtered_posterior(hmm, episodes.Z[5:6]), w[5:6], atol=0)


def test_extreme_segments_do_not_underflow(episodes):
    hmm = fit_segment_hmm(episodes.Z[:500], 2, n_iter=10, random_state=0)
    Z = np.full((3, 5), 1e3)
    w = filtered_posterior(hmm, Z)
    assert np.isfinite(w).all()
    np.testing.assert_allclose(w.sum(axis=1), 1)
