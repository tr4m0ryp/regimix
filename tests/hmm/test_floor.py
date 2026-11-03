"""The variance floor: stock hmmlearn collapses on zero-heavy data; the floored M-step does not (lemma L3)."""
import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from regimix.hmm import FlooredGaussianHMM, fit_segment_hmm, floor_covars


def rounded(n=4000, S=7, seed=0):
    return np.round(np.random.default_rng(seed).normal(0, 0.5, (n, S)))


def test_stock_collapses_below_min_covar():
    h = fit_segment_hmm(rounded(), 2, "diag", n_iter=100, tol=1e-4, min_covar=1e-3, enforce_min_covar=False,
                        random_state=0)
    assert np.ravel(h.covars_).min() < 1e-2 * 1e-3


@pytest.mark.parametrize("cov", ["diag", "spherical", "full", "tied"])
def test_floor_holds_and_em_is_monotone(cov):
    h = fit_segment_hmm(rounded(1500), 2, cov, n_iter=100, tol=-np.inf, min_covar=1e-3,
                        enforce_min_covar=True, random_state=0)
    assert isinstance(h, FlooredGaussianHMM)
    assert np.ravel(h.covars_).min() >= 1e-3 * (1 - 1e-12)
    hist = np.asarray(h.monitor_.history)
    assert np.all(np.diff(hist) >= -1e-9 * np.abs(hist[1:]))


def test_inactive_floor_is_bit_identical(episodes):
    kw = dict(n_states=2, n_iter=15, tol=0, min_covar=1e-3, random_state=0)
    a = fit_segment_hmm(episodes.Z[:500], enforce_min_covar=True, **kw)
    b = fit_segment_hmm(episodes.Z[:500], enforce_min_covar=False, **kw)
    assert np.ravel(b.covars_).min() > 1e-3
    for name in ("means_", "covars_", "transmat_", "startprob_"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_floor_is_the_constrained_maximiser_1d(rng):
    # For one variance, the M-step objective -n/2 (log v + s/v) is unimodal with peak at s, so the
    # maximiser on [floor, inf) is max(s, floor).
    for s in (1e-6, 0.5):
        v = np.linspace(1e-3, 2, 200_001)
        best = v[np.argmax(-(np.log(v) + s / v))]
        assert abs(best - max(s, 1e-3)) < 1e-4
        np.testing.assert_allclose(floor_covars(np.array([[s]]), "diag", 1e-3), [[max(s, 1e-3)]])


def test_full_floor_clips_eigenvalues_only_when_needed(rng):
    A = rng.normal(size=(3, 3))
    c = A @ A.T + np.eye(3)
    assert floor_covars(c[None], "full", 1e-3)[0] is not None
    np.testing.assert_array_equal(floor_covars(c, "tied", 1e-3), c)
    s = np.outer([1.0, 2.0, 0.0], [1.0, 2.0, 0.0])
    f = floor_covars(s, "tied", 0.1)
    assert np.linalg.eigvalsh(f).min() >= 0.1 - 1e-12
    np.testing.assert_allclose(np.linalg.eigvalsh(f)[-1], 5.0)
    assert isinstance(FlooredGaussianHMM(2), GaussianHMM)
