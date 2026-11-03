import numpy as np
import pytest
from scipy.stats import norm

from regimix.mixture import mixture_probability


def test_zero_distance_driftless_is_one_half(rng):
    w = rng.dirichlet(np.ones(3), size=20)
    p = mixture_probability(w, 0.0, rng.uniform(0.1, 10, 20), [0.3, 1.0, 4.0])
    np.testing.assert_allclose(p, 0.5, rtol=0, atol=1e-15)  # exact up to the rounding of sum(w)


def test_one_regime_is_the_random_walk_probit(rng):
    d, tau = rng.normal(size=50), rng.uniform(0.5, 20, 50)
    p = mixture_probability(np.ones((50, 1)), d, tau, [1.7])
    np.testing.assert_allclose(p, norm.cdf(d / (1.7 * np.sqrt(tau))), rtol=0, atol=1e-15)


def test_convex_combination_and_monotone_in_distance(rng):
    w = rng.dirichlet(np.ones(2), size=1)
    d = np.linspace(-5, 5, 101)
    p = mixture_probability(np.repeat(w, 101, 0), d, 4.0, [0.5, 2.0])
    lo = mixture_probability(np.tile([1, 0], (101, 1)), d, 4.0, [0.5, 2.0])
    hi = mixture_probability(np.tile([0, 1], (101, 1)), d, 4.0, [0.5, 2.0])
    assert np.all(np.diff(p) > 0)
    np.testing.assert_allclose(p, w[0, 0] * lo + w[0, 1] * hi, atol=1e-15)


def test_exact_under_fixed_regime_brownian_motion():
    rng = np.random.default_rng(1)
    sd, mu, tau, d = 1.5, 0.2, 9.0, -0.8
    end = d + mu * tau + sd * np.sqrt(tau) * rng.standard_normal(2_000_000)
    p = mixture_probability([[1.0]], d, tau, [sd], [mu])[0]
    assert abs(p - np.mean(end > 0)) < 4 * np.sqrt(p * (1 - p) / end.size)


def test_sigma_min_and_validation():
    p = mixture_probability([[1.0]], 1.0, 1.0, [0.0], sigma_min=0.5)
    np.testing.assert_allclose(p, norm.cdf(2.0))
    with pytest.raises(ValueError):
        mixture_probability([[1.0]], 1.0, 1.0, [0.0])
    with pytest.raises(ValueError):
        mixture_probability([[1.0]], 1.0, 0.0, [1.0])
