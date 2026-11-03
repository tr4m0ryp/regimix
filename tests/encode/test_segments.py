import numpy as np
import pytest

from regimix import segments_from_cumulative, segments_from_path
from regimix.encode import check_edges


def test_cumulative_and_path_agree(rng):
    x = np.cumsum(rng.normal(size=500))
    edges = [0, 1, 3, 7, 15]
    t = np.arange(20, 500)
    Zp = segments_from_path(x, None, t, edges)
    cum = np.column_stack([x[t] - x[t - e] for e in edges[1:]])
    Zc = segments_from_cumulative(cum, edges[1:])
    np.testing.assert_allclose(Zp, Zc, rtol=0, atol=1e-12)


def test_oldest_first_and_unit_time_scaling():
    x = np.arange(20, dtype=float) ** 2
    Z = segments_from_path(x, None, [10], [0, 1, 4])
    # newest (9, 10]: 100 - 81 = 19 over 1; oldest (6, 9]: 81 - 36 = 45 over 3
    np.testing.assert_allclose(Z, [[45 / np.sqrt(3), 19.0]])


def test_no_lookahead(rng):
    x = np.cumsum(rng.normal(size=300))
    times = np.sort(rng.uniform(0, 300, size=300))
    t = np.array([100.5, 200.25])
    Z0 = segments_from_path(x, times, t, [0, 2, 5, 11])
    for k, tk in enumerate(t):
        y = x.copy()
        y[times > tk] += rng.normal(size=(times > tk).sum()) * 100
        np.testing.assert_array_equal(segments_from_path(y, times, t[k:k + 1], [0, 2, 5, 11]), Z0[k:k + 1])


def test_unit_time_scaling_equalises_brownian_segments(rng):
    x = np.cumsum(rng.normal(scale=1.3, size=2_000_000))
    Z = segments_from_path(x, None, np.arange(64, len(x), 64), [0, 1, 3, 7, 15, 31, 63])
    np.testing.assert_allclose(Z.var(axis=0), 1.3 ** 2, rtol=0.05)


def test_before_start_is_nan_and_clip():
    Z = segments_from_path(np.arange(10.0) * 10, None, [2, 9], [0, 1, 5])
    assert np.isnan(Z[0, 0]) and np.isfinite(Z[1]).all()
    Zc = segments_from_path(np.arange(10.0) * 10, None, [9], [0, 1, 5], clip=6)
    assert Zc.max() == 6


def test_bad_edges():
    for bad in ([1, 2], [0], [0, 3, 3], [0, 2, 1]):
        with pytest.raises(ValueError):
            check_edges(bad)
    with pytest.raises(ValueError):
        segments_from_cumulative(np.zeros((3, 2)), [1, 2, 3])
