import numpy as np

from regimix import RegimeExceedanceClassifier, diagnostics, make_regime_episodes


def test_collapse_is_flagged_only_without_the_floor():
    ep = make_regime_episodes(2000, scales=(0.3, 1.0), round_to=1.0, random_state=0)
    kw = dict(n_iter=100, tol=1e-4, min_covar=1e-3, calibration="none")
    stock = RegimeExceedanceClassifier(enforce_min_covar=False, **kw).fit(
        ep.Z, ep.y, distance=ep.distance, horizon=ep.horizon)
    floored = RegimeExceedanceClassifier(enforce_min_covar=True, **kw).fit(
        ep.Z, ep.y, distance=ep.distance, horizon=ep.horizon)
    rs, rf = diagnostics.report(stock, ep.Z), diagnostics.report(floored, ep.Z)
    assert rs.states[0]["collapse_ratio"] < 1 and any("collapsed" in w for w in rs.warnings)
    assert rf.states[0]["collapse_ratio"] >= 1 - 1e-9
    assert not any("collapsed" in w for w in rf.warnings)
    assert rs.zero_share.max() > 0.2


def test_healthy_fit_report(episodes):
    clf = RegimeExceedanceClassifier().fit(episodes.Z[:1500], episodes.y[:1500],
                                                    distance=episodes.distance[:1500],
                                                    horizon=episodes.horizon[:1500])
    r = diagnostics.report(clf, episodes.Z[1500:], episodes.distance[1500:], episodes.horizon[1500:],
                           episodes.y[1500:], n_repeats=1)
    st = r.states[0]
    assert st["spread_ratio"] > 2 and st["dwell"].min() > 2
    np.testing.assert_allclose(st["stationary"].sum(), 1)
    assert r.posterior[0]["distinct_share"] > 0.5
    assert r.importance.shape == (episodes.Z.shape[1],)
    assert r.importance[-1] > 0  # the most recent segment carries the regime
    assert "seed 0" in r.summary()


def test_gate_detection():
    w = np.tile([[1.0, 0.0], [0.0, 1.0]], (50, 1))
    pc = diagnostics.posterior_cardinality(w)
    assert pc["distinct"] == 2 and pc["sure_share"] == 1.0
