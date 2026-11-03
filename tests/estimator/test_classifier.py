import pickle

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.metrics import log_loss

from regimix import CALIBRATIONS, RegimeExceedanceClassifier


def split(ep, n=2000):
    tr = (ep.Z[:n], ep.y[:n], ep.distance[:n], ep.horizon[:n])
    te = (ep.Z[n:], ep.y[n:], ep.distance[n:], ep.horizon[n:])
    return tr, te


@pytest.mark.parametrize("cal", CALIBRATIONS)
def test_predict_proba_is_a_distribution(episodes, cal):
    (Z, y, d, h), (Zt, yt, dt, ht) = split(episodes, 1200)
    clf = RegimeExceedanceClassifier(n_iter=15, calibration=cal).fit(Z, y, distance=d, horizon=h)
    P = clf.predict_proba(Zt, dt, ht)
    assert P.shape == (len(Zt), 2)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    assert set(np.unique(clf.predict(Zt, dt, ht))) <= {0, 1}


def test_no_calibration_returns_the_raw_mixture(episodes):
    (Z, y, d, h), _ = split(episodes, 800)
    clf = RegimeExceedanceClassifier(n_iter=10, calibration="none").fit(Z, y, distance=d, horizon=h)
    np.testing.assert_allclose(clf.predict_proba(Z, d, h)[:, 1], expit(np.clip(clf.raw_logit(Z, d, h), -9, 9)))


def test_one_state_is_the_pooled_probit(episodes):
    (Z, y, d, h), _ = split(episodes, 800)
    clf = RegimeExceedanceClassifier(n_states=1, n_iter=5, calibration="none", sigma_min=0).fit(
        Z, y, distance=d, horizon=h)
    x = Z.ravel()
    var = (1e-2 + ((x - x.mean()) ** 2).sum()) / x.size  # hmmlearn's covars_prior = 1e-2 in the M-step
    p = np.clip(norm.cdf(d / (np.sqrt(var) * np.sqrt(h))), 1e-4, 1 - 1e-4)
    np.testing.assert_allclose(clf.raw_logit(Z, d, h), np.log(p / (1 - p)), rtol=1e-9)


def test_regimes_beat_one_pooled_spread_on_regime_data(episodes):
    (Z, y, d, h), (Zt, yt, dt, ht) = split(episodes)
    loss = {}
    for K in (1, 2):
        clf = RegimeExceedanceClassifier(n_states=K, random_state=0).fit(Z, y, distance=d, horizon=h)
        loss[K] = log_loss(yt, clf.predict_proba(Zt, dt, ht)[:, 1])
    assert loss[2] < loss[1] - 0.003


def test_posterior_tracks_the_true_regime(episodes):
    (Z, y, d, h), (Zt, *_rest) = split(episodes)
    clf = RegimeExceedanceClassifier(random_state=0).fit(Z, y, distance=d, horizon=h)
    w = clf.regime_posterior(Zt, next_step=False)
    acc = np.mean((w[:, 1] > 0.5) == (episodes.regime[2000:] == 1))
    assert acc > 0.8


def test_sklearn_api_and_pickle(episodes):
    (Z, y, d, h), _ = split(episodes, 600)
    clf = RegimeExceedanceClassifier(n_iter=10, n_seeds=2, cv=3)
    assert clone(clf).get_params() == clf.get_params()
    clf.fit(Z, np.where(y == 1, "above", "below"), distance=d, horizon=h)
    assert list(clf.classes_) == ["above", "below"]
    clf2 = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(clf.decision_function(Z, d, h), clf2.decision_function(Z, d, h))
    assert len(clf.hmms_) == 2


def test_dataframe_columns(episodes):
    pd = pytest.importorskip("pandas")
    (Z, y, d, h), _ = split(episodes, 600)
    df = pd.DataFrame(Z, columns=[f"s{i}" for i in range(Z.shape[1])]).assign(dist=d, hor=h)
    a = RegimeExceedanceClassifier(n_iter=10, distance_col="dist", horizon_col="hor").fit(df, y)
    b = RegimeExceedanceClassifier(n_iter=10).fit(Z, y, distance=d, horizon=h)
    np.testing.assert_array_equal(a.predict_proba(df), b.predict_proba(Z, d, h))


def test_errors(episodes):
    (Z, y, d, h), _ = split(episodes, 300)
    with pytest.raises(ValueError):
        RegimeExceedanceClassifier(calibration="bad").fit(Z, y, distance=d, horizon=h)
    with pytest.raises(ValueError):
        RegimeExceedanceClassifier().fit(Z, y)
    with pytest.raises(ValueError):
        RegimeExceedanceClassifier().fit(Z, np.zeros(len(Z)), distance=d, horizon=h)
    clf = RegimeExceedanceClassifier(n_iter=5).fit(Z, y, distance=d, horizon=h)
    with pytest.raises(ValueError):
        clf.predict_proba(Z[:, 1:], d, h)
    with pytest.raises(ValueError):
        clf.predict_proba(Z, d, -h)
