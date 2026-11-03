"""Lemma L2: labels enter only through a calibrator fitted on out-of-fold outputs of label-free HMMs."""
import numpy as np
from sklearn.model_selection import KFold

from regimix import RegimeExceedanceClassifier
from regimix.estimator import fit_hmm, oof_raw, raw_outputs

HMM = dict(n_states=2, covariance_type="diag", n_iter=15, tol=1e-2, min_covar=5e-2, enforce_min_covar=True)
MIX = dict(next_step=True, drift=False, sigma_min=0.05)


def test_hmm_is_label_free(episodes):
    Z, d, h, y = episodes.Z[:800], episodes.distance[:800], episodes.horizon[:800], episodes.y[:800]
    kw = dict(n_states=2, n_iter=15, calibration="none")
    a = RegimeExceedanceClassifier(**kw).fit(Z, y, distance=d, horizon=h)
    b = RegimeExceedanceClassifier(**kw).fit(Z, np.random.default_rng(0).permutation(y), distance=d, horizon=h)
    np.testing.assert_array_equal(a.hmms_[0].means_, b.hmms_[0].means_)
    np.testing.assert_array_equal(a.raw_logit(Z, d, h), b.raw_logit(Z, d, h))


def test_oof_outputs_come_from_hmms_that_never_saw_the_fold(episodes):
    Z, d, h = episodes.Z[:600], episodes.distance[:600], episodes.horizon[:600]
    splits = list(KFold(3).split(Z))
    z, w, cov = oof_raw(Z, d, h, splits, HMM, MIX, seed=5)
    assert cov.all()
    for tr, va in splits:
        zr, wr = raw_outputs(fit_hmm(Z[tr], HMM, 5), Z[va], d[va], h[va], **MIX)
        np.testing.assert_array_equal(z[va], zr)
        np.testing.assert_array_equal(w[va], wr)
    # Corrupting one test row changes no other output of its fold: its HMM never read the fold.
    tr, va = splits[1]
    Z2 = Z.copy()
    Z2[va[0]] = 50.0
    z2, _, _ = oof_raw(Z2, d, h, [(tr, va)], HMM, MIX, seed=5)
    np.testing.assert_array_equal(z2[va[1:]], z[va[1:]])


def test_calibrator_is_fitted_on_the_stored_oof_logits(episodes):
    Z, d, h, y = episodes.Z[:900], episodes.distance[:900], episodes.horizon[:900], episodes.y[:900]
    clf = RegimeExceedanceClassifier(n_iter=15, cv=KFold(3)).fit(Z, y, distance=d, horizon=h)
    z, _, _ = oof_raw(Z, d, h, list(KFold(3).split(Z)), HMM | {"n_iter": 15}, MIX, clf.seeds_[0])
    np.testing.assert_array_equal(clf.oof_logit_[0], z)


def test_time_series_split_leaves_the_head_uncovered(episodes):
    Z, d, h, y = episodes.Z[:600], episodes.distance[:600], episodes.horizon[:600], episodes.y[:600]
    clf = RegimeExceedanceClassifier(n_iter=10).fit(Z, y, distance=d, horizon=h)
    assert np.isnan(clf.oof_logit_[0, :100]).all() and np.isfinite(clf.oof_logit_[0, -100:]).all()
