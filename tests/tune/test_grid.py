import numpy as np
from sklearn.model_selection import KFold

import regimix.tune.grid as grid
from regimix import RegimeExceedanceClassifier, grid_search


def test_one_hmm_fit_per_fold_seed_and_config(episodes, monkeypatch):
    calls = []
    real = grid.fit_hmm
    monkeypatch.setattr(grid, "fit_hmm", lambda *a, **k: calls.append(1) or real(*a, **k))
    Z, y, d, h = episodes.Z[:600], episodes.y[:600], episodes.distance[:600], episodes.horizon[:600]
    res = grid_search(Z, y, d, h, fit_grid=[{"n_iter": 10}, {"n_iter": 10, "n_states": 3}],
                      sigma_min=(0.05, 0.2), cv=KFold(3), n_seeds=2)
    assert len(calls) == 2 * 3 * 2
    assert len(res) == 2 * (2 * 2 * 2) * 4
    assert all(np.isfinite(r["logloss"]) for r in res)
    assert res[0]["logloss"] == min(r["logloss"] for r in res)
    RegimeExceedanceClassifier(**res[0]["params"], cv=KFold(3)).fit(Z, y, distance=d, horizon=h)
