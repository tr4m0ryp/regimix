"""Quickstart on synthetic data with a known answer.

A random walk switches between a calm and an active regime. At each decision time we see the last 31 steps,
encode them as five unit-time scaled segments and ask: will the walk be above a threshold 10 steps from now?
The script compares one pooled spread (n_states=1) with two regimes, prints the diagnostics, and shows the
same encoding applied to a single long path with ``segments_from_path``.

Run: python examples/quickstart.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from regimix import RegimeExceedanceClassifier, diagnostics, make_regime_episodes, segments_from_path  # noqa: E402


def episodes_demo():
    ep = make_regime_episodes(8000, edges=(0, 1, 3, 7, 15, 31), horizon=10, scales=(0.5, 2.0), stay=0.95,
                              random_state=0)
    n = 6000  # fit on the first 6000 episodes, evaluate on the last 2000
    tr = dict(distance=ep.distance[:n], horizon=ep.horizon[:n])
    te = dict(distance=ep.distance[n:], horizon=ep.horizon[n:])
    for K in (1, 2):
        clf = RegimeExceedanceClassifier(n_states=K, random_state=0).fit(ep.Z[:n], ep.y[:n], **tr)
        p = clf.predict_proba(ep.Z[n:], **te)[:, 1]
        print(f"n_states={K}: log-loss {log_loss(ep.y[n:], p):.4f}  Brier {brier_score_loss(ep.y[n:], p):.4f}")
    rep = diagnostics.report(clf, ep.Z[n:], y=ep.y[n:], n_repeats=1, **te)
    print(rep.summary())
    w = clf.regime_posterior(ep.Z[n:], next_step=False)
    print(f"regime recovered for {np.mean((w[:, 1] > 0.5) == (ep.regime[n:] == 1)):.1%} of held-out episodes")


def path_demo():
    rng = np.random.default_rng(1)
    regime = np.zeros(60_000, dtype=int)
    for t in range(1, regime.size):  # one long path whose spread switches now and then
        regime[t] = regime[t - 1] if rng.random() < 0.98 else 1 - regime[t - 1]
    x = np.cumsum(rng.standard_normal(regime.size) * np.where(regime == 1, 2.0, 0.5))
    edges, h = [0, 1, 3, 7, 15, 31], 10
    t = np.arange(31, x.size - h, 7)  # decision times
    Z = segments_from_path(x, None, t, edges)
    d = rng.normal(0, 4, t.size)  # signed distance: the threshold sits d below the current level
    y = (x[t + h] > x[t] - d).astype(int)
    cut = int(0.7 * t.size)  # time-ordered split: never shuffle episodes from one path
    for K in (1, 2):
        clf = RegimeExceedanceClassifier(n_states=K).fit(Z[:cut], y[:cut], distance=d[:cut], horizon=h)
        p = clf.predict_proba(Z[cut:], distance=d[cut:], horizon=h)[:, 1]
        print(f"single path, n_states={K}: log-loss {log_loss(y[cut:], p):.4f}")
    w = clf.regime_posterior(Z[cut:])
    print(f"mean active-regime weight when active {w[regime[t[cut:]] == 1, 1].mean():.2f}, "
          f"when calm {w[regime[t[cut:]] == 0, 1].mean():.2f}")


if __name__ == "__main__":
    episodes_demo()
    path_demo()
