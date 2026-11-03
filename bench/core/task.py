"""Turn observed paths into a benchmark task with a time-ordered held-out tail.

Every series is put on a unit scale (its 1-step increment sd over its training part only), then each decision
time gives one episode: lookback segments Z (the public encoder), signed distance d = X(t) - threshold,
horizon tau, label y = 1{X(t + tau) > threshold}, plus the inputs of the volatility baselines (realised
variance per unit time over the whole lookback and over each lookback window, and, for training rows only,
the realised variance over the horizon as the HAR regression target).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from regimix import segments_from_path

RV_FLOOR = 1e-4


@dataclass
class Task:
    name: str
    Z: np.ndarray
    d: np.ndarray
    tau: np.ndarray
    y: np.ndarray
    block: np.ndarray      # time block id (bootstrap unit and calibration group), increasing in time
    cluster: np.ndarray    # series id
    test: np.ndarray       # bool, the held-out tail
    rv: np.ndarray         # realised variance per unit time over the whole lookback
    har_X: np.ndarray      # log realised variance over each lookback window
    har_y: np.ndarray      # log realised variance over the horizon (NaN on test rows; never used there)
    edges: list
    meta: dict = field(default_factory=dict)


def _idx(times, when):
    return np.searchsorted(times, when, side="right") - 1


def _series(s, edges, horizon, test_frac, min_cover, cut_time):
    times = np.asarray(s["times"], dtype=np.float64)
    v = np.asarray(s["values"], dtype=np.float64)
    t = np.asarray(s["t"], dtype=np.float64)
    h = np.broadcast_to(np.asarray(s.get("h", horizon), dtype=np.float64), t.shape).copy()
    u = np.asarray(s["u"], dtype=np.float64)
    block = np.asarray(s["block"])
    eS = float(edges[-1])
    ok = (t - eS >= times[0]) & (t + h <= times[-1])
    cnt = _idx(times, t + h) - _idx(times, t - eS)
    ok &= cnt >= min_cover * (eS + h)
    if cut_time is None:  # per-series split: the last test_frac of this series' time
        cut_time = times[0] + (1 - test_frac) * (times[-1] - times[0])
    test = t >= cut_time
    ok &= test | (t + h < cut_time)  # embargo: a training label never reads the held-out tail
    t, h, u, block, test = t[ok], h[ok], u[ok], block[ok], test[ok]
    dt = np.diff(times)
    dv = np.diff(v)
    unit = dt == dt.min()
    scale = np.std(dv[unit & (times[1:] < cut_time)])
    x = v / scale
    sq = np.concatenate([[0.0], np.where(dt <= 5 * dt.min(), np.diff(x) ** 2, 0.0)]).cumsum()

    def rv(a, b):
        return np.maximum((sq[_idx(times, b)] - sq[_idx(times, a)]) / (b - a), RV_FLOOR)

    Z = segments_from_path(x, times, t, edges)
    level_now = x[_idx(times, t)]
    level_end = x[_idx(times, t + h)]
    thr = level_now + u / scale
    har_X = np.column_stack([np.log(rv(t - w, t)) for w in edges[1:]])
    har_y = np.where(test, np.nan, np.log(rv(t, t + h)))
    return dict(Z=Z, d=level_now - thr, tau=h, y=(level_end > thr).astype(int), block=block, test=test,
                rv=rv(t - eS, t), har_X=har_X, har_y=har_y)


def build_task(series, edges, horizon, test_frac=0.3, min_cover=0.9, name="task", split="global"):
    """series: list of dicts with values, times, t (decision times), u (threshold offset, value units), block,
    optional h (per-episode horizon). split="global" cuts every series at one common time (shared calendar);
    "per_series" cuts each series at its own (1 - test_frac) point."""
    cut = None
    if split == "global":
        lo = min(np.min(s["t"]) for s in series)
        hi = max(np.max(s["t"]) for s in series)
        cut = lo + (1 - test_frac) * (hi - lo)
    parts = []
    for i, s in enumerate(series):
        p = _series(s, edges, horizon, test_frac, min_cover, cut)
        p["cluster"] = np.full(len(p["y"]), i)
        parts.append(p)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = np.isfinite(cat["Z"]).all(1)
    cat = {k: v[keep] for k, v in cat.items()}
    order = np.lexsort((cat["cluster"], cat["block"]))  # time-ordered by block
    cat = {k: v[order] for k, v in cat.items()}
    meta = dict(n=int(len(cat["y"])), n_test=int(cat["test"].sum()), n_series=len(series),
                pos_rate=float(cat["y"].mean()), split=split, test_frac=test_frac, horizon=horizon)
    return Task(name=name, edges=list(edges), meta=meta, **cat)
