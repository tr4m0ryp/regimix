"""Metrics and paired standard errors for dependent episodes.

Episodes in one time block share a path, so per-episode SEs are invalid. Per-episode loss differences are
summed within blocks; the SE comes from a moving-block bootstrap over the time-ordered block sums (ratio of
summed differences to summed counts, 2,000 draws), with the block length set by the autocorrelation of the
block means, and a Newey-West HAC SE as a check. With independent clusters (separate series) the bootstrap
resamples whole clusters instead.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def logloss_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(p, y, bins: int = 15) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


def metrics(p, y) -> dict:
    return dict(logloss=float(logloss_vec(p, y).mean()), brier=float(np.mean((p - y) ** 2)), ece=ece(p, y))


def _group_sums(x, g):
    _, inv = np.unique(g, return_inverse=True)  # unique ids sort ascending = time order for block ids
    return np.bincount(inv, weights=x), np.bincount(inv).astype(np.float64)


def _acf_lag(m) -> int:
    m = m - m.mean()
    nb = len(m)
    den = (m * m).sum()
    if den == 0:
        return 1
    thr = 2 / np.sqrt(nb)
    for k in range(1, min(nb // 4, 200)):
        if abs((m[k:] * m[:-k]).sum() / den) < thr:
            return k
    return min(nb // 4, 200)


def newey_west_se(m, lag: int) -> float:
    m = np.asarray(m) - np.mean(m)
    n = len(m)
    s = (m * m).sum() / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (m[k:] * m[:-k]).sum() / n
    return float(np.sqrt(max(s, 0) / n))


def paired(diff, block, cluster=None, mode: str = "block", n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean paired difference with its SE and 95 % CI. mode "block": moving-block bootstrap over time blocks;
    "cluster": resample whole clusters."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    mean = float(diff.mean())
    if mode == "cluster":
        s, c = _group_sums(diff, cluster)
        k = len(s)
        draws = rng.integers(0, k, size=(n_boot, k))
        boot = s[draws].sum(1) / c[draws].sum(1)
        out = dict(method=f"cluster bootstrap over {k} series", n_units=int(k))
        z = s / c
        lag = 0
    else:
        s, c = _group_sums(diff, block)
        nb = len(s)
        z = s / c
        L = max(_acf_lag(z), int(np.ceil(nb ** (1 / 3))))
        starts = rng.integers(0, nb - L + 1, size=(n_boot, int(np.ceil(nb / L))))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :nb]
        boot = s[idx].sum(1) / c[idx].sum(1)
        lag = L
        out = dict(method=f"moving-block bootstrap over {nb} time blocks, block length {L}", n_units=int(nb))
    se = float(boot.std(ddof=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out.update(mean=mean, se=se, ci95=[float(lo), float(hi)], n=int(len(diff)),
               z=float(mean / se) if se > 0 else None)
    if mode == "block":
        out["hac_se_block_means"] = newey_west_se(z, lag)
    return out
