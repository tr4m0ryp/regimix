"""M4 competition, hourly subset (414 series, the official train + test files concatenated).

Task: from a 48-hour history, will log y at h hours ahead (h in {12, 24}) exceed log y now + u, with
u = c x the series' training 1-step sd x sqrt(h), c drawn from {-0.5, 0, +0.5}? (A pure "above now" target
gives distance 0, where every driftless random-walk arm returns exactly 1/2, so the offsets keep the task
contested and comparable across series.) Each series is split at its own 70 % point; blocks are one day of
one series; the bootstrap resamples whole series.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.task import build_task

BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{part}/Hourly-{part_l}.csv"
EDGES = [0, 1, 2, 4, 8, 16, 32, 48]
HORIZONS = (12, 24)
C = (-0.5, 0.0, 0.5)


def download(cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for part in ("Train", "Test"):
        dst = cache / f"Hourly-{part.lower()}.csv"
        if not dst.exists():
            urllib.request.urlretrieve(BASE.format(part=part, part_l=part.lower()), dst)
        frames.append(pd.read_csv(dst, index_col=0))
    return frames


def make(cache: Path, every: int = 9, test_frac: float = 0.3, seed: int = 0):
    train, test = download(cache)
    rng = np.random.default_rng(seed)
    series = []
    for i, key in enumerate(train.index):
        y = np.concatenate([train.loc[key].dropna().to_numpy(), test.loc[key].dropna().to_numpy()])
        v = np.log(y.astype(np.float64))
        times = np.arange(v.size, dtype=np.float64)
        dv = np.diff(v[: int(0.7 * v.size)])
        sd = np.std(dv) if np.std(dv) > 0 else 1.0
        for h in HORIZONS:
            t = times[(np.arange(v.size) + i) % every == 0]
            u = rng.choice(C, size=t.size) * sd * np.sqrt(h)
            block = (i * 1000 + t // 24).astype(np.int64)
            series.append(dict(values=v, times=times, t=t, u=u, block=block, h=float(h), sid=i))
    task = build_task(series, EDGES, max(HORIZONS), test_frac=test_frac, min_cover=0.99, name="m4_hourly",
                      split="per_series")
    sid = np.array([s["sid"] for s in series])
    task.cluster = sid[task.cluster]  # both horizons of one series share a cluster
    return task
