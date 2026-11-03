"""Large US stocks, daily adjusted closes (Yahoo Finance chart endpoint), 2005-2024.

Task: will the close 10 trading days ahead exceed today's close x (1 + u), u drawn from {-2 %, 0, +2 %}?
Lookback edges in trading days. Episodes are grouped by date (the bootstrap block), the split is one common
date. Tickers are today's large constituents with full history, so the panel has survivorship bias; that
affects every arm equally and is recorded in SCOPE.md.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.task import build_task

URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1=1104537600&period2=1735689600"
       "&interval=1d&events=div%2Csplits")
TICKERS = ("AAPL MSFT AMZN JPM JNJ XOM PG KO PEP WMT INTC CSCO ORCL IBM T VZ PFE MRK ABT MCD HD LOW NKE DIS BA "
           "CAT DE MMM HON GE UNP UPS FDX CVX COP SLB OXY BAC C WFC GS MS AXP USB BK CL KMB GIS HSY MO LLY BMY "
           "AMGN GILD MDT UNH CVS TGT COST SBUX").split()
EDGES = [0, 1, 3, 7, 15, 31]
HORIZON = 10
U = (-0.02, 0.0, 0.02)


def download(cache: Path, tickers=TICKERS) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for tk in tickers:
        dst = cache / f"{tk}.json"
        for attempt in range(4):
            if dst.exists():
                break
            try:
                req = urllib.request.Request(URL.format(tk=tk), headers={"User-Agent": "Mozilla/5.0"})
                dst.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            except OSError:
                time.sleep(5.0 * (attempt + 1))
            time.sleep(0.5)
        if not dst.exists():
            continue  # unavailable ticker: dropped, and listed by make()
        r = json.loads(dst.read_text())["chart"]["result"][0]
        adj = np.asarray(r["indicators"]["adjclose"][0]["adjclose"], dtype=np.float64)
        day = pd.to_datetime(np.asarray(r["timestamp"]), unit="s").normalize()
        s = pd.Series(adj, index=day).dropna()
        out[tk] = s[~s.index.duplicated()]
    return out


def make(cache: Path, every: int = 4, test_frac: float = 0.3, seed: int = 0):
    """``every``: one decision day in ``every`` per ticker (staggered by ticker) to bound the episode count."""
    data = download(cache)
    missing = sorted(set(TICKERS) - set(data))
    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in data.values()])))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    rng = np.random.default_rng(seed)
    series = []
    for i, (tk, s) in enumerate(sorted(data.items())):
        s = s[s > 0]
        times = pos[s.index].to_numpy().astype(np.float64)  # trading-day index on the common calendar
        t = times[(np.arange(times.size) + i) % every == 0]
        u = np.log1p(rng.choice(U, size=t.size))
        series.append(dict(values=np.log(s.to_numpy()), times=times, t=t, u=u, block=t.astype(np.int64)))
    task = build_task(series, EDGES, HORIZON, test_frac=test_frac, min_cover=0.95, name="equities_daily")
    task.meta.update(tickers=sorted(data), missing_tickers=missing)
    return task
