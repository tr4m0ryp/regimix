"""EUR/USD 1-minute bars (HistData.com free ASCII M1), one rolling-horizon episode per hour.

Task: at one random minute of every hour, will the price 30 minutes later be above the current price + u,
u drawn from {-3, 0, +3} pips? Lookback edges in minutes. The raw zips stay in the data cache, outside the repo.
"""
from __future__ import annotations

import io
import re
import time
import urllib.parse
import urllib.request
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.task import build_task

PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/{y}"
GET = "https://www.histdata.com/get.php"
EDGES = [0, 1, 3, 7, 15, 31, 63]
HORIZON = 30
PIPS = (-3, 0, 3)


def download(cache: Path, years) -> list[Path]:
    """Fetch one zip per year (skips files already present)."""
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for y in years:
        dst = cache / f"eurusd_m1_{y}.zip"
        if not dst.exists():
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
            opener.addheaders = [("User-Agent", "Mozilla/5.0")]
            page = opener.open(PAGE.format(y=y), timeout=60).read().decode("utf-8", "replace")
            tk = re.search(r'id="tk" value="([0-9a-f]+)"', page).group(1)
            body = urllib.parse.urlencode(dict(tk=tk, date=y, datemonth=y, platform="ASCII", timeframe="M1",
                                               fxpair="EURUSD")).encode()
            req = urllib.request.Request(GET, data=body, headers={"Referer": PAGE.format(y=y)})
            data = opener.open(req, timeout=300).read()
            zipfile.ZipFile(io.BytesIO(data)).testzip()
            dst.write_bytes(data)
            time.sleep(1.0)
        out.append(dst)
    return out


def load_bars(paths) -> pd.DataFrame:
    """Minute closes with times in minutes since the epoch (the file's fixed EST clock, no DST)."""
    frames = []
    for p in paths:
        with zipfile.ZipFile(p) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(z.open(name), sep=";", header=None, usecols=[0, 4], names=["ts", "close"],
                             dtype={"ts": str, "close": np.float64})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    df["minute"] = (t.astype("int64") // 60_000_000_000).astype(np.int64)
    return df.drop_duplicates("minute").sort_values("minute").reset_index(drop=True)[["minute", "close"]]


def make(cache: Path, years=range(2016, 2024), test_frac: float = 0.3, seed: int = 0):
    bars = load_bars(download(cache, years))
    times = bars["minute"].to_numpy().astype(np.float64)
    logp = np.log(bars["close"].to_numpy())
    rng = np.random.default_rng(seed)
    hours = np.unique((times // 60).astype(np.int64))
    t_dec = hours * 60.0 + rng.integers(0, 60, size=hours.size)
    t_dec = t_dec[np.isin(t_dec, times)]  # decide only at a minute with a bar
    px = np.exp(np.interp(t_dec, times, logp))
    u_pips = rng.choice(PIPS, size=t_dec.size)
    u_log = np.log1p(u_pips * 1e-4 / px)  # threshold = price + u pips, in log units
    block = (t_dec // 1440).astype(np.int64)  # trading day (file clock)
    return build_task([dict(values=logp, times=times, t=t_dec, u=u_log, block=block)], EDGES, HORIZON,
                      test_frac=test_frac, min_cover=0.9, name="eurusd_m1")
