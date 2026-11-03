"""The variants. Each answers one condition of "what would have to be true for the mixture to win":

- k_sweep: more regimes (K = 4) on the raw segments; K = 3 already beat K = 2 in the main run.
- rv_relative / har_relative: the strongest competitor reads a continuous volatility level. Divide the segments
  and the distance by that level (lookback realised vol, or the HAR forecast), so the HMM models only how the
  spread is changing inside the lookback (accelerating or calming) around a continuous level. Unit-free
  regime spreads then multiply the level: p = sum_j w_j Phi(d / (s_j sigma_i sqrt(tau))).
- small_n: the learning curve. With few training episodes a 2-state HMM (7 parameters) may beat boosting and
  match a 2-parameter calibrated probit; the most recent n training episodes are used.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..arms import baselines as B
from ..core.stats import logloss_vec, paired
from .common import regime_logits, score


def scaled_runs(task, ref, seeds, mode, log):
    tr, te = ~task.test, task.test
    yte, blk, clu = task.y[te], task.block[te], task.cluster[te]
    rows = []
    args = (task.y, task.block, tr, te, seeds)
    L = regime_logits(task.Z, task.d, task.tau, *args, fit_kw=dict(n_states=4))
    rows.append(score("k4", L, yte, ref, blk, clu, mode, "raw segments, K = 4"))
    log(rows[-1]["variant"], rows[-1]["logloss"])
    for lvl_name, lvl in (("rv", np.sqrt(task.rv)), ("har", B.har_sigma(task, tr))):
        Zs, ds = task.Z / lvl[:, None], task.d / lvl
        for K in (2, 3):
            L = regime_logits(Zs, ds, task.tau, *args, fit_kw=dict(n_states=K))
            rows.append(score(f"{lvl_name}_relative_k{K}", L, yte, ref, blk, clu, mode,
                              f"segments and distance divided by the {lvl_name} spread, K = {K}"))
            log(rows[-1]["variant"], rows[-1]["logloss"])
    return rows


def small_n(task, seeds, sizes, mode, log):
    """Train on the most recent n training episodes; every arm sees the same rows; held-out tail unchanged."""
    tr_all = np.flatnonzero(~task.test)
    te = task.test
    yte, blk, clu = task.y[te], task.block[te], task.cluster[te]
    rows = []
    for n in sizes:
        tr = np.zeros(len(task.y), bool)
        tr[tr_all[-n:]] = True
        sub_te = te
        L = {"regimix": regime_logits(task.Z, task.d, task.tau, task.y, task.block, tr, sub_te, seeds),
             "k1_pooled_probit": regime_logits(task.Z, task.d, task.tau, task.y, task.block, tr, sub_te, seeds,
                                               fit_kw=dict(n_states=1)),
             "rv_probit": B.rv_probit(task, tr, sub_te)[None, :],
             "har_probit": B.har_probit(task, tr, sub_te)[None, :],
             "hgb": np.vstack([B.hgb_plain(task, tr, sub_te, s) for s in seeds])}
        loss = {a: np.mean([logloss_vec(expit(z), yte) for z in v], axis=0) for a, v in L.items()}
        row = dict(n_train=int(n), logloss={a: float(v.mean()) for a, v in loss.items()}, ours_minus={})
        for a in L:
            if a != "regimix":
                pr = paired(loss["regimix"] - loss[a], blk, clu, mode=mode)
                row["ours_minus"][a] = dict(mean=pr["mean"], se=pr["se"], ci95=pr["ci95"])
        rows.append(row)
        log("small_n", n, row["logloss"])
    return rows
