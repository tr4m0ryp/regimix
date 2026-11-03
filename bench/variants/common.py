"""Shared pieces of the variant runs: re-reading the main run's held-out logits and scoring a variant."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..arms.regime import DEFAULT_FIT, BlockedKFold, Fits, readout
from ..core.stats import logloss_vec, paired

REF_ARMS = ("regimix", "k1_pooled_probit", "k3", "rv_probit", "har_probit", "hgb", "logreg")


def load_main(pred_dir, task):
    """Held-out logits of the main run, checked against the rebuilt task."""
    z = np.load(pred_dir / f"{task.name}_test_logits.npz")
    if not np.array_equal(z["y"], task.y[task.test]):
        raise RuntimeError("the rebuilt task does not match the main run's held-out rows")
    return {a: z[a] for a in z.files if a not in ("y", "block")}


def regime_logits(Z, d, tau, y, block, tr, te, seeds, fit_kw=None, mix=None, calibration="platt"):
    """Held-out calibrated logits of the regime mixture (one row per seed) for arbitrary inputs."""
    splits = list(BlockedKFold(5).split(Z[tr], y[tr], block[tr]))
    out = []
    for rs in seeds:
        seed = int(np.random.RandomState(rs).randint(0, 2 ** 31 - 1))
        fits = Fits(Z[tr], splits, {**DEFAULT_FIT, **(fit_kw or {})}, seed)
        r = readout(fits, Z[tr], d[tr], tau[tr], y[tr], Z[te], d[te], tau[te], mix=mix, calibration=calibration)
        out.append(r["logit"])
    return np.vstack(out)


def score(name, logits, y, ref, block, cluster, mode, note=""):
    """Metrics of a variant and its paired differences against the reference arms (variant - arm)."""
    loss = np.mean([logloss_vec(expit(z), y) for z in np.atleast_2d(logits)], axis=0)
    row = dict(variant=name, note=note, logloss=float(loss.mean()), vs={})
    for a in REF_ARMS:
        if a in ref:
            la = np.mean([logloss_vec(expit(z), y) for z in np.atleast_2d(ref[a])], axis=0)
            pr = paired(loss - la, block, cluster, mode=mode)
            row["vs"][a] = dict(mean=pr["mean"], se=pr["se"], ci95=pr["ci95"])
    return row
