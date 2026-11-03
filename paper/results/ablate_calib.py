"""Ablation of the out-of-fold calibration: in-sample Platt and no calibration against regimix.

Run from the repo root: python -m paper.results.ablate_calib {fx,eq,m4}
Writes paper/results/calib_<name>.json. Same task, seeds, folds and HMM fits as bench/arms/run.py; the
regimix arm is recomputed and checked against bench/results/<name>.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from bench.arms.regime import DEFAULT_FIT, DEFAULT_MIX, BlockedKFold, Fits, readout
from bench.core import cli
from bench.core.stats import logloss_vec, paired
from bench.data import equities, fx, m4
from regimix.calibration import apply_calibrator, fit_calibrator
from regimix.estimator.crossfit import raw_outputs

MAKERS = {"fx": (fx.make, "block"), "eq": (equities.make, "block"), "m4": (m4.make, "cluster")}
OUT = Path(__file__).resolve().parent


def main(name: str, seeds=(0, 1, 2)):
    make, mode = MAKERS[name]
    t0 = time.time()
    task = make(cli.DATA / f"regimix_{name}")
    tr, te = ~task.test, task.test
    Ztr, dtr, htr, ytr = task.Z[tr], task.d[tr], task.tau[tr], task.y[tr]
    Zte, dte, hte, yte = task.Z[te], task.d[te], task.tau[te], task.y[te]
    splits = list(BlockedKFold(5).split(Ztr, ytr, task.block[tr]))
    logits = {"regimix": [], "insample_platt": [], "no_calibration": []}
    for rs in seeds:
        seed = int(np.random.RandomState(rs).randint(0, 2 ** 31 - 1))
        fits = Fits(Ztr, splits, DEFAULT_FIT, seed)
        logits["regimix"].append(readout(fits, Ztr, dtr, htr, ytr, Zte, dte, hte)["logit"])
        z_in, w_in = raw_outputs(fits.final, Ztr, dtr, htr, **DEFAULT_MIX)
        z_te, w_te = raw_outputs(fits.final, Zte, dte, hte, **DEFAULT_MIX)
        cal = fit_calibrator("platt", z_in, w_in, ytr, 1.0)
        logits["insample_platt"].append(apply_calibrator("platt", cal, z_te, w_te))
        logits["no_calibration"].append(z_te)
        print(f"[{name}] seed {rs} done {time.time() - t0:.0f}s", flush=True)
    loss = {a: np.mean([logloss_vec(expit(z), yte) for z in L], axis=0) for a, L in logits.items()}
    ref = json.loads((cli.RESULTS / f"{name}.json").read_text())["arms"]["regimix"]["logloss"]
    if abs(loss["regimix"].mean() - ref) > 1e-9:
        raise AssertionError(f"regimix {loss['regimix'].mean()} differs from bench result {ref}")
    blk, clu = task.block[te], task.cluster[te]
    arms = {a: dict(logloss=float(loss[a].mean())) for a in logits}
    for a in ("insample_platt", "no_calibration"):
        arms[a]["paired_ours_minus_arm"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                            for k, v in paired(loss["regimix"] - loss[a], blk, clu, mode=mode).items()
                                            if not isinstance(v, np.ndarray)}
    out = dict(dataset=name, seeds=list(seeds), n_test=int(te.sum()), arms=arms, wall_s=round(time.time() - t0, 1))
    (OUT / f"calib_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
