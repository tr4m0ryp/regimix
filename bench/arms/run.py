"""Run every arm on one task over several seeds and write the results JSON.

An arm's per-episode loss is its log-loss averaged over seeds (the expected loss of one seeded run); paired
differences "ours - arm" are taken on those per-episode losses. Negative means ours is better.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from regimix import RegimeExceedanceClassifier, diagnostics

from ..core.stats import logloss_vec, metrics, paired
from . import baselines as B
from .regime import DEFAULT_FIT, BlockedKFold, Fits, gate_readout, readout

FIT_ARMS = {"k2": {}, "k1": dict(n_states=1), "k3": dict(n_states=3), "k2_nofloor": dict(enforce_min_covar=False)}
READ_ARMS = {  # arm name -> (fit arm, mixture overrides, calibration)
    "regimix": ("k2", {}, "platt"),
    "k1_pooled_probit": ("k1", {}, "platt"),
    "k3": ("k3", {}, "platt"),
    "nofloor_stock_em": ("k2_nofloor", {}, "platt"),
    "next_step_off": ("k2", dict(next_step=False), "platt"),
    "sigma_min_off": ("k2", dict(sigma_min=1e-6), "platt"),
    "drift_on": ("k2", dict(drift=True), "platt"),
    "isotonic": ("k2", {}, "isotonic"),
    "platt_state": ("k2", {}, "platt_state"),
}


def _diag(fits_list, task, te, min_covar, label):
    clf = SimpleNamespace(hmms_=[f.final for f in fits_list], min_covar=min_covar, next_step=True)
    rep = diagnostics.report(clf, task.Z[te][:20000])
    st = rep.states[0]
    return dict(label=label, spread=np.sqrt(st["var"]).round(4).tolist(), dwell=np.round(st["dwell"], 3).tolist(),
                spread_ratio=float(st["spread_ratio"]), collapse_ratio=float(st["collapse_ratio"]),
                posterior=rep.posterior[0], zero_share=rep.zero_share.round(4).tolist(), warnings=rep.warnings)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def run_task(task, seeds, out_json: Path, pred_dir: Path | None = None, boot_mode="block", extra_fit_arms=None,
             extra_read_arms=None, log=print):
    t0 = time.time()
    tr, te = ~task.test, task.test
    Ztr, dtr, htr, ytr = task.Z[tr], task.d[tr], task.tau[tr], task.y[tr]
    Zte, dte, hte, yte = task.Z[te], task.d[te], task.tau[te], task.y[te]
    splits = list(BlockedKFold(5).split(Ztr, ytr, task.block[tr]))
    fit_arms = {**FIT_ARMS, **(extra_fit_arms or {})}
    read_arms = {**READ_ARMS, **(extra_read_arms or {})}
    logits = {}  # arm -> list over seeds of held-out logits
    diag, check = {}, None
    for si, rs in enumerate(seeds):
        seed = int(np.random.RandomState(rs).randint(0, 2 ** 31 - 1))  # the estimator's own seed draw
        fits = {}
        for fa, kw in fit_arms.items():
            fits[fa] = Fits(Ztr, splits, {**DEFAULT_FIT, **kw}, seed)
        log(f"[{task.name}] seed {seed}: HMM fits done {time.time() - t0:.0f}s")
        ours = None
        for arm, (fa, mix, cal) in read_arms.items():
            r = readout(fits[fa], Ztr, dtr, htr, ytr, Zte, dte, hte, mix=mix, calibration=cal)
            logits.setdefault(arm, []).append(r["logit"])
            if arm == "regimix":
                ours = r
        logits.setdefault("gate_ablation", []).append(gate_readout(fits["k2"], Ztr, dtr, htr, ytr, Zte, dte, hte)["logit"])
        if si == 0:
            diag["k2"] = _diag([fits["k2"]], task, te, DEFAULT_FIT["min_covar"], "floor on (default)")
            diag["k2_nofloor"] = _diag([fits["k2_nofloor"]], task, te, DEFAULT_FIT["min_covar"], "stock EM")
            check = _api_check(task, tr, te, rs, logits["regimix"][0])
        logits.setdefault("hgb", []).append(B.hgb_plain(task, tr, te, seed))
        logits.setdefault("hgb_plus_regime", []).append(B.hgb_stack(task, tr, te, seed, ours))
        log(f"[{task.name}] seed {seed}: all arms {time.time() - t0:.0f}s")
    for name, fn in (("rv_probit", B.rv_probit), ("har_probit", B.har_probit), ("logreg", B.logreg)):
        logits[name] = [fn(task, tr, te)] * len(seeds)  # deterministic
    loss = {a: np.mean([logloss_vec(expit(z), yte) for z in L], axis=0) for a, L in logits.items()}
    res = {}
    for a, L in logits.items():
        per_seed = [metrics(expit(z), yte) for z in L]
        res[a] = dict(logloss=float(loss[a].mean()),
                      brier=float(np.mean([m["brier"] for m in per_seed])),
                      ece=float(np.mean([m["ece"] for m in per_seed])),
                      logloss_seed_sd=float(np.std([m["logloss"] for m in per_seed])),
                      ens_logloss=metrics(expit(np.mean(L, axis=0)), yte)["logloss"])
    blk, clu = task.block[te], task.cluster[te]
    for a in res:
        if a != "regimix":
            res[a]["paired_ours_minus_arm"] = paired(loss["regimix"] - loss[a], blk, clu, mode=boot_mode)
    res["hgb_plus_regime"]["paired_stack_minus_hgb"] = paired(loss["hgb_plus_regime"] - loss["hgb"], blk, clu,
                                                              mode=boot_mode)
    out = dict(dataset=task.name, meta=task.meta, edges=task.edges, seeds=list(seeds), n_test=int(te.sum()),
               n_train=int(tr.sum()), loss="log-loss (nats), lower is better; paired mean = ours - arm",
               arms=res, diagnostics=diag, api_check=check, wall_s=round(time.time() - t0, 1))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(out), indent=1))
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_dir / f"{task.name}_test_logits.npz", y=yte, block=blk,
                            **{a: np.vstack(L) for a, L in logits.items()})
    return out


def _api_check(task, tr, te, rs, bench_logit):
    """The headline arm must equal the public estimator fitted with the same splitter and random_state."""
    clf = RegimeExceedanceClassifier(cv=BlockedKFold(5), random_state=rs)
    clf.fit(task.Z[tr], task.y[tr], distance=task.d[tr], horizon=task.tau[tr], groups=task.block[tr])
    pub = clf.decision_function(task.Z[te], distance=task.d[te], horizon=task.tau[te])
    diff = float(np.max(abs(pub - bench_logit)))
    if diff > 1e-8:
        raise AssertionError(f"bench readout differs from the public estimator by {diff}")
    return dict(note="the regimix arm equals RegimeExceedanceClassifier(cv=BlockedKFold(5), random_state=0)",
                max_abs_logit_diff=diff)
