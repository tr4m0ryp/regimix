"""Shared entry point of the per-dataset scripts: thread caps, paths, arguments, logging."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

DATA = Path(os.environ.get("REGIMIX_DATA", Path(__file__).resolve().parents[3] / "_data"))
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(name: str, make, boot_mode: str = "block", **make_kw):
    """Build the task with ``make(cache_dir, **make_kw)`` and run every arm; writes results/<name>.json."""
    from ..arms.run import run_task

    ap = argparse.ArgumentParser(description=f"regimix benchmark: {name}")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RESULTS / f"{name}.json")
    args = ap.parse_args()
    t0 = time.time()
    task = make(DATA / f"regimix_{name}", **make_kw)
    print(f"[{name}] task built in {time.time() - t0:.0f}s: {task.meta['n']} episodes, "
          f"{task.meta['n_test']} held out, positive rate {task.meta['pos_rate']:.3f}", flush=True)
    out = run_task(task, list(range(args.seeds)), args.out, pred_dir=DATA / "regimix_predictions",
                   boot_mode=boot_mode, log=lambda s: print(s, flush=True))
    for arm, r in sorted(out["arms"].items(), key=lambda kv: kv[1]["logloss"]):
        pr = r.get("paired_ours_minus_arm")
        tail = f"  ours-arm {pr['mean']:+.5f} (SE {pr['se']:.5f})" if pr else ""
        print(f"  {arm:20s} logloss {r['logloss']:.5f} brier {r['brier']:.5f} ece {r['ece']:.4f}{tail}")
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
