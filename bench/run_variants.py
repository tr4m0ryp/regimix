"""Variants after the main run (reads its held-out logits from the data cache).
Run from the repo root: python -m bench.run_variants <fx|eq|m4> [--seeds 3]"""
import argparse
import json
import time

from .core import cli
from .data import equities, fx, m4
from .variants.common import load_main
from .variants.plan import scaled_runs, small_n

SETS = {"fx": (fx.make, "block"), "eq": (equities.make, "block"), "m4": (m4.make, "cluster")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SETS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sizes", type=int, nargs="*", default=[300, 1000, 3000])
    args = ap.parse_args()
    t0 = time.time()
    make, mode = SETS[args.dataset]
    task = make(cli.DATA / f"regimix_{args.dataset}")
    ref = load_main(cli.DATA / "regimix_predictions", task)
    seeds = list(range(args.seeds))
    log = lambda *a: print(f"[{args.dataset} {time.time() - t0:.0f}s]", *a, flush=True)
    out = dict(dataset=task.name, seeds=seeds, paired="variant - arm (negative: variant better)",
               scaled=scaled_runs(task, ref, seeds, mode, log), small_n=small_n(task, seeds, args.sizes, mode, log))
    out["wall_s"] = round(time.time() - t0, 1)
    (cli.RESULTS / f"{args.dataset}_variants.json").write_text(json.dumps(out, indent=1))
    log("done")
