"""Wall time of the batched Baum-Welch against stock hmmlearn ``fit`` on many short sequences.

Run from the repo root: python -m bench.run_speed. Writes bench/results/speed.json. Both backends run the same
init, the same M-step and a fixed number of iterations (tol = 0), so the timing compares equal work; the
largest parameter and log-likelihood differences are stored beside the times.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hmmlearn  # noqa: E402
import numpy as np  # noqa: E402

from regimix import make_regime_episodes  # noqa: E402
from regimix.hmm import fit_segment_hmm  # noqa: E402

SIZES = (1_000, 10_000, 100_000)
N_ITER = 10


def one(Z, backend):
    t0 = time.perf_counter()
    h = fit_segment_hmm(Z, 2, n_iter=N_ITER, tol=0.0, backend=backend, random_state=0)
    return time.perf_counter() - t0, h


def main():
    ep = make_regime_episodes(max(SIZES), scales=(0.5, 2.0), stay=0.95, random_state=7)
    rows = []
    for n in SIZES:
        Z = ep.Z[:n]
        tn, a = one(Z, "numpy")
        th, b = one(Z, "hmmlearn")
        rel = max(float(np.max(np.abs(getattr(a, k) - getattr(b, k)) / np.maximum(np.abs(getattr(b, k)), 1e-300)))
                  for k in ("startprob_", "transmat_", "means_", "covars_"))
        ll = float(np.max(np.abs(np.subtract(list(a.monitor_.history), list(b.monitor_.history)))
                          / np.abs(list(b.monitor_.history))))
        rows.append(dict(n_sequences=n, segments=int(Z.shape[1]), n_iter=N_ITER, batched_s=round(tn, 3),
                         hmmlearn_s=round(th, 3), speedup=round(th / tn, 1), max_rel_param_diff=rel,
                         max_rel_loglik_diff=ll))
        print(rows[-1], flush=True)
    out = dict(hmmlearn=hmmlearn.__version__, numpy=np.__version__, python=platform.python_version(),
               machine=platform.processor() or platform.machine(), threads=1, rows=rows)
    path = Path(__file__).resolve().parent / "results" / "speed.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
