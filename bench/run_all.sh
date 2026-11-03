#!/usr/bin/env bash
# Every benchmark and its variants, two processes at a time with one thread each (2 threads in total).
# Data download to $REGIMIX_DATA (default ../_data next to the repo); results to bench/results/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p bench/results/logs
run() { local mod=$1 arg=$2 log=$3; $PY -m "bench.$mod" $arg > "bench/results/logs/$log.log" 2>&1; }
( run run_fx "" fx; run run_m4 "" m4; run run_variants m4 m4_variants ) &
( run run_equities "" eq; run run_variants fx fx_variants; run run_variants eq eq_variants ) &
wait
