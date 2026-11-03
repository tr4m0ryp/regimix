# Benchmark scope, cuts and results

**Verdict:** the regime mixture **beats the one-regime pooled probit on EUR/USD only** (1-minute: -0.0079 nats,
SE 0.0014; a tie on equities, where the regimes also pass the diagnostics, and a loss on M4), **ties a continuous
realised-volatility probit** on all three datasets, and **loses
to a HAR-style probit** on EUR/USD (+0.0043, SE 0.0012); it narrowly beats the HAR-style probit on M4
(-0.0007, SE 0.0003), where the realised-vol probit is the better probit and a tie. Six follow-up variants did not change that. Adding the
regime outputs to gradient boosting as extra inputs **never helped** on any dataset. This is a tie with the strong
volatility baselines, and it is reported as one.

Negative paired values mean regimix is better. Unit: log-loss in nats on the held-out tail (the last 30 % of
time, never shuffled), averaged over 3 seeds.

## 1. What ran

| | EUR/USD 1-minute | US large-cap daily | M4 hourly |
|---|---|---|---|
| script | `bench/run_fx.py` | `bench/run_equities.py` | `bench/run_m4.py` |
| source | HistData.com free ASCII M1, 2016-2023 | Yahoo Finance chart endpoint, 2005-2024, 59 tickers | M4 Hourly train + test files (GitHub) |
| task | random minute each hour; P(price in 30 min > price + u), u in {-3, 0, +3} pips | P(close in 10 days > close x (1 + u)), u in {-2 %, 0, +2 %} | P(log y in h hours > log y + u), h in {12, 24}, u = c x sd1 x sqrt(h), c in {-0.5, 0, 0.5} |
| lookback edges | 0, 1, 3, 7, 15, 31, 63 min | 0, 1, 3, 7, 15, 31 days | 0, 1, 2, 4, 8, 16, 32, 48 h |
| train / held out | 34,202 / 13,078 | 51,360 / 22,125 | 51,952 / 23,288 |
| SE method | moving-block bootstrap over 749 trading days (block length 10) | moving-block bootstrap over 1,500 dates (block length 12) | cluster bootstrap over 414 series |
| wall (2 threads in total) | 12 min + 14 min variants | 18 min + 17 min variants | 19 min + 24 min variants |

- **Split:** the held-out tail is a common date for FX and equities, and each series' own 70 % point for M4.
  Training rows whose label reads the tail are dropped, which is an embargo of one horizon.
- **Scaling:** each series is divided by the sd of its 1-step increments over the training part only, so every
  arm sees the same unit-scale segments and distances.
- **Calibration:** every arm gets a Platt map fitted on training rows. Regime arms use out-of-fold outputs from
  5 contiguous time-block folds. The `regimix` arm equals the public `RegimeExceedanceClassifier(cv=BlockedKFold(5))`
  bit for bit: the run asserts it, and `api_check` records it in each JSON.
- **Paired SE:** per-episode loss differences are summed within time blocks. Episodes in one block share a path,
  so a per-episode SE would be invalid. The bootstrap has 2,000 draws, and its block length is the first lag
  where the autocorrelation of the block means falls inside 2/sqrt(n). A Newey-West SE on block means is stored as
  a check (`hac_se_block_means`). M4 series are independent of each other, so whole series are resampled.
- **Raw predictions:** held-out logits per arm and seed go to `_data/regimix_predictions/*.npz`, outside git
  (3-5 MB each).

## 2. Cuts from the original benchmark plan, and why

| cut | planned | here | why |
|---|---|---|---|
| FX years | 2015-2024 | 2016-2023 | budget; 8 whole-year files give 47k episodes, enough for SE ~0.001 |
| equity universe | ~500 constituents, every day | 59 large caps (BK missing from the endpoint), every 4th day, staggered by ticker | budget (2.5 h on 2 threads); `yfinance` not installed, so the same Yahoo endpoint is read directly. Today's large caps with full history give **survivorship bias**, which affects every arm the same way |
| M4 episodes | every step, u = 0 | every 9th hour, u = c x sd x sqrt(h) | budget. With u = 0 the distance is 0 and every driftless arm returns exactly 1/2, so the offsets keep the task contested |
| seeds | 2 | 3 | the task asks for >= 3 |
| zero-inflated and Student-t emission arms | ablation arms | not run | the package does not ship these families (outside the two hmmlearn patches) |
| K sweep | {2, 3, 4} | 1, 2, 3 in the main run; 4 in the variants | budget |
| stock-hmmlearn arm | `GaussianHMM.fit` | `enforce_min_covar=False`: stock M-step, batched E-step | the batched fit gives hmmlearn's parameters to a relative 2.4e-13 and runs 41x (10^3 sequences) to 131-139x (10^4-10^5) faster, `bench/results/speed.json` from `python -m bench.run_speed` |
| HMM iterations | n_iter 60, tol 1e-2 | unchanged | the tolerance is on the summed log-likelihood, so at these sizes every fit runs all 60 iterations |

## 3. Main results (held-out log-loss; in brackets: regimix - arm, +- paired SE)

| arm | EUR/USD 1-min | US large-cap daily | M4 hourly |
|---|---|---|---|
| **regimix** (K = 2, floor, next step, Platt) | **0.6201** | **0.6505** | **0.5878** |
| one-regime pooled probit (K = 1, the plain reference) | 0.6280 (-0.0079 +- 0.0014) | 0.6508 (-0.0003 +- 0.0005) | 0.5849 (+0.0029 +- 0.0002) |
| realised-vol probit (whole lookback) | 0.6177 (+0.0024 +- 0.0015) | 0.6505 (+0.0000 +- 0.0008) | 0.5869 (+0.0010 +- 0.0006) |
| HAR-style probit | 0.6158 (+0.0043 +- 0.0012) | 0.6496 (+0.0008 +- 0.0005) | 0.5885 (-0.0007 +- 0.0003) |
| gate ablation (one-hot "last segment large") | 0.6276 (-0.0075 +- 0.0012) | 0.6541 (-0.0036 +- 0.0005) | 0.5910 (-0.0032 +- 0.0003) |
| logistic regression on (Z, \|Z\|, d, tau, d/sqrt tau) | 0.6288 (-0.0087 +- 0.0016) | 0.6523 (-0.0018 +- 0.0008) | 0.3476 (+0.2402 +- 0.0049) |
| HistGradientBoosting, same inputs | 0.6195 (+0.0006 +- 0.0010) | 0.6530 (-0.0025 +- 0.0008) | 0.2126 (+0.3752 +- 0.0053) |
| K = 3 | 0.6182 (+0.0019 +- 0.0006) | 0.6502 (+0.0003 +- 0.0001) | 0.5876 (+0.0002 +- 0.0001) |
| stock EM (no floor) | identical | identical | identical |
| next step off | 0.6200 (+0.0001 +- 0.0002) | 0.6509 (-0.0005 +- 0.0001) | 0.5881 (-0.0003 +- 0.0000) |
| sigma_min off | identical | identical | identical |
| drift on | 0.6201 (+0.0000 +- 0.0001) | 0.6598 (-0.0093 +- 0.0028) | 0.6907 (-0.1028 +- 0.0024) |
| isotonic calibration | 0.6201 (+0.0000 +- 0.0004) | 0.6508 (-0.0003 +- 0.0003) | 0.5802 (+0.0076 +- 0.0006) |
| per-regime Platt | 0.6199 (+0.0002 +- 0.0003) | 0.6514 (-0.0010 +- 0.0002) | 0.5858 (+0.0020 +- 0.0003) |

**Regime outputs as extra inputs: boosting + regime posterior + raw mixture logit minus boosting alone:** +0.0004
(SE 0.0004) on FX, +0.0011 (SE 0.0008) on equities, +0.0015 (SE 0.0004) on M4. Appending the regime outputs
**never helped** and hurt significantly on M4.

**Diagnostics (`regimix.diagnostics.report`, K = 2):**
- **FX:** dwell 19.6 / 8.7 segments, spread ratio 3.4. The regime conditions (dwell >= 2 segments, spread ratio >= 2) hold.
- **Equities:** dwell 14.9 / 3.9 segments, spread ratio 2.7. The conditions hold.
- **M4:** dwell 1.0 / 1.3 segments, spread ratio 1.09. The report warns on both counts, and here regimix loses to K = 1.
- **Variance floor:** it never binds on these data. With stock EM, the smallest variance is 4.8x (FX), 8.3x and
  30x the floor, so the floor arm is bit-identical. The public evidence for the floor fix stays the synthetic
  repro in `patches/hmmlearn/`.

## 4. What would have to be true for it to win, and the variants tried (`*_variants.json`)

The main run met the regime conditions on FX and equities, but the competitor that won is not a slow
volatility estimator: it reads a continuous volatility level, and our untested reading is that a two-level
regime only coarsens that level. The
variants below each test one way around that. They are scored on the same held-out tail, so a single
win at 2-3 SE among about 18 comparisons can be chance. A variant is kept only if it wins on all three datasets.

| variant | idea | EUR/USD | equities | M4 | kept |
|---|---|---|---|---|---|
| K = 4 (raw segments) | more regimes; K = 3 had beaten K = 2 | vs rv -0.0003 +- 0.0012, vs HAR +0.0016 +- 0.0010 | vs rv +0.0001 +- 0.0008, vs HAR +0.0009 +- 0.0005 | vs rv +0.0032 +- 0.0007 | no |
| rv-relative, K = 2 | divide segments and distance by the lookback realised vol, so regimes model only the change in spread around a continuous level | vs rv **-0.0002 +- 0.0001** | vs rv -0.0000 +- 0.0000 | vs rv +0.0023 +- 0.0003 | no |
| rv-relative, K = 3 | same, K = 3 | vs rv **-0.0005 +- 0.0002** | vs rv -0.0000 +- 0.0000 | vs rv +0.0060 +- 0.0005 | no |
| HAR-relative, K = 2 | same idea on the HAR forecast level | vs HAR +0.0000 +- 0.0001 | vs HAR +0.0000 +- 0.0000 | vs HAR +0.0022 +- 0.0003 | no |
| HAR-relative, K = 3 | same, K = 3 | vs HAR **-0.0001 +- 0.0000** | vs HAR +0.0000 +- 0.0000 | vs HAR +0.0011 +- 0.0002 | no |
| small n (most recent 300 / 1,000 / 3,000 training episodes) | a 7-parameter HMM against boosting and the 2-parameter probits when labels are scarce | vs rv +0.0035 / +0.0021 / +0.0032 (SE ~0.0015); vs boosting -0.028 / -0.023 / -0.016 | vs rv +0.0051 / **-0.0016** / -0.0004 (SE 0.0011 / 0.0007 / 0.0008); vs boosting -0.156 / -0.049 / -0.032 | vs rv **-0.0041 / -0.0026 / -0.0028** (SE ~0.0006) | no (see note) |

Exact values (both signs, against every reference arm) are in `fx_variants.json`, `eq_variants.json` and `m4_variants.json`.
- **Relative variants:** dividing by the level turns the regime mixture into a **near-copy of the level arm**
  (paired SE 0.00001-0.0002). On FX three of the four gain 0.0001-0.0005 nats over it, which is significant but tiny
  (HAR-relative K = 2 ties). On equities they tie (HAR-relative K = 3 is a significant 0.00003-nat loss),
  and on M4 they lose. Not kept.
- **Small n:** regimix beats boosting at every size on FX and equities, by a wide margin when n is small. It
  beats the realised-vol probit only on M4 and at one equities size. **M4 caveat:** M4 training rows are
  ordered by series, so the "most recent n" are the last series' training parts. That M4 row is a transfer
  test to other series, not a learning curve. Not kept: there is no consistent win over the probits.
- **K = 3 against K = 2:** K = 3 is better on FX and equities (FX -0.0019 +- 0.0006, equities -0.0003 +-
  0.0001) and ties on M4 (-0.0002 +- 0.0001, 95 % interval covers zero). It is the only ablation that helps on more than one dataset, but it still does not beat the
  volatility probits, and the default stays K = 2 because K = 3 was picked on the held-out tail.

**Classification (improvement discipline):** "lost a comparison", reopenable. The baseline that wins (a
continuous realised-volatility or HAR probit) meets the design constraints: it is label-light, closed-form and
calibrated. A gain would need a quantity the level misses and the regime sees: for example, regime switching
inside the horizon (Hardy 2001 sums over regime paths, and the mixture here ignores it), or longer horizons
where persistence matters more than the current level. Neither was tested here.

## 5. Reproduce

```
OMP_NUM_THREADS=2 PY=/path/to/python bench/run_all.sh     # downloads to ../_data, ~1 h wall on 2 threads
python -m bench.run_fx --seeds 3                          # one dataset
python -m bench.run_variants fx                           # its variants (needs the main run's predictions)
```
