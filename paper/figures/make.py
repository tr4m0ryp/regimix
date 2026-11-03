"""Paper figures (vector PDF), from the public benchmark results of the regimix repo.

fig1_paired.pdf       the claim: paired held-out log-loss difference (regimix - arm) with the repo's 95 % bootstrap
                      CI, per arm and dataset, plus the stacking row (boosting + regime outputs - boosting)
fig2_reliability.pdf  calibration: reliability diagram of regimix, the one-regime probit and the HAR-style probit
                      on each dataset (seed-averaged held-out logits), ECE in the legend

Usage: python figures/make.py [--results DIR] [--preds DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DS = [("fx", "eurusd_m1", "EUR/USD 1-minute"), ("eq", "equities_daily", "US large-cap daily"),
      ("m4", "m4_hourly", "M4 hourly")]
ROWS = [("k1_pooled_probit", "one-regime probit (K=1)"), ("gate_ablation", "gate ablation"),
        ("logreg", "logistic regression"), ("hgb", "gradient boosting"), ("k3", "regimix K=3"),
        ("rv_probit", "realised-vol probit"), ("har_probit", "HAR-style probit")]
# Okabe-Ito
BLUE, ORANGE, GREEN, VERM, GREY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666", "#CC79A7"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "serif"})


def fig1(results: Path, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
    labels = [r[1] for r in ROWS] + ["boosting + regime\nminus boosting"]
    ypos = np.arange(len(labels))[::-1]
    for ax, (f, _, title) in zip(axes, DS):
        js = json.loads((results / f"{f}.json").read_text())
        vals = []
        for arm, _ in ROWS:
            p = js["arms"][arm]["paired_ours_minus_arm"]
            vals.append((p["mean"], *p["ci95"]))
        s = js["arms"]["hgb_plus_regime"]["paired_stack_minus_hgb"]
        vals.append((s["mean"], *s["ci95"]))
        lim = 0.014
        ax.axvline(0, color=GREY, lw=0.8)
        for i, (y, (m, lo, hi)) in enumerate(zip(ypos, vals)):
            stack = i == len(vals) - 1
            col = PURPLE if stack else (BLUE if hi < 0 else VERM if lo > 0 else GREY)
            if abs(m) > lim:  # off-scale: annotate the value instead of drawing it
                ax.annotate(f"{m:+.3f}", xy=(np.sign(m) * lim * 0.97, y), xytext=(np.sign(m) * lim * 0.45, y),
                            va="center", ha="center", fontsize=6.5, color=col,
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
                continue
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="s" if stack else "o", ms=3.5, color=col,
                        capsize=2, lw=1)
        ax.axhline(0.5, color=GREY, lw=0.4, ls=":")
        ax.set_xlim(-lim, lim)
        ax.set_title(title)
        ax.set_xlabel("paired log-loss diff. (nats)")
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels)
    fig.text(0.99, 0.005, "regimix - arm; < 0: regimix better. Bars: 95 % block / cluster bootstrap CI.",
             ha="right", fontsize=6.5, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def reliability(p, y, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    xs, ys, ns, e = [], [], [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            xs.append(p[m].mean()), ys.append(y[m].mean()), ns.append(m.sum())
        if m.any():
            e += m.sum() * abs(p[m].mean() - y[m].mean())
    return np.array(xs), np.array(ys), np.array(ns), e / len(p)


def fig2(preds: Path, out: Path):
    arms = [("regimix", "regimix K=2", BLUE, "o"), ("k1_pooled_probit", "K=1 probit", VERM, "^"),
            ("har_probit", "HAR probit", GREEN, "s")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0))
    for ax, (_, name, title) in zip(axes, DS):
        z = np.load(preds / f"{name}_test_logits.npz")
        y = z["y"]
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--")
        for arm, lab, col, mk in arms:
            p = expit(z[arm].mean(axis=0))  # seed-averaged logit
            xs, ys, _, e = reliability(p, y)
            ax.plot(xs, ys, marker=mk, ms=3, lw=0.9, color=col, label=f"{lab} (ECE {e:.3f})")
        ax.set_title(title)
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False, handlelength=1.6, fontsize=6.5)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("observed frequency")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "bench" / "results")
    ap.add_argument("--preds", type=Path, default=REPO.parent / "_data" / "regimix_predictions")
    a = ap.parse_args()
    fig1(a.results, HERE / "fig1_paired.pdf")
    fig2(a.preds, HERE / "fig2_reliability.pdf")
    print("wrote", HERE / "fig1_paired.pdf", HERE / "fig2_reliability.pdf")


if __name__ == "__main__":
    main()
