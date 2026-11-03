"""One call that checks the failure modes of the method on your data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .importance import segment_importance
from .states import posterior_cardinality, state_table, weights_of, zero_share

__all__ = ["DiagnosticReport", "report"]


@dataclass
class DiagnosticReport:
    """Per-seed state tables and posterior shape, per-segment zero share and importance, and warnings."""

    states: list
    posterior: list
    zero_share: np.ndarray
    importance: np.ndarray | None
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        for i, (st, po) in enumerate(zip(self.states, self.posterior)):
            lines.append(f"seed {i}: spread {np.round(np.sqrt(st['var']), 4).tolist()} "
                         f"dwell {np.round(st['dwell'], 2).tolist()} spread_ratio {st['spread_ratio']:.2f} "
                         f"collapse_ratio {st['collapse_ratio']:.3g} distinct_share {po['distinct_share']:.3f} "
                         f"sure_share {po['sure_share']:.3f}")
        lines.append(f"zero share per segment: {np.round(self.zero_share, 3).tolist()}")
        if self.importance is not None:
            lines.append(f"segment importance: {np.round(self.importance, 5).tolist()}")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def report(clf, Z, distance=None, horizon=None, y=None, zero_atol: float = 1e-12, n_repeats: int = 3,
           random_state=0) -> DiagnosticReport:
    """Diagnose a fitted :class:`regimix.RegimeExceedanceClassifier` on segments Z.

    Checks: variance collapse (``collapse_ratio < 1``), a gate-like posterior (few distinct weight vectors or
    most episodes near-certain), short dwell (< 2 segments), weak regime contrast (spread ratio < 2) and flat
    segments. Segment importance is computed when ``distance`` and ``horizon`` are given.
    """
    Z = np.asarray(Z, dtype=np.float64)
    states, post, warns = [], [], []
    for i, hmm in enumerate(clf.hmms_):
        st = state_table(hmm, clf.min_covar)
        po = posterior_cardinality(weights_of(hmm, Z, bool(clf.next_step)))
        states.append(st)
        post.append(po)
        if hmm.n_components < 2:
            continue
        if st["collapse_ratio"] < 1:
            warns.append(f"seed {i}: a state variance is {st['collapse_ratio']:.2g}x min_covar (collapsed); "
                         "use enforce_min_covar=True")
        if po["distinct_share"] < 0.05 or po["sure_share"] > 0.9:
            warns.append(f"seed {i}: the regime posterior behaves like a gate "
                         f"(distinct share {po['distinct_share']:.3f}, sure share {po['sure_share']:.3f})")
        if st["dwell"].min() < 2:
            warns.append(f"seed {i}: a regime lasts {st['dwell'].min():.2f} segments on average (< 2)")
        if st["spread_ratio"] < 2:
            warns.append(f"seed {i}: regime spreads differ by only {st['spread_ratio']:.2f}x (< 2)")
    zs = zero_share(Z, zero_atol)
    if zs.max() > 0.2:
        warns.append(f"up to {zs.max():.0%} of segments are exactly flat; check the collapse ratio")
    imp = None
    if distance is not None and horizon is not None:
        imp = segment_importance(clf, Z, distance, horizon, y, n_repeats, random_state)
    return DiagnosticReport(states, post, zs, imp, warns)
