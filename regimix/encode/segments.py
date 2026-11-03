"""Multi-resolution lookback segments of a scalar process, scaled to unit time.

For lookback edges ``0 = e_0 < e_1 < ... < e_S`` the segment ``s`` of an episode decided at time ``t`` is the
increment of the process over ``(t - e_s, t - e_{s-1}]`` divided by ``sqrt(e_s - e_{s-1})``. Under a driftless
Brownian motion with one fixed variance every segment then has the same distribution, which is what lets a
single hidden Markov model read all of them as one short sequence. Segments are returned oldest first, so the
last column is the most recent move.
"""
from __future__ import annotations

import numpy as np

__all__ = ["segments_from_cumulative", "segments_from_path", "check_edges"]


def check_edges(edges) -> np.ndarray:
    """Validate lookback edges: a strictly increasing 1-D sequence that starts at 0."""
    e = np.asarray(edges, dtype=np.float64).ravel()
    if e.size < 2 or e[0] != 0 or np.any(np.diff(e) <= 0):
        raise ValueError("edges must start at 0 and increase strictly, e.g. [0, 1, 3, 7]")
    return e


def _scale(increments: np.ndarray, edges: np.ndarray, clip: float | None) -> np.ndarray:
    """Divide each segment by sqrt(duration), reverse to oldest first and optionally clip."""
    z = increments / np.sqrt(np.diff(edges))[None, :]
    z = z[:, ::-1]
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return np.ascontiguousarray(z)


def segments_from_cumulative(cum, lags, clip: float | None = None) -> np.ndarray:
    """Segments from trailing changes.

    Parameters
    ----------
    cum : array of shape (n, L)
        ``cum[i, k]`` is the change of the process over the last ``lags[k]`` time units before episode i's
        decision time, ``X(t_i) - X(t_i - lags[k])``.
    lags : increasing sequence of L positive durations.
    clip : optional bound c; segments are clipped to ``[-c, c]`` after scaling.

    Returns
    -------
    Z : array of shape (n, L), oldest segment first, each scaled to unit time.
    """
    cum = np.atleast_2d(np.asarray(cum, dtype=np.float64))
    edges = check_edges(np.concatenate([[0.0], np.asarray(lags, dtype=np.float64).ravel()]))
    if cum.shape[1] != edges.size - 1:
        raise ValueError(f"cum has {cum.shape[1]} columns but {edges.size - 1} lags were given")
    full = np.column_stack([np.zeros(len(cum)), cum])
    return _scale(np.diff(full, axis=1), edges, clip)


def _value_at(values: np.ndarray, times: np.ndarray | None, when: np.ndarray) -> np.ndarray:
    """Last observed value at or before each time in ``when``; NaN before the first observation."""
    if times is None:
        idx = np.asarray(when)
        if not np.allclose(idx, np.round(idx)):
            raise ValueError("without times, decision times and edges must be whole sample indices")
        idx = np.round(idx).astype(np.int64)
        out = np.full(idx.shape, np.nan)
        ok = (idx >= 0) & (idx < len(values))
        out[ok] = values[idx[ok]]
        return out
    pos = np.searchsorted(times, when, side="right") - 1
    out = np.full(pos.shape, np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def segments_from_path(values, times, t_decision, edges, clip: float | None = None) -> np.ndarray:
    """Segments read from one observed path at many decision times.

    Parameters
    ----------
    values : array of shape (m,), the observed process.
    times : array of shape (m,) of increasing observation times, or None to use sample indices.
        With times, the value at any moment is the last observation at or before it (no interpolation
        forward), so nothing observed after a decision time can enter its segments.
    t_decision : array of shape (n,), the decision times (sample indices when ``times`` is None).
    edges : lookback edges ``[0, e_1, ..., e_S]`` in the units of ``times``.
    clip : optional bound c; segments are clipped to ``[-c, c]`` after scaling.

    Returns
    -------
    Z : array of shape (n, S), oldest first. Rows whose lookback starts before the first observation are NaN.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    e = check_edges(edges)
    t = np.atleast_1d(np.asarray(t_decision, dtype=np.float64))
    if times is not None:
        times = np.asarray(times, dtype=np.float64).ravel()
        if times.shape != values.shape or np.any(np.diff(times) < 0):
            raise ValueError("times must match values in length and be non-decreasing")
    level = _value_at(values, times, t[:, None] - e[None, :])  # (n, S + 1), newest first
    return _scale(level[:, :-1] - level[:, 1:], e, clip)
