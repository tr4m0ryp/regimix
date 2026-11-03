"""Input handling: arrays or a DataFrame holding the segments, the distance and the horizon."""
from __future__ import annotations

import numpy as np

__all__ = ["resolve_inputs", "encode_labels"]


def _is_frame(Z) -> bool:
    return hasattr(Z, "columns") and hasattr(Z, "to_numpy")


def resolve_inputs(Z, distance, horizon, distance_col=None, horizon_col=None, segment_cols=None):
    """Return (Z, distance, horizon) as float arrays of shapes (n, S), (n,), (n,).

    ``Z`` may be an array or a DataFrame. With a DataFrame, ``distance_col`` / ``horizon_col`` name columns
    used when ``distance`` / ``horizon`` are not passed, and the segments are ``segment_cols`` (default: every
    remaining column, in frame order, which must be oldest first). Scalars broadcast to every episode.
    """
    if _is_frame(Z):
        if distance is None and distance_col is not None:
            distance = Z[distance_col].to_numpy(np.float64)
        if horizon is None and horizon_col is not None:
            horizon = Z[horizon_col].to_numpy(np.float64)
        if segment_cols is None:
            drop = {c for c in (distance_col, horizon_col) if c is not None}
            segment_cols = [c for c in Z.columns if c not in drop]
        Z = Z[list(segment_cols)].to_numpy(np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (n_episodes, n_segments)")
    if not np.isfinite(Z).all():
        raise ValueError("Z contains NaN or inf; drop or impute those episodes first")
    if distance is None or horizon is None:
        raise ValueError("distance and horizon are required (as arrays or via distance_col / horizon_col)")
    n = len(Z)
    d = np.broadcast_to(np.asarray(distance, dtype=np.float64), (n,)).copy()
    h = np.broadcast_to(np.asarray(horizon, dtype=np.float64), (n,)).copy()
    if not (np.isfinite(d).all() and np.isfinite(h).all()):
        raise ValueError("distance and horizon must be finite")
    if np.any(h <= 0):
        raise ValueError("horizon must be strictly positive")
    return Z, d, h


def encode_labels(y) -> tuple[np.ndarray, np.ndarray]:
    """(classes, y01): exactly two classes; the larger one is the positive class (exceeded the threshold)."""
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"y must hold exactly two classes, found {classes.size}")
    return classes, (y == classes[1]).astype(int)
