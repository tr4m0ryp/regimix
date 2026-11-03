"""Turn a partial path into unit-time scaled lookback segments."""
from .segments import check_edges, segments_from_cumulative, segments_from_path

__all__ = ["check_edges", "segments_from_cumulative", "segments_from_path"]
