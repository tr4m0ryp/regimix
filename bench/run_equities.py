"""US large-cap daily benchmark. Run from the repo root: python -m bench.run_equities [--seeds 3]"""
from .core import cli
from .data import equities

if __name__ == "__main__":
    cli.main("eq", equities.make)
