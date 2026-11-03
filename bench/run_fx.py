"""EUR/USD 1-minute benchmark. Run from the repo root: python -m bench.run_fx [--seeds 3]"""
from .core import cli
from .data import fx

if __name__ == "__main__":
    cli.main("fx", fx.make)
