"""M4 hourly benchmark. Run from the repo root: python -m bench.run_m4 [--seeds 3]"""
from .core import cli
from .data import m4

if __name__ == "__main__":
    cli.main("m4", m4.make, boot_mode="cluster")
