"""Repro tests for the two hmmlearn patches in ``patches/hmmlearn``.

Each test copies the installed hmmlearn package into a temporary directory, applies a patch with ``patch -p3``
(the patches use the upstream ``src/hmmlearn/...`` layout) and runs code against the patched copy in a
subprocess. The stock package is checked to still show the bug, so the tests also notice when an hmmlearn
release has fixed it. Set ``REGIMIX_UPSTREAM_SUITE=1`` to also run hmmlearn's whole test suite on the patched
copy (about 10 minutes).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

hmmlearn = pytest.importorskip("hmmlearn")
from hmmlearn.hmm import GaussianHMM  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
MIN_COVAR = HERE / "0001-gaussianhmm-apply-min-covar-after-every-m-step.patch"
SPHERICAL = HERE / "0002-gaussianhmm-fix-spherical-covars-shape.patch"
needs_patch = pytest.mark.skipif(shutil.which("patch") is None, reason="the patch tool is not installed")


def patched_copy(tmp_path: Path, *patches: Path) -> Path:
    dst = tmp_path / "hmmlearn"
    shutil.copytree(Path(hmmlearn.__file__).parent, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for p in patches:
        subprocess.run(["patch", "-p3", "--batch", "--quiet", "-i", str(p)], cwd=dst, check=True)
    return tmp_path


def run_py(root: Path, code: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "2"}
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True,
                         text=True, cwd=root)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def run_pytest(root: Path, target: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "2"}
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir",
                          str(root), target], env=env, capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stdout[-3000:]


REPRO_MIN_COVAR = """
    import warnings, numpy as np, hmmlearn
    from hmmlearn.hmm import GaussianHMM
    assert hmmlearn.__file__.startswith({root!r})
    warnings.simplefilter("ignore")
    X = np.round(np.random.RandomState(0).normal(0, 0.5, (1000, 7)))
    h = GaussianHMM(2, "diag", min_covar=1e-3, n_iter=100, tol=1e-4, random_state=0)
    h.fit(X.reshape(-1, 1), [7] * 1000)
    print(h._covars_.min())
"""


def _stock_collapse() -> float:
    X = np.round(np.random.RandomState(0).normal(0, 0.5, (1000, 7)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = GaussianHMM(2, "diag", min_covar=1e-3, n_iter=100, tol=1e-4, random_state=0)
        h.fit(X.reshape(-1, 1), [7] * 1000)
    return float(h._covars_.min())


@needs_patch
@pytest.mark.parametrize("stack", [(MIN_COVAR,), (SPHERICAL,), (MIN_COVAR, SPHERICAL), (SPHERICAL, MIN_COVAR)])
def test_patches_apply_alone_and_together(tmp_path, stack):
    patched_copy(tmp_path, *stack)


def test_stock_min_covar_is_not_a_floor():
    v = _stock_collapse()
    if v >= 1e-3:
        pytest.skip("the installed hmmlearn already applies min_covar in the M-step")
    assert v < 0.1 * 1e-3


@needs_patch
def test_min_covar_patch_holds_the_floor(tmp_path):
    root = patched_copy(tmp_path, MIN_COVAR)
    v = float(run_py(root, REPRO_MIN_COVAR.format(root=str(root))))
    assert v >= 1e-3 * (1 - 1e-12)


@needs_patch
def test_min_covar_patch_upstream_tests(tmp_path):
    root = patched_copy(tmp_path, MIN_COVAR)
    run_pytest(root, "hmmlearn/tests/test_gaussian_hmm_min_covar.py")


def test_stock_spherical_covars_shape_is_wrong():
    X = np.random.RandomState(0).normal(size=(400, 2))
    h = GaussianHMM(2, "spherical", n_iter=3, random_state=0).fit(X)
    if h.covars_.shape == (2, 2, 2):
        pytest.skip("the installed hmmlearn already returns one matrix per state")
    assert h.covars_.shape == (4, 2, 2) and h._covars_.shape == (2, 2)


@needs_patch
def test_spherical_patch_shape_and_sampling(tmp_path):
    root = patched_copy(tmp_path, SPHERICAL)
    run_py(root, f"""
        import numpy as np, hmmlearn
        from hmmlearn.hmm import GaussianHMM
        assert hmmlearn.__file__.startswith({str(root)!r})
        rng = np.random.RandomState(0)
        X = np.concatenate([rng.normal(0, 1, (300, 2)), rng.normal(5, 3, (300, 2))])
        h = GaussianHMM(2, "spherical", n_iter=5, random_state=0).fit(X)
        assert h._covars_.shape == (2,), h._covars_.shape
        assert h.covars_.shape == (2, 2, 2), h.covars_.shape
        for k in range(2):
            np.testing.assert_allclose(h.covars_[k], h._covars_[k] * np.eye(2))
        ll_before = h.score(X)
        h.covars_ = np.tile(h._covars_[:, None], (1, 2))  # the legacy (n_components, n_features) layout
        assert h.covars_.shape == (2, 2, 2)
        np.testing.assert_allclose(h.score(X), ll_before)
    """)
    run_pytest(root, "hmmlearn/tests/test_gaussian_hmm_spherical_shape.py")


@needs_patch
@pytest.mark.skipif(os.environ.get("REGIMIX_UPSTREAM_SUITE") != "1", reason="set REGIMIX_UPSTREAM_SUITE=1")
def test_upstream_suite_with_both_patches(tmp_path):
    root = patched_copy(tmp_path, MIN_COVAR, SPHERICAL)
    run_pytest(root, "hmmlearn/tests")
