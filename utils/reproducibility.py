"""
utils/reproducibility.py — Global Seed & Reproducibility Control
=================================================================
Sets all random seeds (Python, NumPy, PyTorch) from a single master seed.
Must be called once at the entry point of every script and notebook.

Reference
---------
Pineau, J., et al. (2021). Improving Reproducibility in Machine Learning
Research (A Report from the NeurIPS 2019 Reproducibility Program).
Journal of Machine Learning Research, 22(164), 1–20.
URL: https://jmlr.org/papers/v22/20-303.html
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python built-in random, NumPy, PyTorch (CPU + CUDA),
    and the OS-level hash seed.

    Parameters
    ----------
    seed : int
        Master random seed. Default 42 (project-wide convention).

    Notes
    -----
    PyTorch CUDA determinism requires additionally setting
    `torch.backends.cudnn.deterministic = True`, which may reduce
    GPU throughput. Acceptable for research reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch — import guarded to avoid hard dependency during testing
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not required for data preprocessing steps


def get_rng(seed: int = 42) -> np.random.Generator:
    """
    Return a seeded NumPy Generator for use within a module.

    Preferred over np.random.seed() for thread-safe local randomness.

    Parameters
    ----------
    seed : int
        Seed for this generator instance.

    Returns
    -------
    np.random.Generator
        A seeded Generator instance (PCG64 algorithm).

    Example
    -------
    >>> rng = get_rng(42)
    >>> rng.integers(0, 100, size=5)
    array([51, 92, 14, 71, 60])
    """
    return np.random.default_rng(seed)


def get_device(prefer_cuda: bool = True) -> "torch.device":
    """
    Select the compute device, preferring CUDA when available.

    In Colab, GPU availability depends on the runtime type selected via
    Runtime > Change runtime type > T4 GPU / A100 GPU. This function does
    not request a GPU — it only detects whether one is currently attached.

    Parameters
    ----------
    prefer_cuda : bool
        If True (default), returns a CUDA device when one is visible.
        If False, forces CPU regardless of availability (useful for
        debugging numerical issues that may be CUDA-nondeterministic).

    Returns
    -------
    torch.device

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    import torch

    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[device] Using GPU: {name}")
    else:
        device = torch.device("cpu")
        if prefer_cuda:
            print("[device] No GPU detected — falling back to CPU. "
                  "In Colab: Runtime > Change runtime type > GPU.")
        else:
            print("[device] CPU forced (prefer_cuda=False).")
    return device


if __name__ == "__main__":
    # Smoke test
    set_global_seed(42)
    rng = get_rng(42)
    sample = rng.integers(0, 1000, size=5).tolist()
    assert sample == [756, 56, 678, 770, 146], f"Unexpected RNG output: {sample}"
    print(f"Reproducibility smoke test passed. RNG sample: {sample}")
