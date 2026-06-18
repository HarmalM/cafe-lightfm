"""
smoke_test.py — Phase 0 Environment Validation
===============================================
Run this file immediately after cloning the repository and installing
requirements.txt to confirm the environment is correctly configured.

Pass criteria
-------------
1. All required packages import without error
2. Pinned versions match expectations
3. Global seed produces deterministic output
4. Config dataclasses instantiate correctly
5. Logger writes to logs/ without error
6. Directory structure is intact

Usage
-----
    python smoke_test.py

Expected output: "=== ALL PHASE 0 CHECKS PASSED ==="
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


def check(label: str, condition: bool, detail: str = "") -> None:
    """Print pass/fail for a single check."""
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not condition:
        raise AssertionError(f"Smoke test failed: {label}. {detail}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== CAFE-LightFM Phase 0 Smoke Test ===\n")

# 1. Python version
print("── 1. Python Version ──")
major, minor = sys.version_info[:2]
check("Python >= 3.10", (major, minor) >= (3, 10), f"Found {major}.{minor}")

# 2. Core scientific packages
print("\n── 2. Core Package Imports ──")
required_packages = [
    ("numpy",       "np"),
    ("scipy",       "scipy"),
    ("pandas",      "pd"),
    ("sklearn",     "sklearn"),
    ("matplotlib",  "matplotlib"),
]
for pkg_name, alias in required_packages:
    try:
        mod = importlib.import_module(pkg_name)
        ver = getattr(mod, "__version__", "unknown")
        check(f"import {pkg_name}", True, f"v{ver}")
    except ImportError as e:
        check(f"import {pkg_name}", False, str(e))

# ML frameworks — soft check (may not be installed in CI)
print("\n── 3. ML Framework Imports (soft) ──")
for pkg in ["torch", "lightfm"]:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        check(f"import {pkg}", True, f"v{ver}")
    except ImportError:
        print(f"  [WARN] {pkg} not installed — required for model training")

# 3. Project module imports
print("\n── 4. Project Module Imports ──")
try:
    from config import (
        CAFELightFMConfig, DataConfig, EvalConfig,
        NUM_STAGES, STAGE_NAMES, STAGE_WEIGHTS,
        DEFAULT_MODEL_CFG, DEFAULT_DATA_CFG, DEFAULT_EVAL_CFG,
        PATHS, PERSISTENT_ROOT, IS_COLAB,
    )
    check("import config", True)
except Exception as e:
    check("import config", False, str(e))

try:
    from utils.reproducibility import set_global_seed, get_rng
    check("import utils.reproducibility", True)
except Exception as e:
    check("import utils.reproducibility", False, str(e))

try:
    from utils.logger import get_logger
    check("import utils.logger", True)
except Exception as e:
    check("import utils.logger", False, str(e))

# 3b. Runtime environment (Colab / Drive / GPU) — requires config imported above
print("\n── 4b. Runtime Environment ──")
print(f"  Running in Colab : {IS_COLAB}")
print(f"  Persistent root  : {PERSISTENT_ROOT}")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU              : NOT DETECTED "
              "(Runtime > Change runtime type > GPU, if training is intended)")
except ImportError:
    print("  GPU              : torch not installed, cannot check")

# 4. Config correctness
print("\n── 5. Config Assertions ──")
check("NUM_STAGES == 3", NUM_STAGES == 3)
check("STAGE_NAMES length", len(STAGE_NAMES) == 3)
check("STAGE_WEIGHTS sum to 1.0", abs(sum(STAGE_WEIGHTS) - 1.0) < 1e-9,
      f"sum={sum(STAGE_WEIGHTS)}")
check("DEFAULT_MODEL_CFG.embedding_dim == 64",
      DEFAULT_MODEL_CFG.embedding_dim == 64)
check("DEFAULT_DATA_CFG.random_seed == 42",
      DEFAULT_DATA_CFG.random_seed == 42)
check("Stage weight order (finalization highest)",
      STAGE_WEIGHTS[2] > STAGE_WEIGHTS[1] > STAGE_WEIGHTS[0])

# Config JSON round-trip
import tempfile, json
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
    tmp_path = tmp.name
DEFAULT_MODEL_CFG.to_json(tmp_path)
restored = CAFELightFMConfig.from_json(tmp_path)
check("Config JSON round-trip",
      restored.embedding_dim == DEFAULT_MODEL_CFG.embedding_dim)

# 5. Reproducibility
print("\n── 6. Reproducibility ──")
set_global_seed(42)
import numpy as np
a = np.random.randint(0, 1000, size=5).tolist()
set_global_seed(42)
b = np.random.randint(0, 1000, size=5).tolist()
check("NumPy seed reproducibility", a == b, f"run1={a}, run2={b}")

rng = get_rng(42)
s1 = rng.integers(0, 1000, size=3).tolist()
rng2 = get_rng(42)
s2 = rng2.integers(0, 1000, size=3).tolist()
check("Generator seed reproducibility", s1 == s2)

# 6. Logger
print("\n── 7. Logger ──")
log_dir = PATHS["logs"]
log = get_logger("smoke_test", log_dir=log_dir)
log.info("Phase 0 smoke test — logger operational")
check("Logger created", log is not None)
log_files = list(log_dir.glob("smoke_test_*.log"))
check("Log file written", len(log_files) >= 1,
      f"found {len(log_files)} file(s) in {log_dir}")

# 7. Directory structure
print("\n── 8. Directory Structure ──")
# Code directories ship with the repo, relative to ROOT (the cloned repo dir)
code_dirs = ["models/cafe_lightfm", "models/baselines", "utils",
             "experiments/ablation", "experiments/baselines",
             "experiments/evaluation"]
for d in code_dirs:
    check(f"code dir: {d}", (ROOT / d).is_dir())

# Data/log/checkpoint directories are persistent — under Drive in Colab,
# under ROOT otherwise. PATHS already points to the correct location.
persistent_dirs = {
    "data_raw": PATHS["data_raw"],
    "data_processed": PATHS["data_processed"],
    "data_splits": PATHS["data_splits"],
    "logs": PATHS["logs"],
    "checkpoints": PATHS["checkpoints"],
}
for label, p in persistent_dirs.items():
    check(f"persistent dir: {label}", p.is_dir(), str(p))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 45)
print("=== ALL PHASE 0 CHECKS PASSED ===")
print("=" * 45)
print("\nEnvironment is ready for Phase 1 (Data Pipeline).")
print(f"Project root : {ROOT}")
print(f"Python       : {sys.version.split()[0]}")
