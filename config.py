"""
config.py — CAFE-LightFM Centralized Configuration Registry
============================================================
All hyperparameters, paths, and experimental constants are declared here.
No magic numbers appear anywhere else in the codebase.

Design rationale
----------------
Using a dataclass-based config (rather than YAML-only) provides:
  - Type safety and IDE autocompletion
  - Immutability enforcement via frozen=True
  - Simple serialization to JSON for experiment logging

Reference
---------
Wilson, G., et al. (2017). Good Enough Practices in Scientific Computing.
PLOS Computational Biology, 13(6), e1005510.
DOI: 10.1371/journal.pcbi.1005510
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple


# ── Paths ─────────────────────────────────────────────────────────────────────
#
# Colab + Google Drive support
# -----------------------------
# Code (the cloned git repo) lives in the Colab VM's ephemeral filesystem
# under /content/<repo_name>. This is fast but wiped on disconnect.
#
# Data, checkpoints, and logs must live on Drive (persistent) so a session
# disconnect does not lose training progress or processed datasets.
#
# Resolution order:
#   1. CAFE_LIGHTFM_DRIVE_ROOT env var, if set (manual override)
#   2. /content/drive/MyDrive/cafe_lightfm, if Drive is mounted (Colab)
#   3. ROOT_DIR (this file's directory), for local/non-Colab development
#
# Code-relative paths (config.py, models/, utils/) always resolve against
# ROOT_DIR — they ship with the repo and don't need to persist independently.

ROOT_DIR: Path = Path(__file__).parent.resolve()


def _is_colab() -> bool:
    """Detect whether code is running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_drive_root() -> Path:
    """
    Resolve the persistent storage root.

    Raises
    ------
    RuntimeError
        If running in Colab but Drive is not mounted at the expected path.
        Call `from google.colab import drive; drive.mount('/content/drive')`
        before importing this module.
    """
    env_override = os.environ.get("CAFE_LIGHTFM_DRIVE_ROOT")
    if env_override:
        return Path(env_override)

    if _is_colab():
        drive_path = Path("/content/drive/MyDrive/cafe_lightfm")
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError(
                "Running in Colab but Google Drive is not mounted. "
                "Run this first:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')"
            )
        drive_path.mkdir(parents=True, exist_ok=True)
        return drive_path

    return ROOT_DIR  # local development fallback


PERSISTENT_ROOT: Path = _resolve_drive_root()
IS_COLAB: bool = _is_colab()

PATHS = {
    "data_raw":        PERSISTENT_ROOT / "data" / "raw",
    "data_processed":  PERSISTENT_ROOT / "data" / "processed",
    "data_splits":     PERSISTENT_ROOT / "data" / "splits",
    "logs":            PERSISTENT_ROOT / "logs",
    "checkpoints":     PERSISTENT_ROOT / "outputs" / "checkpoints",
    "results":         PERSISTENT_ROOT / "outputs",
}

# Auto-create persistent directories (idempotent, safe on every import)
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)


# ── Decision Stage Constants ───────────────────────────────────────────────────
# Three stages defined per Section 5.1 of the CAFE-LightFM proposal:
#   s1 = Exploration, s2 = Comparison, s3 = Finalization
#
# Stage labels are integers {0, 1, 2} throughout the codebase.
# "Stage" is a higher-order construct spanning multiple sessions —
# NOT equivalent to a session or time step (see proposal §3).

NUM_STAGES: int = 3
STAGE_NAMES: Tuple[str, ...] = ("exploration", "comparison", "finalization")
STAGE_TO_IDX: dict = {name: idx for idx, name in enumerate(STAGE_NAMES)}
IDX_TO_STAGE: dict = {idx: name for idx, name in enumerate(STAGE_NAMES)}

# Stage-Weighted NDCG weights (finalization receives highest weight)
# Rationale: finalization-stage errors are most consequential to users.
STAGE_WEIGHTS: Tuple[float, ...] = (0.2, 0.3, 0.5)


# ── Model Hyperparameters ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class CAFELightFMConfig:
    """
    Hyperparameter registry for CAFE-LightFM.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of user/item/feature latent embeddings.
        Typical range: [32, 256]. Default follows LightFM convention [1].
    num_stages : int
        Number of decision stages K. Fixed at 3 per formal definition §5.1.
    attention_hidden_dim : int
        Hidden layer size of the Stage-Conditioned Attention (SCA) network.
        Controls capacity of per-stage weight matrices W_{s_j}.
    dropout_rate : float
        Dropout applied to feature embeddings during training.
        Regularization to prevent stage-specific overfitting.
    learning_rate : float
        Adam optimizer learning rate [2].
    weight_decay : float
        L2 regularization coefficient on all parameters.
    batch_size : int
        Number of (user, positive_item, negative_item) WARP triplets per batch.
    max_epochs : int
        Maximum training epochs (early stopping applied separately).
    early_stopping_patience : int
        Number of epochs without NDCG@10 improvement before stopping.
    warp_n_samples : int
        Number of negative samples per positive in WARP loss approximation [3].
    k_values : List[int]
        Cutoff values K for NDCG@K and Precision@K evaluation.

    References
    ----------
    [1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
        Recommendations. RecSys 2015 Workshops. CEUR-WS Vol. 1448, pp. 14-21.
    [2] Kingma, D.P., & Ba, J. (2015). Adam: A Method for Stochastic
        Optimization. ICLR 2015. arXiv:1412.6980.
    [3] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to
        Large Vocabulary Image Annotation. IJCAI 2011, pp. 2764-2770.
    """
    embedding_dim: int = 64
    num_stages: int = NUM_STAGES
    attention_hidden_dim: int = 128
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 100
    early_stopping_patience: int = 10
    warp_n_samples: int = 10
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])

    def to_json(self, path: Path | str) -> None:
        """Serialize config to JSON for experiment reproducibility."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "CAFELightFMConfig":
        """Deserialize config from JSON."""
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


@dataclass(frozen=True)
class BaselineLightFMConfig:
    """Hyperparameter config for vanilla LightFM baseline (Kula, 2015)."""
    embedding_dim: int = 64
    loss: str = "warp"          # options: "warp", "bpr", "logistic"
    max_sampled: int = 10       # WARP negative samples
    learning_rate: float = 0.05
    max_epochs: int = 100
    early_stopping_patience: int = 10
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])


# ── Data Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataConfig:
    """
    Dataset and splitting configuration.

    Attributes
    ----------
    random_seed : int
        Master seed. All random operations derive from this.
    train_ratio : float
        Fraction of interactions assigned to training split.
    val_ratio : float
        Fraction of interactions assigned to validation split.
        Test ratio = 1 - train_ratio - val_ratio.
    min_interactions_per_user : int
        Users with fewer interactions are excluded (cold-start filter).
    min_interactions_per_item : int
        Items with fewer interactions are excluded.
    stage_split_strategy : str
        "temporal" — split within each stage by interaction timestamp.
        "user_holdout" — hold out last session of each stage per user.
    """
    random_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 3
    stage_split_strategy: str = "temporal"


# ── Evaluation Configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalConfig:
    """
    Evaluation protocol configuration.

    stage_weights : Tuple[float, ...]
        Per-stage weights for Stage-Weighted NDCG metric.
        Higher weight on finalization (s3) reflects decision salience.
    alpha_bonferroni : float
        Family-wise error rate after Bonferroni correction.
        With 6 baselines × 3 stages × 3 K-values = 54 comparisons,
        per-comparison alpha = 0.05 / 54 ≈ 0.0009.
    n_bootstrap : int
        Bootstrap resampling iterations for confidence intervals.
    """
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    stage_weights: Tuple[float, ...] = STAGE_WEIGHTS
    alpha_family: float = 0.05
    n_bootstrap: int = 1000


# ── Default Instances (import-ready) ─────────────────────────────────────────

DEFAULT_MODEL_CFG  = CAFELightFMConfig()
DEFAULT_DATA_CFG   = DataConfig()
DEFAULT_EVAL_CFG   = EvalConfig()


if __name__ == "__main__":
    # Quick sanity check
    print("=== CAFE-LightFM Config Sanity Check ===")
    print(f"Model  : {DEFAULT_MODEL_CFG}")
    print(f"Data   : {DEFAULT_DATA_CFG}")
    print(f"Eval   : {DEFAULT_EVAL_CFG}")
    print(f"Stages : {STAGE_NAMES}")
    print(f"Weights: {STAGE_WEIGHTS}")
    assert sum(STAGE_WEIGHTS) == 1.0, "Stage weights must sum to 1.0"
    assert DEFAULT_MODEL_CFG.num_stages == NUM_STAGES
    print("All assertions passed.")
