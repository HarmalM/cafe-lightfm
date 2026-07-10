"""
Phase 3, Step 7b - CAFE-LightFM fixed-stage-weights ablation.

Validation-only synthetic-v3 pipeline check.
No scientific superiority or necessity claim should be made from this script.

This script compares:
    model_a: Full CAFE-LightFM with learned stage-conditioned attention
    model_b: CAFE-LightFM with fixed stage-conditioned attention weights

RNG-control note:
    For fair comparison, torch.manual_seed(seed) is set immediately before
    model construction and immediately before training for both Full CAFE and
    FixedStageWeights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from data.synthetic_generator_v3 import generate_synthetic_dataset_v3
from data.interaction_matrix import build_interaction_matrix
from experiments.training_loop import TrainingConfig, train_cafe_lightfm
from experiments.step7_ablation_study import run_cafe_variant_paired_comparison
from models.cafe_lightfm.cafe_lightfm import CAFELightFM


SEED = 42

CHECKPOINT_DIR = Path("outputs/checkpoints")
FULL_CAFE_CHECKPOINT = CHECKPOINT_DIR / "cafe_lightfm_v3_seed42.pt"
FIXED_STAGE_WEIGHTS_CHECKPOINT = CHECKPOINT_DIR / "cafe_lightfm_fixed_stage_weights_v3_seed42.pt"


FIXED_STAGE_ALPHA: Dict[str, torch.Tensor] = {
    "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
    "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
    "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
}


def _validate_fixed_alpha(fixed_alpha: Dict[str, torch.Tensor]) -> None:
    expected_stages = {"S1", "S2", "S3"}

    if set(fixed_alpha.keys()) != expected_stages:
        raise ValueError(
            f"fixed_alpha must cover exactly {expected_stages}, "
            f"got {set(fixed_alpha.keys())}"
        )

    for stage, vec in fixed_alpha.items():
        if not torch.is_tensor(vec):
            raise ValueError(f"fixed_alpha[{stage}] must be a torch.Tensor")

        if vec.ndim != 1:
            raise ValueError(
                f"fixed_alpha[{stage}] must be 1-D, got shape {tuple(vec.shape)}"
            )

        if not torch.isfinite(vec).all():
            raise ValueError(f"fixed_alpha[{stage}] contains non-finite values")

        if torch.any(vec < 0):
            raise ValueError(f"fixed_alpha[{stage}] contains a negative weight")

        total = vec.sum().item()
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"fixed_alpha[{stage}] must sum to 1.0 within tolerance, got {total}"
            )


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not extract model state_dict from checkpoint. Expected a raw "
        "state_dict or a dict containing 'model_state_dict' or 'state_dict'."
    )


def _make_full_cafe_model(bundle, embedding_dim: int) -> CAFELightFM:
    return CAFELightFM(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        n_categories=bundle.n_categories,
        n_programs=bundle.n_programs,
        n_stages=3,
        embedding_dim=embedding_dim,
    )


def build_or_train_full_cafe_model(
    bundle,
    config: TrainingConfig,
    checkpoint_path: Path = FULL_CAFE_CHECKPOINT,
) -> Tuple[CAFELightFM, Optional[object], str]:
    """
    Load Full CAFE-LightFM checkpoint if available.
    If missing, retrain Full CAFE-LightFM and save the checkpoint.

    RNG policy:
        - seed before model construction
        - seed before training
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    model = _make_full_cafe_model(bundle=bundle, embedding_dim=config.embedding_dim)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model, None, "loaded_checkpoint"

    torch.manual_seed(config.seed)
    training_log = train_cafe_lightfm(
        model=model,
        bundle=bundle,
        config=config,
    )

    model.eval()

    if training_log.epochs:
        assert training_log.epochs[-1].mean_loss < training_log.epochs[0].mean_loss, (
            "Full CAFE-LightFM training loss did not decrease. "
            "Check optimizer/data before trusting evaluation output."
        )

    torch.save(model.state_dict(), checkpoint_path)

    return model, training_log, "retrained_missing_checkpoint"


def build_fixed_stage_weights_model(
    bundle,
    embedding_dim: int = 64,
    fixed_alpha: Optional[Dict[str, torch.Tensor]] = None,
) -> CAFELightFM:
    fixed_alpha = fixed_alpha or FIXED_STAGE_ALPHA
    _validate_fixed_alpha(fixed_alpha)

    model = CAFELightFM(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        n_categories=bundle.n_categories,
        n_programs=bundle.n_programs,
        n_stages=3,
        embedding_dim=embedding_dim,
        fixed_alpha=fixed_alpha,
    )

    return model


def run_fixed_stage_weights_ablation(seed: int = SEED) -> dict:
    torch.manual_seed(seed)

    records = generate_synthetic_dataset_v3(seed=seed)
    bundle = build_interaction_matrix(records)

    config = TrainingConfig(
        n_epochs=10,
        lr=0.05,
        max_sampled=10,
        margin=1.0,
        embedding_dim=64,
        n_stages=3,
        seed=seed,
    )

    full_model, full_training_log, full_model_source = build_or_train_full_cafe_model(
        bundle=bundle,
        config=config,
        checkpoint_path=FULL_CAFE_CHECKPOINT,
    )

    torch.manual_seed(seed)
    fixed_model = build_fixed_stage_weights_model(
        bundle=bundle,
        embedding_dim=config.embedding_dim,
        fixed_alpha=FIXED_STAGE_ALPHA,
    )

    torch.manual_seed(seed)
    fixed_training_log = train_cafe_lightfm(
        model=fixed_model,
        bundle=bundle,
        config=config,
    )

    fixed_model.eval()

    if fixed_training_log.epochs:
        assert (
            fixed_training_log.epochs[-1].mean_loss
            < fixed_training_log.epochs[0].mean_loss
        ), (
            "fixedStageWeights training loss did not decrease. "
            "Check optimizer/data before trusting evaluation output."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(fixed_model.state_dict(), FIXED_STAGE_WEIGHTS_CHECKPOINT)

    comparison = run_cafe_variant_paired_comparison(
        model_a=full_model,
        model_b=fixed_model,
        bundle=bundle,
        k_values=(5, 10, 20),
    )

    return {
        "variant_name": "fixed_stage_weights",
        "model_a": "full_cafe_lightfm",
        "model_b": "fixed_stage_weights",
        "full_model_source": full_model_source,
        "full_checkpoint_path": str(FULL_CAFE_CHECKPOINT),
        "fixed_checkpoint_path": str(FIXED_STAGE_WEIGHTS_CHECKPOINT),
        "comparison": comparison,
        "rng_control": {
            "seed": seed,
            "full_model_seeded_before_construction": True,
            "full_model_seeded_before_training": True,
            "fixed_model_seeded_before_construction": True,
            "fixed_model_seeded_before_training": True,
        },
        "scope_reminder": (
            "VALIDATION-ONLY: synthetic-v3 pipeline check. No scientific "
            "superiority or necessity claim is made."
        ),
    }


def test_fixed_alpha_validation_accepts_locked_values():
    _validate_fixed_alpha(FIXED_STAGE_ALPHA)


def test_fixed_alpha_validation_rejects_bad_stage_keys():
    bad = {
        "S1": torch.tensor([0.5, 0.5]),
        "S2": torch.tensor([0.5, 0.5]),
    }

    try:
        _validate_fixed_alpha(bad)
        raise AssertionError("expected ValueError for missing S3")
    except ValueError:
        pass


def test_fixed_alpha_validation_rejects_non_unit_sum():
    bad = {
        "S1": torch.tensor([0.5, 0.6]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }

    try:
        _validate_fixed_alpha(bad)
        raise AssertionError("expected ValueError for row not summing to 1.0")
    except ValueError:
        pass


def test_fixed_alpha_validation_rejects_negative_weight():
    bad = {
        "S1": torch.tensor([1.2, -0.2]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }

    try:
        _validate_fixed_alpha(bad)
        raise AssertionError("expected ValueError for negative weight")
    except ValueError:
        pass


def test_locked_weights_match_step6_source_values():
    assert torch.allclose(
        FIXED_STAGE_ALPHA["S1"],
        torch.tensor([0.6144, 0.3856]),
        atol=1e-4,
    )

    assert torch.allclose(
        FIXED_STAGE_ALPHA["S2"],
        torch.tensor([0.6035, 0.3965]),
        atol=1e-4,
    )

    assert torch.allclose(
        FIXED_STAGE_ALPHA["S3"],
        torch.tensor([0.6724, 0.3276]),
        atol=1e-4,
    )


if __name__ == "__main__":
    test_fixed_alpha_validation_accepts_locked_values()
    test_fixed_alpha_validation_rejects_bad_stage_keys()
    test_fixed_alpha_validation_rejects_non_unit_sum()
    test_fixed_alpha_validation_rejects_negative_weight()
    test_locked_weights_match_step6_source_values()

    print("step7b_fixed_stage_weights.py smoke checks passed.")
