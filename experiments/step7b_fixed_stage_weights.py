"""
Phase 3, Step 7b - CAFE-LightFM-fixed-stage-weights ablation.

Purpose (validation-only, synthetic-v3 pipeline check - no scientific claim):
    Tests whether the empirical behavior of Full (learned stage-conditioned
    attention, W_base/W_s trained end-to-end via gradient descent) differs
    from a variant in which the SCA attention weights are FROZEN, from
    initialization, at the exact per-stage values already discovered by the
    trained Full checkpoint (Phase 3, Step 6 aggregated alpha matrix).

    Only the regular CAFE-LightFM trainable parameters used by the existing
    training loop are trained. The attention distribution alpha(f, s) is a
    constant, non-learned tensor in this variant.

    This isolates whether gradient-based learning of stage-conditioned
    attention adds predictive value beyond simply having a fixed,
    stage-differentiated reweighting.

Validation scope:
    This is synthetic-v3 validation-only. No scientific superiority,
    necessity, or publishable performance claim should be made from this
    script. Formal claims remain reserved for the Prolific pilot/full study.

Author: CAFE-LightFM project (Paper I). Seed = 42 throughout.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

# --- Reused, not reimplemented -------------------------------------------
# This script must not duplicate ranking, NDCG, Precision, or statistical
# testing logic. It only wires one new fixed-alpha model variant into the
# existing data, training, and evaluation pipeline.
from data.synthetic_generator_v3 import generate_synthetic_dataset_v3
from data.interaction_matrix import build_interaction_matrix
from experiments.training_loop import TrainingConfig, train_cafe_lightfm
from experiments.step4_integration import run_ndcg_precision_paired_comparison
from models.cafe_lightfm.cafe_lightfm import CAFELightFM


SEED = 42


# Frozen per-stage attention weights reused verbatim from the Phase 3
# Step 6 aggregated feature-stage matrix.
# Order: [category, program].
FIXED_STAGE_ALPHA: Dict[str, torch.Tensor] = {
    "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
    "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
    "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
}


def _validate_fixed_alpha(fixed_alpha: Dict[str, torch.Tensor]) -> None:
    """Fail fast if the frozen attention distributions are malformed."""
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


def build_fixed_stage_weights_model(
    bundle,
    embedding_dim: int = 64,
    fixed_alpha: Optional[Dict[str, torch.Tensor]] = None,
) -> CAFELightFM:
    """Construct a CAFE-LightFM instance whose SCA attention is frozen."""
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
    """Train and evaluate the fixed-stage-weights variant on synthetic-v3.

    Reuses:
        - synthetic-v3 generation,
        - interaction-matrix construction,
        - existing CAFE-LightFM training loop,
        - existing paired NDCG/Precision comparison runner.

    This function only adds one model variant:
        CAFE-LightFM-fixed-stage-weights.
    """
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

    model = build_fixed_stage_weights_model(
        bundle=bundle,
        embedding_dim=config.embedding_dim,
        fixed_alpha=FIXED_STAGE_ALPHA,
    )

    # train_cafe_lightfm returns a TrainingLog and trains the model in-place.
    training_log = train_cafe_lightfm(
        model=model,
        bundle=bundle,
        config=config,
    )

    trained_model = model

    comparison = run_ndcg_precision_paired_comparison(
        full_model_checkpoint="outputs/checkpoints/cafe_lightfm_v3_seed42.pt",
        variant_model=trained_model,
        variant_name="fixed_stage_weights",
        bundle=bundle,
        k_values=(5, 10, 20),
    )

    comparison["training_log"] = training_log
    comparison["scope_reminder"] = (
        "VALIDATION-ONLY: synthetic-v3 pipeline check. No scientific "
        "superiority or necessity claim is made. Formal claims are "
        "reserved for the Prolific pilot (N=50) / full study (N=300)."
    )

    return comparison


# --------------------------------------------------------------------- #
# Smoke tests
# --------------------------------------------------------------------- #

def test_fixed_alpha_validation_accepts_locked_values():
    """The locked Step-6-derived weights must pass validation unchanged."""
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
    """Guards against silent drift between this file and the Step 6 report."""
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
