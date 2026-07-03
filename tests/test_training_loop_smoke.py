"""
test_training_loop_smoke.py

Smoke test for experiments/training_loop.py (Phase 2, Step 4).

Validates:
    1. CAFE-LightFM mean loss decreases over 10 epochs.
    2. LightFM baseline completes 10 epochs with finite, non-negative loss.
       NOTE: baseline loss is expected to be 0.0 on the Step 4 synthetic
       dataset because all users have interacted with all 10 items globally,
       leaving no valid negatives under the global exclusion scope. This is
       correct WARP behavior, not a bug (confirmed 2026-06-26).
    3. CAFE-LightFM records finite per-stage loss (S1, S2, S3) every epoch.
    4. Reproducibility under seed=42 for both models.

REQUIRES PyTorch. Run in Colab:
    python -m tests.test_training_loop_smoke
"""

from __future__ import annotations

import torch

from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2, MASTER_SEED
from experiments.training_loop import (
    TrainingConfig,
    train_baseline,
    train_cafe_lightfm,
)
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

SEED = 42


def _setup():
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    config = TrainingConfig(n_epochs=10, seed=SEED)
    return bundle, config


def _build_cafe(bundle, config) -> CAFELightFM:
    torch.manual_seed(SEED)
    return CAFELightFM(
        bundle.n_users,
        bundle.n_items,
        bundle.n_categories,
        bundle.n_programs,
        config.n_stages,
        config.embedding_dim,
    )


def _build_baseline(bundle, config) -> LightFMPyTorch:
    torch.manual_seed(SEED)
    return LightFMPyTorch(
        bundle.n_users,
        bundle.n_items,
        bundle.n_categories,
        bundle.n_programs,
        config.embedding_dim,
    )


def test_cafe_loss_decreases() -> None:
    bundle, config = _setup()
    model = _build_cafe(bundle, config)
    log = train_cafe_lightfm(model, bundle, config)
    loss_epoch1 = log.epochs[0].mean_loss
    loss_epoch10 = log.epochs[-1].mean_loss
    assert loss_epoch10 < loss_epoch1, (
        f"CAFE-LightFM mean loss did not decrease: "
        f"epoch1={loss_epoch1:.4f}, epoch10={loss_epoch10:.4f}"
    )
    print(
        f"[PASS] CAFE-LightFM mean loss decreased: "
        f"epoch1={loss_epoch1:.4f} -> epoch10={loss_epoch10:.4f}"
    )


def test_baseline_training_completes_and_loss_is_finite() -> None:
    """
    On the Step 4 synthetic dataset, all 20 users have interacted with
    all 10 items globally, so no valid negative exists under the global
    exclusion scope. WARP correctly returns 0.0 loss (no violators found,
    pos_score * 0.0 path). The test therefore checks completion and
    finiteness only -- not strict decrease.
    """
    bundle, config = _setup()
    model = _build_baseline(bundle, config)
    log = train_baseline(model, bundle, config)
    assert len(log.epochs) == config.n_epochs
    for result in log.epochs:
        assert torch.isfinite(torch.tensor(result.mean_loss)), (
            f"Non-finite baseline loss at epoch {result.epoch}"
        )
        assert result.mean_loss >= 0.0
    print(
        f"[PASS] LightFM-Baseline completed {config.n_epochs} epochs with "
        f"finite loss (0.0 expected: no valid global negatives on this dataset)"
    )


def test_cafe_per_stage_losses_present() -> None:
    bundle, config = _setup()
    model = _build_cafe(bundle, config)
    log = train_cafe_lightfm(model, bundle, config)
    assert len(log.epochs) == config.n_epochs
    for result in log.epochs:
        for stage in ["S1", "S2", "S3"]:
            assert stage in result.loss_by_stage, (
                f"Missing stage {stage} in epoch {result.epoch}"
            )
            assert torch.isfinite(torch.tensor(result.loss_by_stage[stage]))
    print(
        "[PASS] CAFE-LightFM records finite per-stage loss "
        "(S1, S2, S3) for every epoch"
    )


def test_reproducibility() -> None:
    bundle, config = _setup()

    # CAFE-LightFM
    cafe_a = _build_cafe(bundle, config)
    log_a = train_cafe_lightfm(cafe_a, bundle, config)
    cafe_b = _build_cafe(bundle, config)
    log_b = train_cafe_lightfm(cafe_b, bundle, config)
    assert abs(log_a.epochs[0].mean_loss - log_b.epochs[0].mean_loss) < 1e-6
    assert abs(log_a.epochs[-1].mean_loss - log_b.epochs[-1].mean_loss) < 1e-6
    print(
        f"[PASS] CAFE-LightFM reproducible under seed={config.seed}: "
        f"epoch1={log_a.epochs[0].mean_loss:.4f}, "
        f"epoch10={log_a.epochs[-1].mean_loss:.4f}"
    )

    # LightFM Baseline
    base_a = _build_baseline(bundle, config)
    log_base_a = train_baseline(base_a, bundle, config)
    base_b = _build_baseline(bundle, config)
    log_base_b = train_baseline(base_b, bundle, config)
    assert abs(log_base_a.epochs[0].mean_loss - log_base_b.epochs[0].mean_loss) < 1e-6
    assert abs(log_base_a.epochs[-1].mean_loss - log_base_b.epochs[-1].mean_loss) < 1e-6
    print(
        f"[PASS] LightFM-Baseline reproducible under seed={config.seed}: "
        f"epoch1={log_base_a.epochs[0].mean_loss:.4f}, "
        f"epoch10={log_base_a.epochs[-1].mean_loss:.4f}"
    )


if __name__ == "__main__":
    test_cafe_loss_decreases()
    test_baseline_training_completes_and_loss_is_finite()
    test_cafe_per_stage_losses_present()
    test_reproducibility()
    print("=== ALL TRAINING LOOP (STEP 4) SMOKE TESTS PASSED ===")
