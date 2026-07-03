"""
test_training_loop_smoke.py

Smoke test for experiments/training_loop.py (Phase 2, Step 4).

Validates:
    1. Training completes without error for both CAFE-LightFM and the
       LightFM baseline over 10 epochs.
    2. Mean loss for CAFE-LightFM is strictly lower at epoch 10 than at
       epoch 1 -- confirms the WARP + Adagrad + SCA gradient path
       updates the model in the right direction.
    3. The LightFM baseline completes 10 epochs and reports finite,
       non-negative loss. On this synthetic smoke-test dataset, the
       baseline may start at 0.0, so strict loss decrease is not required.
    4. CAFE-LightFM produces per-stage loss values for all three stages
       (S1, S2, S3) throughout training -- confirms stage-conditioning is
       structurally present.
    5. Reproducibility: re-running with identical seed produces identical
       epoch-1 and epoch-10 mean-loss values for both models.

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
    bundle, config = _setup()
    model = _build_baseline(bundle, config)

    log = train_baseline(model, bundle, config)

    assert len(log.epochs) == config.n_epochs, (
        f"Baseline did not complete all epochs: "
        f"expected={config.n_epochs}, got={len(log.epochs)}"
    )

    for result in log.epochs:
        loss_tensor = torch.tensor(result.mean_loss)

        assert torch.isfinite(loss_tensor), (
            f"Baseline produced non-finite loss at epoch {result.epoch}: "
            f"{result.mean_loss}"
        )

        assert result.mean_loss >= 0.0, (
            f"Baseline loss should be non-negative at epoch {result.epoch}: "
            f"{result.mean_loss}"
        )

    print(
        f"[PASS] LightFM-Baseline training completed with finite loss: "
        f"epoch1={log.epochs[0].mean_loss:.4f}, "
        f"epoch10={log.epochs[-1].mean_loss:.4f}"
    )


def test_cafe_per_stage_losses_present() -> None:
    bundle, config = _setup()
    model = _build_cafe(bundle, config)

    log = train_cafe_lightfm(model, bundle, config)

    assert len(log.epochs) == config.n_epochs, (
        f"CAFE-LightFM did not complete all epochs: "
        f"expected={config.n_epochs}, got={len(log.epochs)}"
    )

    for result in log.epochs:
        for stage in ["S1", "S2", "S3"]:
            assert stage in result.loss_by_stage, (
                f"Missing stage {stage} in epoch {result.epoch}"
            )

            loss_tensor = torch.tensor(result.loss_by_stage[stage])
            assert torch.isfinite(loss_tensor), (
                f"Non-finite loss for stage {stage} at epoch {result.epoch}: "
                f"{result.loss_by_stage[stage]}"
            )

    print(
        "[PASS] CAFE-LightFM records finite per-stage loss "
        "(S1, S2, S3) for every epoch"
    )


def test_reproducibility() -> None:
    bundle, config = _setup()

    # --- CAFE-LightFM reproducibility ---
    cafe_model_a = _build_cafe(bundle, config)
    cafe_log_a = train_cafe_lightfm(cafe_model_a, bundle, config)

    cafe_model_b = _build_cafe(bundle, config)
    cafe_log_b = train_cafe_lightfm(cafe_model_b, bundle, config)

    assert abs(cafe_log_a.epochs[0].mean_loss - cafe_log_b.epochs[0].mean_loss) < 1e-6
    assert abs(cafe_log_a.epochs[-1].mean_loss - cafe_log_b.epochs[-1].mean_loss) < 1e-6

    print(
        f"[PASS] CAFE-LightFM reproducible under seed={config.seed}: "
        f"epoch1={cafe_log_a.epochs[0].mean_loss:.4f}, "
        f"epoch10={cafe_log_a.epochs[-1].mean_loss:.4f}"
    )

    # --- LightFM baseline reproducibility ---
    baseline_model_a = _build_baseline(bundle, config)
    baseline_log_a = train_baseline(baseline_model_a, bundle, config)

    baseline_model_b = _build_baseline(bundle, config)
    baseline_log_b = train_baseline(baseline_model_b, bundle, config)

    assert (
        abs(
            baseline_log_a.epochs[0].mean_loss
            - baseline_log_b.epochs[0].mean_loss
        )
        < 1e-6
    )

    assert (
        abs(
            baseline_log_a.epochs[-1].mean_loss
            - baseline_log_b.epochs[-1].mean_loss
        )
        < 1e-6
    )

    print(
        f"[PASS] LightFM-Baseline reproducible under seed={config.seed}: "
        f"epoch1={baseline_log_a.epochs[0].mean_loss:.4f}, "
        f"epoch10={baseline_log_a.epochs[-1].mean_loss:.4f}"
    )


if __name__ == "__main__":
    test_cafe_loss_decreases()
    test_baseline_training_completes_and_loss_is_finite()
    test_cafe_per_stage_losses_present()
    test_reproducibility()
    print("=== ALL TRAINING LOOP (STEP 4) SMOKE TESTS PASSED ===")
