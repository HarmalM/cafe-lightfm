"""
Smoke tests for Phase 3, Step 7b fixed_alpha support in StageConditionedAttention.

These tests verify that:
1. Default learned-attention behavior still works.
2. freeze_uniform still works.
3. fixed_alpha returns the expected per-stage alpha values.
4. malformed fixed_alpha inputs are rejected.
5. fixed_alpha and freeze_uniform are mutually exclusive at forward time.
6. pre-existing call sites are not broken.

Run from the repo root:

    PYTHONPATH=/content/cafe-lightfm pytest -q tests/test_step7b_sca_fixed_alpha_smoke.py
"""

from __future__ import annotations

import pytest
import torch

from models.cafe_lightfm.sca_layer import StageConditionedAttention


SEED = 42
N_STAGES = 3
N_FEATURES = 2
EMBEDDING_DIM = 8
BATCH_SIZE = 8

LOCKED_FIXED_ALPHA = {
    "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
    "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
    "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
}


def _make_layer(**kwargs) -> StageConditionedAttention:
    torch.manual_seed(SEED)
    return StageConditionedAttention(
        n_stages=N_STAGES,
        embedding_dim=EMBEDDING_DIM,
        **kwargs,
    )


def _dummy_batch():
    torch.manual_seed(SEED)
    feature_embeddings = torch.randn(
        BATCH_SIZE,
        N_FEATURES,
        EMBEDDING_DIM,
    )
    stage_idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    return feature_embeddings, stage_idx


def test_default_behavior_unchanged():
    layer = _make_layer()
    feature_embeddings, stage_idx = _dummy_batch()

    weighted_sum, alpha = layer(feature_embeddings, stage_idx)

    assert weighted_sum.shape == (BATCH_SIZE, EMBEDDING_DIM)
    assert alpha.shape == (BATCH_SIZE, N_FEATURES)

    expected_uniform = torch.full_like(alpha, 1.0 / N_FEATURES)

    # Because w_base and w_stage are zero-initialized, default alpha is uniform at init.
    assert torch.allclose(alpha, expected_uniform, atol=1e-6)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6)


def test_freeze_uniform_still_works():
    layer = _make_layer()
    feature_embeddings, stage_idx = _dummy_batch()

    weighted_default, alpha_default = layer(feature_embeddings, stage_idx)
    weighted_frozen, alpha_frozen = layer(
        feature_embeddings,
        stage_idx,
        freeze_uniform=True,
    )

    assert weighted_frozen.shape == (BATCH_SIZE, EMBEDDING_DIM)
    assert alpha_frozen.shape == (BATCH_SIZE, N_FEATURES)

    # At zero initialization, freeze_uniform should match the default path.
    assert torch.allclose(alpha_frozen, alpha_default, atol=1e-6)
    assert torch.allclose(weighted_frozen, weighted_default, atol=1e-6)
    assert torch.allclose(alpha_frozen.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6)


def test_fixed_alpha_returns_expected_per_stage_weights():
    layer = _make_layer(fixed_alpha=LOCKED_FIXED_ALPHA)
    feature_embeddings, stage_idx = _dummy_batch()

    weighted_sum, alpha = layer(feature_embeddings, stage_idx)

    expected_alpha = torch.stack(
        [
            LOCKED_FIXED_ALPHA["S1"],
            LOCKED_FIXED_ALPHA["S2"],
            LOCKED_FIXED_ALPHA["S3"],
            LOCKED_FIXED_ALPHA["S1"],
            LOCKED_FIXED_ALPHA["S2"],
            LOCKED_FIXED_ALPHA["S3"],
            LOCKED_FIXED_ALPHA["S1"],
            LOCKED_FIXED_ALPHA["S2"],
        ],
        dim=0,
    )

    assert weighted_sum.shape == (BATCH_SIZE, EMBEDDING_DIM)
    assert alpha.shape == (BATCH_SIZE, N_FEATURES)
    assert torch.allclose(alpha, expected_alpha, atol=1e-6)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6)


def test_fixed_alpha_rejects_malformed_input():
    bad_missing_stage = {
        "S1": torch.tensor([0.5, 0.5]),
        "S2": torch.tensor([0.5, 0.5]),
    }

    with pytest.raises(ValueError):
        _make_layer(fixed_alpha=bad_missing_stage)

    bad_non_unit_sum = {
        "S1": torch.tensor([0.5, 0.6]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }

    with pytest.raises(ValueError):
        _make_layer(fixed_alpha=bad_non_unit_sum)

    bad_negative = {
        "S1": torch.tensor([1.2, -0.2]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }

    with pytest.raises(ValueError):
        _make_layer(fixed_alpha=bad_negative)


def test_freeze_uniform_and_fixed_alpha_are_mutually_exclusive():
    layer = _make_layer(fixed_alpha=LOCKED_FIXED_ALPHA)
    feature_embeddings, stage_idx = _dummy_batch()

    with pytest.raises(ValueError):
        layer(
            feature_embeddings,
            stage_idx,
            freeze_uniform=True,
        )


def test_pre_existing_call_sites_not_broken():
    # Exact old-style construction used before Step 7b.
    layer = StageConditionedAttention(
        n_stages=N_STAGES,
        embedding_dim=EMBEDDING_DIM,
    )

    feature_embeddings, stage_idx = _dummy_batch()

    weighted_sum, alpha = layer(feature_embeddings, stage_idx)

    assert weighted_sum.shape == (BATCH_SIZE, EMBEDDING_DIM)
    assert alpha.shape == (BATCH_SIZE, N_FEATURES)


if __name__ == "__main__":
    test_default_behavior_unchanged()
    test_freeze_uniform_still_works()
    test_fixed_alpha_returns_expected_per_stage_weights()
    test_fixed_alpha_rejects_malformed_input()
    test_freeze_uniform_and_fixed_alpha_are_mutually_exclusive()
    test_pre_existing_call_sites_not_broken()

    print("test_step7b_sca_fixed_alpha_smoke.py: all checks passed.")
