"""
Smoke tests for the `fixed_alpha` hook on StageConditionedAttention /
CAFELightFM (Phase 3, Step 7b — fixed-stage-weights ablation).

IMPORTANT: import from the REAL modules after integration, not from
sca_layer_fixed_alpha_patch.py (which is a proposed-patch reference, not
the production module). Run this file only after the patch has been
merged into the actual repo files and re-uploaded.

Run: pytest tests/test_step7b_sca_fixed_alpha_smoke.py -v
"""

from __future__ import annotations

import torch
import pytest

# After integration, replace with the real import path, e.g.:
# from models.cafe_lightfm.sca_layer import StageConditionedAttention
# from models.cafe_lightfm.cafe_lightfm import CAFELightFM
from models.cafe_lightfm.sca_layer import StageConditionedAttention  # noqa: F401

SEED = 42
EMBEDDING_DIM = 64
N_STAGES = 3
N_FEATURES = 2

LOCKED_FIXED_ALPHA = {
    "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
    "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
    "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
}


def _make_layer(**kwargs) -> StageConditionedAttention:
    torch.manual_seed(SEED)
    return StageConditionedAttention(
        embedding_dim=EMBEDDING_DIM, n_stages=N_STAGES, n_features=N_FEATURES, **kwargs
    )


def _dummy_batch(batch_size: int = 8):
    torch.manual_seed(SEED)
    feature_embeds = torch.randn(batch_size, N_FEATURES, EMBEDDING_DIM)
    stage_idx = torch.randint(0, N_STAGES, (batch_size,))
    return feature_embeds, stage_idx


# 1. Default behavior (fixed_alpha=None, freeze_uniform=False) must be
#    byte-identical to pre-patch Steps 1-7 behavior: alpha rows sum to 1.0,
#    finite, correct shape, and NOT constant across the batch (i.e., it is
#    genuinely the learned softmax path, not accidentally short-circuited).
def test_default_behavior_unchanged():
    layer = _make_layer()  # fixed_alpha=None, freeze_uniform=False (defaults)
    feature_embeds, stage_idx = _dummy_batch()
    alpha = layer(feature_embeds, stage_idx)

    assert alpha.shape == (8, N_FEATURES)
    assert torch.all(torch.isfinite(alpha))
    row_sums = alpha.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    # a freshly-initialized learned layer should not degenerate to a
    # perfectly uniform row for every sample (sanity check, not a strict
    # correctness proof)
    assert not torch.allclose(alpha, torch.full_like(alpha, 1.0 / N_FEATURES))


# 2. freeze_uniform=True must still work exactly as in Step 7 (regression
#    guard — this is the existing hook the new one must not disturb).
def test_freeze_uniform_still_works():
    layer = _make_layer(freeze_uniform=True)
    feature_embeds, stage_idx = _dummy_batch()
    alpha = layer(feature_embeds, stage_idx)

    expected = torch.full((8, N_FEATURES), 1.0 / N_FEATURES)
    assert torch.allclose(alpha, expected)


# 3. fixed_alpha returns exactly the caller-specified per-stage row,
#    correctly indexed by stage_idx, regardless of feature_embeds content
#    (the whole point of the ablation: attention no longer depends on the
#    learned embeddings at all).
def test_fixed_alpha_returns_expected_per_stage_weights():
    layer = _make_layer(fixed_alpha=LOCKED_FIXED_ALPHA)
    feature_embeds, _ = _dummy_batch(batch_size=3)
    stage_idx = torch.tensor([0, 1, 2])  # S1, S2, S3

    alpha = layer(feature_embeds, stage_idx)

    assert torch.allclose(alpha[0], LOCKED_FIXED_ALPHA["S1"], atol=1e-6)
    assert torch.allclose(alpha[1], LOCKED_FIXED_ALPHA["S2"], atol=1e-6)
    assert torch.allclose(alpha[2], LOCKED_FIXED_ALPHA["S3"], atol=1e-6)

    # changing feature_embeds must NOT change the output (bypass confirmed)
    alpha_again = layer(torch.randn_like(feature_embeds) * 100, stage_idx)
    assert torch.allclose(alpha, alpha_again)


# 3b. fixed_alpha must fail fast on malformed input (mirrors sw_ndcg's
#     _validate_weights edge cases).
def test_fixed_alpha_rejects_malformed_input():
    bad_missing_stage = {"S1": torch.tensor([0.5, 0.5]), "S2": torch.tensor([0.5, 0.5])}
    with pytest.raises(ValueError):
        _make_layer(fixed_alpha=bad_missing_stage)

    bad_negative = {
        "S1": torch.tensor([1.2, -0.2]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }
    with pytest.raises(ValueError):
        _make_layer(fixed_alpha=bad_negative)

    bad_sum = {
        "S1": torch.tensor([0.5, 0.6]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }
    with pytest.raises(ValueError):
        _make_layer(fixed_alpha=bad_sum)


# 3c. freeze_uniform and fixed_alpha are mutually exclusive.
def test_freeze_uniform_and_fixed_alpha_are_mutually_exclusive():
    with pytest.raises(ValueError):
        _make_layer(freeze_uniform=True, fixed_alpha=LOCKED_FIXED_ALPHA)


# 4. No previous call site is broken: constructing the layer with ONLY the
#    pre-existing positional/keyword arguments (as every Step 1-7 call site
#    does) must still succeed without needing to know about fixed_alpha.
def test_pre_existing_call_sites_not_broken():
    # exact signature used throughout Steps 1-7, per the handoff docs
    layer = StageConditionedAttention(
        embedding_dim=EMBEDDING_DIM, n_stages=N_STAGES
    )
    feature_embeds, stage_idx = _dummy_batch()
    alpha = layer(feature_embeds, stage_idx)
    assert alpha.shape == (8, N_FEATURES)

    layer_noattn = StageConditionedAttention(
        embedding_dim=EMBEDDING_DIM, n_stages=N_STAGES, freeze_uniform=True
    )
    alpha_noattn = layer_noattn(feature_embeds, stage_idx)
    assert torch.allclose(
        alpha_noattn, torch.full((8, N_FEATURES), 1.0 / N_FEATURES)
    )


if __name__ == "__main__":
    test_default_behavior_unchanged()
    test_freeze_uniform_still_works()
    test_fixed_alpha_returns_expected_per_stage_weights()
    test_fixed_alpha_rejects_malformed_input()
    test_freeze_uniform_and_fixed_alpha_are_mutually_exclusive()
    test_pre_existing_call_sites_not_broken()
    print("All 6 fixed_alpha smoke checks passed.")
