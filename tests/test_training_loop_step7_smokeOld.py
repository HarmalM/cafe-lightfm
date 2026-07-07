"""
tests/test_training_loop_step7_smoke.py

Phase 3, Step 7: smoke tests for the additive parameters introduced in
experiments/training_loop.py's train_cafe_lightfm (stage_idx_override,
stage_order, stage_to_idx, freeze_uniform).

Fully self-contained (local _setup() helpers, no conftest.py dependency),
per project test-pattern convention (established in
test_sw_ndcg_smoke.py). Uses a minimal duck-typed stub model + stub
bundle rather than the real CAFELightFM / InteractionMatrixBundle, so
these tests exercise ONLY the wiring logic in training_loop.py -- not
the real model's numerics. The full numerical-equivalence regression
check (Step 5 loss trajectory reproduced exactly on the real v3 dataset)
must be run separately in Colab against the real components, per
project convention (this environment has no access to
data/synthetic_generator_v3.py or models/cafe_lightfm/*.py's real
dependency graph).

Reproducibility: seed=42 throughout, per project convention.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

from experiments.training_loop import STAGE_ORDER, STAGE_TO_IDX, TrainingConfig, train_cafe_lightfm

MASTER_SEED = 42


class _StubBundle:
    """Minimal duck-typed stand-in for InteractionMatrixBundle."""

    def __init__(self, positive_pairs_by_stage: Dict[str, Set[Tuple[int, int]]], n_items: int):
        self.positive_pairs_by_stage = positive_pairs_by_stage
        self.n_items = n_items
        self.item_feature_idx_by_item = {i: (0, 0) for i in range(n_items)}


class _StubCAFEModel(nn.Module):
    """
    Minimal duck-typed stand-in for CAFELightFM. Records the exact
    (stage_idx values, freeze_uniform) it was called with on every
    forward call, so tests can assert on training_loop.py's wiring
    without depending on the real model.
    """

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.call_log: List[Tuple[Tuple[int, ...], bool]] = []

    def forward(self, user_idx, item_idx, category_idx, program_idx, stage_idx, freeze_uniform=False):
        self.call_log.append((tuple(stage_idx.tolist()), freeze_uniform))
        # Score depends on dummy_param so backward()/optimizer.step() are
        # exercised realistically (non-zero, differentiable path).
        return self.dummy_param.expand(item_idx.shape[0]) + 0.0 * item_idx.float()


def _setup(stage_keys: List[str]) -> Tuple[_StubCAFEModel, _StubBundle, TrainingConfig]:
    random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    # Two positive pairs per stage key, users 0/1, items 0/1 -- enough for
    # warp_loss to run without degenerate all-excluded edge cases.
    pairs_by_stage = {key: {(0, 0), (1, 1)} for key in stage_keys}
    bundle = _StubBundle(positive_pairs_by_stage=pairs_by_stage, n_items=3)
    model = _StubCAFEModel()
    config = TrainingConfig(n_epochs=1, max_sampled=5)
    return model, bundle, config


def test_default_call_uses_module_level_stage_mapping():
    """With all Step 7 params at their defaults, each stage_key must be
    scored with STAGE_TO_IDX[stage_key], and freeze_uniform must be False
    -- i.e., byte-identical wiring to pre-Step-7 behavior."""
    model, bundle, config = _setup(STAGE_ORDER)
    train_cafe_lightfm(model, bundle, config)

    assert len(model.call_log) == len(STAGE_ORDER), "Expected one scorer call per stage."
    for (stage_idx_values, freeze_uniform), stage_key in zip(model.call_log, STAGE_ORDER):
        expected_idx = STAGE_TO_IDX[stage_key]
        assert all(v == expected_idx for v in stage_idx_values), (
            f"Default call: stage_key={stage_key} expected stage_idx="
            f"{expected_idx}, got {stage_idx_values}"
        )
        assert freeze_uniform is False, "Default call must pass freeze_uniform=False."


def test_stage_idx_override_forces_constant_stage_idx():
    """noStage ablation: every stage-loop iteration must score with the
    override value, regardless of which stage's data is being consumed."""
    model, bundle, config = _setup(STAGE_ORDER)
    train_cafe_lightfm(model, bundle, config, stage_idx_override=0)

    assert len(model.call_log) == len(STAGE_ORDER)
    for stage_idx_values, freeze_uniform in model.call_log:
        assert all(v == 0 for v in stage_idx_values), (
            f"stage_idx_override=0 must force stage_idx=0 for every stage, got {stage_idx_values}"
        )
        assert freeze_uniform is False


def test_freeze_uniform_passthrough():
    """noAttention ablation: freeze_uniform=True must reach every scorer
    call, and stage_idx must remain the normal per-stage mapping (the two
    ablations are independent axes)."""
    model, bundle, config = _setup(STAGE_ORDER)
    train_cafe_lightfm(model, bundle, config, freeze_uniform=True)

    assert len(model.call_log) == len(STAGE_ORDER)
    for (stage_idx_values, freeze_uniform), stage_key in zip(model.call_log, STAGE_ORDER):
        expected_idx = STAGE_TO_IDX[stage_key]
        assert all(v == expected_idx for v in stage_idx_values), (
            "freeze_uniform=True must not alter stage_idx wiring."
        )
        assert freeze_uniform is True, "freeze_uniform=True must reach every scorer call."


def test_custom_stage_order_and_mapping_for_2stage():
    """2Stage ablation: a custom stage_order/stage_to_idx pair (e.g. S1 +
    a merged 'decision' stage) must be honored, with no dependence on the
    module-level STAGE_ORDER/STAGE_TO_IDX constants."""
    custom_order = ["S1", "decision"]
    custom_mapping = {"S1": 0, "decision": 1}
    model, bundle, config = _setup(custom_order)

    train_cafe_lightfm(
        model, bundle, config, stage_order=custom_order, stage_to_idx=custom_mapping
    )

    assert len(model.call_log) == len(custom_order), (
        "Expected exactly one scorer call per custom stage key."
    )
    for (stage_idx_values, _), stage_key in zip(model.call_log, custom_order):
        expected_idx = custom_mapping[stage_key]
        assert all(v == expected_idx for v in stage_idx_values), (
            f"Custom stage_key={stage_key} expected stage_idx={expected_idx}, "
            f"got {stage_idx_values}"
        )


def test_missing_stage_key_in_bundle_still_handled():
    """If a stage_order key has no positive pairs in the bundle, the loop
    must record loss_by_stage[key]=0.0 and skip scoring -- unchanged
    pre-existing behavior, re-verified after the Step 7 additions."""
    model, bundle, config = _setup(["S1", "S2"])
    log = train_cafe_lightfm(
        model, bundle, config, stage_order=["S1", "S2", "S3"], stage_to_idx=STAGE_TO_IDX
    )
    assert log.epochs[-1].loss_by_stage["S3"] == 0.0, (
        "A stage_order key absent from the bundle must contribute 0.0 loss."
    )
    # Only S1 and S2 should have triggered a real scorer call.
    assert len(model.call_log) == 2


if __name__ == "__main__":
    test_default_call_uses_module_level_stage_mapping()
    test_stage_idx_override_forces_constant_stage_idx()
    test_freeze_uniform_passthrough()
    test_custom_stage_order_and_mapping_for_2stage()
    test_missing_stage_key_in_bundle_still_handled()
    print("test_training_loop_step7_smoke.py: all 5 checks passed.")
