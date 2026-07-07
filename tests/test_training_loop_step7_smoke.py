"""
tests/test_training_loop_step7_smoke.py

Phase 3, Step 7: smoke tests for the additive parameters introduced in
experiments/training_loop.py's train_cafe_lightfm (stage_idx_override,
stage_order, stage_to_idx, freeze_uniform).

REVISED (2026-07-08, PI-identified correctness issue): the original
version of this test asserted exactly one score_fn call per stage. This
is WRONG -- warp_loss() (models/cafe_lightfm/warp_loss.py) calls
score_fn once for the vectorized positive-score batch, then AGAIN for
every negative-sampling trial (up to max_sampled per positive), so the
true call count per stage is data/RNG-dependent, not 1.

The correct invariant: cafe_scorer(model, bundle, stage_idx=...,
freeze_uniform=...) is constructed as a SINGLE closure ONCE PER STAGE
inside train_cafe_lightfm's loop body (see training_loop.py:
`score_fn = cafe_scorer(model, bundle, stage_idx=effective_stage_idx,
freeze_uniform=freeze_uniform)`, called once per stage_key iteration,
then reused for every warp_loss call within that stage). Consequently
EVERY score_fn call made during a given stage's iteration captures the
SAME (stage_idx, freeze_uniform) pair, regardless of how many times
warp_loss invokes it internally. These tests therefore assert on the
SET of distinct (stage_idx, freeze_uniform) pairs observed across ALL
recorded calls, which is invariant to negative-sampling call count.

Fully self-contained (local _setup() helpers, no conftest.py
dependency), per project test-pattern convention. Uses a minimal
duck-typed stub model + stub bundle rather than the real CAFELightFM /
InteractionMatrixBundle, so these tests exercise ONLY the wiring logic
in training_loop.py -- not the real model's numerics. The full
numerical-equivalence regression check (Step 5 loss trajectory
reproduced exactly on the real v3 dataset) must be run separately in
Colab against the real components.

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
    (stage_idx, freeze_uniform) it was called with on EVERY forward
    call (there will be multiple calls per stage due to WARP's
    negative-sampling loop -- see module docstring). Since stage_idx is
    built via torch.full_like(item_idx, stage_idx) inside cafe_scorer,
    every element of the stage_idx tensor is identical for a given
    call, so recording tuple(stage_idx.tolist())[0] is sufficient.
    """

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.call_log: List[Tuple[int, bool]] = []

    def forward(self, user_idx, item_idx, category_idx, program_idx, stage_idx, freeze_uniform=False):
        stage_idx_scalar = int(stage_idx[0].item())
        self.call_log.append((stage_idx_scalar, freeze_uniform))
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


def _distinct_pairs(call_log: List[Tuple[int, bool]]) -> Set[Tuple[int, bool]]:
    """Reduces a call log to the set of distinct (stage_idx, freeze_uniform)
    pairs observed -- the call-count-invariant assertion target."""
    return set(call_log)


def test_default_call_uses_module_level_stage_mapping():
    """With all Step 7 params at their defaults, the SET of distinct
    (stage_idx, freeze_uniform) pairs used across all calls must be
    exactly {(STAGE_TO_IDX[s], False) for s in STAGE_ORDER} -- i.e.,
    byte-identical wiring to pre-Step-7 behavior. Call COUNT per stage
    is intentionally not asserted (WARP's negative sampling makes it
    variable)."""
    model, bundle, config = _setup(STAGE_ORDER)
    train_cafe_lightfm(model, bundle, config)

    expected_pairs = {(STAGE_TO_IDX[s], False) for s in STAGE_ORDER}
    observed_pairs = _distinct_pairs(model.call_log)

    assert len(model.call_log) >= len(STAGE_ORDER), (
        "Expected at least one scorer call per stage (positive-batch call)."
    )
    assert observed_pairs == expected_pairs, (
        f"Default call: expected exactly the pairs {expected_pairs}, "
        f"observed {observed_pairs}"
    )


def test_stage_idx_override_forces_constant_stage_idx():
    """noStage ablation: EVERY call across ALL stages must use
    stage_idx=0 -- i.e., only ONE distinct pair should ever appear,
    regardless of how many negative-sampling calls occurred."""
    model, bundle, config = _setup(STAGE_ORDER)
    train_cafe_lightfm(model, bundle, config, stage_idx_override=0)

    observed_pairs = _distinct_pairs(model.call_log)
    assert len(model.call_log) >= len(STAGE_ORDER)
    assert observed_pairs == {(0, False)}, (
        f"stage_idx_override=0 must force stage_idx=0 for every call across "
        f"every stage; observed distinct pairs: {observed_pairs}"
    )


def test_freeze_uniform_passthrough():
    """noAttention ablation: freeze_uniform=True must reach EVERY scorer
    call, and stage_idx must remain the normal per-stage mapping (the
    two ablations are independent axes)."""
    model, bundle, config = _setup(STAGE_ORDER)
    train_cafe_lightfm(model, bundle, config, freeze_uniform=True)

    expected_pairs = {(STAGE_TO_IDX[s], True) for s in STAGE_ORDER}
    observed_pairs = _distinct_pairs(model.call_log)

    assert len(model.call_log) >= len(STAGE_ORDER)
    assert observed_pairs == expected_pairs, (
        f"freeze_uniform=True must reach every call with the normal "
        f"per-stage stage_idx mapping; expected {expected_pairs}, "
        f"observed {observed_pairs}"
    )
    assert all(fu is True for _, fu in model.call_log), (
        "Every single call must have freeze_uniform=True, no exceptions."
    )


def test_custom_stage_order_and_mapping_for_2stage():
    """2Stage ablation: a custom stage_order/stage_to_idx pair (e.g. S1 +
    a merged 'decision' stage) must be honored across all calls, with no
    dependence on the module-level STAGE_ORDER/STAGE_TO_IDX constants."""
    custom_order = ["S1", "decision"]
    custom_mapping = {"S1": 0, "decision": 1}
    model, bundle, config = _setup(custom_order)

    train_cafe_lightfm(
        model, bundle, config, stage_order=custom_order, stage_to_idx=custom_mapping
    )

    expected_pairs = {(custom_mapping[s], False) for s in custom_order}
    observed_pairs = _distinct_pairs(model.call_log)

    assert len(model.call_log) >= len(custom_order)
    assert observed_pairs == expected_pairs, (
        f"Custom stage mapping: expected {expected_pairs}, got {observed_pairs}"
    )


def test_missing_stage_key_in_bundle_still_handled():
    """If a stage_order key has no positive pairs in the bundle, the loop
    must record loss_by_stage[key]=0.0 and skip scoring entirely for
    that key -- unchanged pre-existing behavior, re-verified after the
    Step 7 additions. No call with that stage's index should ever
    appear in the call log."""
    model, bundle, config = _setup(["S1", "S2"])  # bundle has NO S3 pairs
    log = train_cafe_lightfm(
        model, bundle, config, stage_order=["S1", "S2", "S3"], stage_to_idx=STAGE_TO_IDX
    )
    assert log.epochs[-1].loss_by_stage["S3"] == 0.0, (
        "A stage_order key absent from the bundle must contribute 0.0 loss."
    )
    observed_pairs = _distinct_pairs(model.call_log)
    assert observed_pairs == {(0, False), (1, False)}, (
        f"Only S1 (idx 0) and S2 (idx 1) should have triggered scorer "
        f"calls; S3 (idx 2) must not appear. Observed: {observed_pairs}"
    )


if __name__ == "__main__":
    test_default_call_uses_module_level_stage_mapping()
    test_stage_idx_override_forces_constant_stage_idx()
    test_freeze_uniform_passthrough()
    test_custom_stage_order_and_mapping_for_2stage()
    test_missing_stage_key_in_bundle_still_handled()
    print("test_training_loop_step7_smoke.py: all 5 checks passed.")
