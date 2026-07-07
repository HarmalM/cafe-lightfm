"""
tests/test_step7_ablation_smoke.py

Phase 3, Step 7: smoke tests for the NEW logic introduced in
experiments/step7_ablation_study.py (remap_bundle_to_2stage,
compute_descriptive_table, run_cafe_variant_paired_comparison).

Fully self-contained (local _setup() helpers, no conftest.py
dependency), per project test-pattern convention. Uses minimal
duck-typed stubs for the bundle and model rather than the real
InteractionMatrixBundle / CAFELightFM, so these tests exercise ONLY the
wiring/aggregation logic this script adds -- not the real model's
numerics or the real synthetic-v3 dataset. The full end-to-end run
(real training, real checkpoints, real bundle) must be executed in
Colab, per project convention.

Reproducibility: seed=42 throughout, per project convention.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

from experiments.step7_ablation_study import (
    compute_descriptive_table,
    remap_bundle_to_2stage,
    run_cafe_variant_paired_comparison,
)

MASTER_SEED = 42


class _StubBundle:
    """Minimal duck-typed stand-in for InteractionMatrixBundle."""

    def __init__(self, positive_pairs_by_stage: Dict[str, Set[Tuple[int, int]]], n_items: int):
        self.positive_pairs_by_stage = positive_pairs_by_stage
        self.n_items = n_items
        self.item_feature_idx_by_item = {i: (0, 0) for i in range(n_items)}


class _StubCAFEModel(nn.Module):
    """
    Minimal duck-typed stand-in for a trained CAFELightFM. Scores are a
    deterministic function of (user_idx, item_idx, stage_idx) so that
    per-user rankings are non-trivial and reproducible, letting the
    paired-comparison / descriptive-table logic be exercised
    meaningfully without a real trained model.
    """

    def __init__(self, stage_bias: float = 0.0):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.stage_bias = stage_bias

    def forward(self, user_idx, item_idx, category_idx, program_idx, stage_idx, freeze_uniform=False):
        # Deterministic, differentiable-shaped score: favors higher
        # item_idx for even stage_idx, lower item_idx for odd stage_idx,
        # offset by stage_bias (so two model instances can be made to
        # differ in a controlled, testable way).
        stage_sign = torch.where(stage_idx % 2 == 0, 1.0, -1.0)
        return self.dummy_param + stage_sign * item_idx.float() + self.stage_bias


def _setup_3stage_bundle() -> _StubBundle:
    pairs_by_stage = {
        "S1": {(0, 1), (1, 2)},
        "S2": {(0, 2), (1, 0)},
        "S3": {(0, 0), (1, 1)},
    }
    return _StubBundle(positive_pairs_by_stage=pairs_by_stage, n_items=4)


def test_remap_bundle_to_2stage_merges_s2_s3_and_preserves_s1():
    bundle = _setup_3stage_bundle()
    remapped = remap_bundle_to_2stage(bundle)

    assert set(remapped.positive_pairs_by_stage.keys()) == {"S1", "decision"}
    assert remapped.positive_pairs_by_stage["S1"] == bundle.positive_pairs_by_stage["S1"]
    assert remapped.positive_pairs_by_stage["decision"] == (
        bundle.positive_pairs_by_stage["S2"] | bundle.positive_pairs_by_stage["S3"]
    )


def test_remap_bundle_to_2stage_does_not_mutate_original():
    bundle = _setup_3stage_bundle()
    original_keys = set(bundle.positive_pairs_by_stage.keys())
    _ = remap_bundle_to_2stage(bundle)
    assert set(bundle.positive_pairs_by_stage.keys()) == original_keys, (
        "remap_bundle_to_2stage must not mutate the original bundle's "
        "positive_pairs_by_stage in place."
    )


def test_remap_bundle_to_2stage_shares_other_fields():
    bundle = _setup_3stage_bundle()
    remapped = remap_bundle_to_2stage(bundle)
    assert remapped.n_items == bundle.n_items
    assert remapped.item_feature_idx_by_item is bundle.item_feature_idx_by_item


def test_compute_descriptive_table_returns_finite_values_in_range():
    bundle = _setup_3stage_bundle()
    model = _StubCAFEModel()
    table = compute_descriptive_table(model, bundle, ("S1", "S2", "S3"))

    assert len(table) == 3 * 3 * 2  # 3 stages x 3 K-values x 2 metrics
    for label, value in table.items():
        assert value == value, f"{label} is NaN unexpectedly (empty stage?)."
        assert 0.0 <= value <= 1.0, f"{label}={value} out of [0,1] range."


def test_compute_descriptive_table_respects_stage_idx_override():
    """A model whose forward output depends on stage_idx must produce a
    DIFFERENT descriptive table under stage_idx_override=0 vs. the
    default per-stage mapping -- confirms the override actually reaches
    the scoring call."""
    bundle = _setup_3stage_bundle()
    model = _StubCAFEModel()

    table_default = compute_descriptive_table(model, bundle, ("S2",))
    table_override = compute_descriptive_table(model, bundle, ("S2",), stage_idx_override=0)

    # S2 default stage_idx=1 (odd -> stage_sign=-1); override=0 (even ->
    # stage_sign=+1) -- rankings, and therefore metric values, must
    # differ for at least one (stage,K,metric) entry.
    assert table_default != table_override, (
        "stage_idx_override did not change the descriptive table -- "
        "override is not reaching the scoring call."
    )


def test_compute_descriptive_table_respects_custom_stage_to_idx():
    bundle = _setup_3stage_bundle()
    remapped = remap_bundle_to_2stage(bundle)
    model = _StubCAFEModel()

    table = compute_descriptive_table(
        model, remapped, ("S1", "decision"), stage_to_idx={"S1": 0, "decision": 1}
    )
    assert len(table) == 2 * 3 * 2  # 2 stages x 3 K-values x 2 metrics
    for label, value in table.items():
        assert value == value, f"{label} unexpectedly NaN."


def test_run_cafe_variant_paired_comparison_produces_expected_keys():
    bundle = _setup_3stage_bundle()
    model_a = _StubCAFEModel(stage_bias=0.0)
    model_b = _StubCAFEModel(stage_bias=5.0)  # differ enough to avoid zero-variance edge case

    summary = run_cafe_variant_paired_comparison(model_a, model_b, bundle)

    assert len(summary) > 0
    for label, r in summary.items():
        assert set(r.keys()) >= {
            "n_pairs", "mean_a", "mean_b", "mean_diff", "t_statistic",
            "cohens_dz", "p_value", "alpha_corrected", "significant",
        }


def test_run_cafe_variant_paired_comparison_stage_idx_override_b():
    """Verifies stage_idx_override_b reaches model_b's scoring calls by
    checking the comparison runs without error and produces results even
    when model_b is 'trained' under a fixed stage_idx (simulating
    noStage's evaluation-time requirement)."""
    bundle = _setup_3stage_bundle()
    model_a = _StubCAFEModel(stage_bias=0.0)
    model_b = _StubCAFEModel(stage_bias=3.0)

    summary = run_cafe_variant_paired_comparison(
        model_a, model_b, bundle, stage_idx_override_b=0
    )
    assert len(summary) > 0


if __name__ == "__main__":
    test_remap_bundle_to_2stage_merges_s2_s3_and_preserves_s1()
    test_remap_bundle_to_2stage_does_not_mutate_original()
    test_remap_bundle_to_2stage_shares_other_fields()
    test_compute_descriptive_table_returns_finite_values_in_range()
    test_compute_descriptive_table_respects_stage_idx_override()
    test_compute_descriptive_table_respects_custom_stage_to_idx()
    test_run_cafe_variant_paired_comparison_produces_expected_keys()
    test_run_cafe_variant_paired_comparison_stage_idx_override_b()
    print("test_step7_ablation_smoke.py: all 8 checks passed.")
