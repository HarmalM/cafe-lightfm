"""
test_precision_at_k_smoke.py

Phase 3, Step 3 smoke test for experiments/precision_at_k.py.

Validates (metric-implementation validation only -- no scientific
performance claims, per the scope note inherited from Steps 1, 1.5, 2):
    1. Perfect ranking -> Precision@K = min(num_positives, K) / K
       (fixed denominator, confirmed 2026-07-04).
    2. Worst ranking -> Precision@K = 0.0.
    3. No relevant items -> Precision@K = 0.0, no division-by-zero.
    3b. k <= 0 raises ValueError (fixed denominator cannot be <= 0).
    4. CAFE-LightFM produces finite Precision@K in [0, 1] for all
       K in {5, 10, 20}, across exactly {S1, S2, S3}.
    5. LightFM baseline produces the same.

Uses the Phase 3 Step 1.5 UNSATURATED validation dataset
(synthetic_generator_v3), per the binding scope boundary in
Phase3_Step1_5_Unsaturated_Validation_Dataset.md -- not v2.

CORRECTION (2026-07-05): the original version of this file assumed
external pytest fixtures (`cafe_model_v3`, `baseline_model_v3`,
`bundle_v3`) that do not exist in this repository's conftest.py, causing
`fixture not found` errors. This version is self-contained, following
the exact pattern already used in tests/test_sw_ndcg_smoke.py: a local
_setup() plus local _build_and_train_cafe()/_build_and_train_baseline()
helpers. No fixtures are required.

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418

REQUIRES PyTorch. Run in Colab:
    python -m tests.test_precision_at_k_smoke
or:
    pytest tests/test_precision_at_k_smoke.py -v
"""

from __future__ import annotations

import math

import torch

from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v3 import generate_synthetic_dataset_v3, MASTER_SEED
from experiments.ndcg import K_VALUES, STAGE_ORDER
from experiments.precision_at_k import (
    compute_precision_at_k,
    print_precision_table,
    stage_wise_precision,
)
from experiments.training_loop import TrainingConfig, train_baseline, train_cafe_lightfm
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

SEED = 42


def _setup():
    dataset = generate_synthetic_dataset_v3(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    config = TrainingConfig(n_epochs=10, seed=SEED)
    return bundle, config


def _build_and_train_cafe(bundle, config) -> CAFELightFM:
    torch.manual_seed(SEED)
    model = CAFELightFM(
        bundle.n_users, bundle.n_items, bundle.n_categories,
        bundle.n_programs, config.n_stages, config.embedding_dim,
    )
    train_cafe_lightfm(model, bundle, config)
    return model


def _build_and_train_baseline(bundle, config) -> LightFMPyTorch:
    torch.manual_seed(SEED)
    model = LightFMPyTorch(
        bundle.n_users, bundle.n_items, bundle.n_categories,
        bundle.n_programs, config.embedding_dim,
    )
    train_baseline(model, bundle, config)
    return model


def _assert_valid_results_shape(results) -> None:
    """
    Validates:
      - stage coverage is exactly {S1, S2, S3}
      - K coverage per stage is exactly {5, 10, 20}
      - every value is finite and in [0, 1]
    """
    assert set(results.keys()) == set(STAGE_ORDER), (
        f"Stage coverage mismatch: got {set(results.keys())}, "
        f"expected {set(STAGE_ORDER)}"
    )
    for stage_key in STAGE_ORDER:
        assert set(results[stage_key].keys()) == set(K_VALUES), (
            f"K coverage mismatch for {stage_key}: got "
            f"{set(results[stage_key].keys())}, expected {set(K_VALUES)}"
        )
        for k in K_VALUES:
            value = results[stage_key][k]
            assert math.isfinite(value), f"{stage_key}@{k} is not finite: {value}"
            assert 0.0 <= value <= 1.0, f"{stage_key}@{k}={value} out of [0,1]"


# --------------------------------------------------------------------------- #
# 1-3b. Pure function tests (no model needed) -- unchanged from prior version
# --------------------------------------------------------------------------- #

def test_precision_at_k_perfect_ranking() -> None:
    """
    All relevant items ranked first. For K <= num_positives, Precision@K
    must equal 1.0. For K > num_positives, Precision@K must equal
    num_positives / K (fixed denominator, confirmed 2026-07-04).
    """
    ranked_relevance = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    assert compute_precision_at_k(ranked_relevance, 3) == 1.0
    for k in (5, 10):
        expected = min(3, k) / k
        got = compute_precision_at_k(ranked_relevance, k)
        assert abs(got - expected) < 1e-9, f"expected {expected}, got {got}"
    print("[PASS] (1) perfect ranking: Precision@K matches fixed-denominator formula")


def test_precision_at_k_worst_ranking() -> None:
    """All relevant items ranked last -> Precision@K = 0.0 for small K."""
    ranked_relevance = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    for k in (3, 5):
        assert compute_precision_at_k(ranked_relevance, k) == 0.0
    print("[PASS] (2) worst ranking: Precision@K = 0.0")


def test_precision_at_k_no_relevant_items() -> None:
    """No relevant items anywhere -> Precision@K = 0.0 for all K, no error."""
    ranked_relevance = [0] * 10
    for k in K_VALUES:
        assert compute_precision_at_k(ranked_relevance, k) == 0.0
    print("[PASS] (3) no relevant items: Precision@K = 0.0, no division-by-zero")


def test_precision_at_k_rejects_non_positive_k() -> None:
    """k <= 0 must raise ValueError (fixed denominator cannot be <= 0)."""
    try:
        compute_precision_at_k([1, 0, 1], 0)
    except ValueError as exc:
        print(f"[PASS] (3b) k<=0 validation raised as expected: {exc}")
        return
    raise AssertionError("Expected ValueError for k=0")


# --------------------------------------------------------------------------- #
# 4-5. Integration tests with trained models on the v3 unsaturated dataset
#      (self-contained -- no external fixtures required)
# --------------------------------------------------------------------------- #

def test_precision_at_k_cafe_lightfm_v3() -> None:
    """Integration: CAFE-LightFM on the unsaturated v3 dataset."""
    bundle, config = _setup()
    model = _build_and_train_cafe(bundle, config)
    results = stage_wise_precision(model, bundle, is_cafe=True)
    _assert_valid_results_shape(results)
    print_precision_table(results, "CAFE-LightFM")
    print("[PASS] (4) CAFE-LightFM: all Precision@K values finite and in [0,1], "
          "stage/K coverage confirmed")


def test_precision_at_k_baseline_v3() -> None:
    """Integration: LightFM baseline on the unsaturated v3 dataset."""
    bundle, config = _setup()
    model = _build_and_train_baseline(bundle, config)
    results = stage_wise_precision(model, bundle, is_cafe=False)
    _assert_valid_results_shape(results)
    print_precision_table(results, "LightFM-Baseline")
    print("[PASS] (5) LightFM-Baseline: all Precision@K values finite and in [0,1], "
          "stage/K coverage confirmed")


if __name__ == "__main__":
    test_precision_at_k_perfect_ranking()
    test_precision_at_k_worst_ranking()
    test_precision_at_k_no_relevant_items()
    test_precision_at_k_rejects_non_positive_k()
    test_precision_at_k_cafe_lightfm_v3()
    test_precision_at_k_baseline_v3()
    print("=== ALL PHASE 3 STEP 3 PRECISION@K SMOKE TESTS PASSED ===")
