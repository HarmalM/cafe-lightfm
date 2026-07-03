"""
test_ndcg_smoke.py

Phase 3, Step 1 smoke test for experiments/ndcg.py.

Validates (no scientific performance claims -- metric validation only):
    1. Perfect ranking (all relevant items first) => NDCG@K == 1.0
    2. Worst ranking (all relevant items last) => NDCG@K < 1.0
    3. All NDCG values in [0, 1] and finite, for both CAFE-LightFM and
       the LightFM baseline, over all three stages and all K in {5,10,20}
    4. Stage coverage: results contain exactly {S1, S2, S3}
    5. Both model signatures work (is_cafe=True and is_cafe=False)
    6. IDCG == 0 edge case (no relevant items) => NDCG@K == 0.0
       (no division-by-zero error)

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418

REQUIRES PyTorch. Run in Colab:
    python -m tests.test_ndcg_smoke
"""

from __future__ import annotations

import torch

from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2, MASTER_SEED
from experiments.ndcg import (
    K_VALUES,
    STAGE_ORDER,
    compute_ndcg_at_k_for_user,
    stage_wise_ndcg,
    print_ndcg_table,
)
from experiments.training_loop import TrainingConfig, train_cafe_lightfm, train_baseline
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

SEED = 42


def _setup():
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
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


# --------------------------------------------------------------------------- #
# Unit tests for core NDCG logic (no model needed)
# --------------------------------------------------------------------------- #

def test_perfect_ranking_gives_ndcg_one() -> None:
    """Perfect ranking: all relevant items ranked first => NDCG@K == 1.0"""
    for k in K_VALUES:
        rel = [1] * k + [0] * 10
        score = compute_ndcg_at_k_for_user(rel, k)
        assert abs(score - 1.0) < 1e-9, f"NDCG@{k} for perfect ranking: {score}"
    print("[PASS] perfect ranking => NDCG@K == 1.0 for all K in {5,10,20}")


def test_worst_ranking_gives_ndcg_less_than_one() -> None:
    """Worst ranking: relevant items at the end => NDCG@K < 1.0"""
    for k in K_VALUES:
        rel = [0] * 10 + [1] * k
        score = compute_ndcg_at_k_for_user(rel, k)
        assert score < 1.0, f"NDCG@{k} for worst ranking should be < 1.0: {score}"
    print("[PASS] worst ranking => NDCG@K < 1.0 for all K in {5,10,20}")


def test_no_relevant_items_gives_zero() -> None:
    """No relevant items in the list => NDCG@K == 0.0 (no division-by-zero)"""
    for k in K_VALUES:
        rel = [0] * 20
        score = compute_ndcg_at_k_for_user(rel, k)
        assert score == 0.0, f"NDCG@{k} with no relevant items: {score}"
    print("[PASS] no relevant items => NDCG@K == 0.0 (no division-by-zero)")


# --------------------------------------------------------------------------- #
# Integration tests with trained models
# --------------------------------------------------------------------------- #

def test_cafe_ndcg_values_in_range() -> None:
    bundle, config = _setup()
    model = _build_and_train_cafe(bundle, config)
    results = stage_wise_ndcg(model, bundle, is_cafe=True)

    assert set(results.keys()) == set(STAGE_ORDER), "Missing stages in results"
    for stage, k_scores in results.items():
        for k, score in k_scores.items():
            assert 0.0 <= score <= 1.0, f"CAFE NDCG@{k} for {stage}={score} out of [0,1]"
            assert not (score != score), f"CAFE NDCG@{k} for {stage} is NaN"

    print_ndcg_table(results, "CAFE-LightFM")
    print("[PASS] CAFE-LightFM: all NDCG@K values finite and in [0,1] for all stages")


def test_baseline_ndcg_values_in_range() -> None:
    bundle, config = _setup()
    model = _build_and_train_baseline(bundle, config)
    results = stage_wise_ndcg(model, bundle, is_cafe=False)

    assert set(results.keys()) == set(STAGE_ORDER)
    for stage, k_scores in results.items():
        for k, score in k_scores.items():
            assert 0.0 <= score <= 1.0, f"Baseline NDCG@{k} for {stage}={score} out of [0,1]"
            assert not (score != score), f"Baseline NDCG@{k} for {stage} is NaN"

    print_ndcg_table(results, "LightFM-Baseline")
    print("[PASS] LightFM-Baseline: all NDCG@K values finite and in [0,1] for all stages")


def test_stage_coverage() -> None:
    bundle, config = _setup()
    model = _build_and_train_cafe(bundle, config)
    results = stage_wise_ndcg(model, bundle, is_cafe=True)
    assert set(results.keys()) == {"S1", "S2", "S3"}
    for stage in STAGE_ORDER:
        assert set(results[stage].keys()) == set(K_VALUES)
    print("[PASS] results contain exactly {S1, S2, S3} x {5, 10, 20}")


if __name__ == "__main__":
    # Unit tests (no model)
    test_perfect_ranking_gives_ndcg_one()
    test_worst_ranking_gives_ndcg_less_than_one()
    test_no_relevant_items_gives_zero()
    # Integration tests (trained models)
    test_cafe_ndcg_values_in_range()
    test_baseline_ndcg_values_in_range()
    test_stage_coverage()
    print("=== ALL PHASE 3 STEP 1 NDCG SMOKE TESTS PASSED ===")
