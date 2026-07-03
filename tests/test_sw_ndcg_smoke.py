"""
test_sw_ndcg_smoke.py

Phase 3, Step 2 smoke test for experiments/sw_ndcg.py.

Validates (metric-implementation validation only -- no scientific
performance claims, per the scope note inherited from Steps 1 and 1.5):
    1. Weighted combination matches a hand-computed value for known
       per-stage NDCG inputs (pure function, no model needed).
    2. Weight validation rejects non-summing-to-1 and mismatched-key
       weight dicts.
    3. Default weights equal the locked project configuration
       (S1=0.20, S2=0.30, S3=0.50).
    4. Both model signatures (is_cafe=True/False) produce finite
       SW-NDCG@K in [0, 1] for all K in {5, 10, 20}.
    5. sw_ndcg() end-to-end output equals
       compute_sw_ndcg_from_stage_results(stage_wise_ndcg(...)) exactly
       (consistency between the two entry points).

Uses the Phase 3 Step 1.5 UNSATURATED validation dataset
(synthetic_generator_v3), per the binding scope boundary in
Phase3_Step1_5_Unsaturated_Validation_Dataset.md: the Step 4 (v2)
dataset leaves the LightFM baseline architecturally blocked (zero
negatives), which would make any baseline-vs-CAFE SW-NDCG comparison
here reflect a dataset artifact rather than the pipeline itself.

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418

REQUIRES PyTorch. Run in Colab:
    python -m tests.test_sw_ndcg_smoke
"""

from __future__ import annotations

import torch

from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v3 import generate_synthetic_dataset_v3, MASTER_SEED
from experiments.ndcg import K_VALUES, stage_wise_ndcg
from experiments.sw_ndcg import (
    DEFAULT_STAGE_WEIGHTS,
    compute_sw_ndcg_from_stage_results,
    print_sw_ndcg_table,
    sw_ndcg,
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


# --------------------------------------------------------------------------- #
# Unit tests for the weighting logic (no model needed)
# --------------------------------------------------------------------------- #

def test_hand_computed_weighted_sum() -> None:
    """Verifies compute_sw_ndcg_from_stage_results() against a manual
    weighted-sum calculation for a single K."""
    stage_results = {
        "S1": {5: 0.60},
        "S2": {5: 0.80},
        "S3": {5: 1.00},
    }
    expected = 0.20 * 0.60 + 0.30 * 0.80 + 0.50 * 1.00  # = 0.86
    result = compute_sw_ndcg_from_stage_results(stage_results, k_values=(5,))
    assert abs(result[5] - expected) < 1e-9, f"expected {expected}, got {result[5]}"
    print(f"[PASS] (1) hand-computed weighted sum matches: SW-NDCG@5={result[5]:.4f}")


def test_weight_validation_rejects_bad_sum() -> None:
    stage_results = {"S1": {5: 0.5}, "S2": {5: 0.5}, "S3": {5: 0.5}}
    try:
        compute_sw_ndcg_from_stage_results(
            stage_results, weights={"S1": 0.20, "S2": 0.30, "S3": 0.40}, k_values=(5,)
        )
    except ValueError as exc:
        print(f"[PASS] (2a) weight-sum validation raised as expected: {exc}")
        return
    raise AssertionError("Expected ValueError for weights summing to 0.90, not 1.0")


def test_weight_validation_rejects_bad_keys() -> None:
    stage_results = {"S1": {5: 0.5}, "S2": {5: 0.5}, "S3": {5: 0.5}}
    try:
        compute_sw_ndcg_from_stage_results(
            stage_results, weights={"S1": 0.20, "S2": 0.30, "S4": 0.50}, k_values=(5,)
        )
    except ValueError as exc:
        print(f"[PASS] (2b) weight-key validation raised as expected: {exc}")
        return
    raise AssertionError("Expected ValueError for weights covering {S1,S2,S4} != {S1,S2,S3}")


def test_default_weights_match_locked_config() -> None:
    assert DEFAULT_STAGE_WEIGHTS == {"S1": 0.20, "S2": 0.30, "S3": 0.50}, (
        f"DEFAULT_STAGE_WEIGHTS={DEFAULT_STAGE_WEIGHTS} does not match locked "
        "project configuration {S1: 0.20, S2: 0.30, S3: 0.50}"
    )
    print("[PASS] (3) default weights match locked config (S1=0.20, S2=0.30, S3=0.50)")


# --------------------------------------------------------------------------- #
# Integration tests with trained models on the v3 unsaturated dataset
# --------------------------------------------------------------------------- #

def test_cafe_sw_ndcg_in_range() -> None:
    bundle, config = _setup()
    model = _build_and_train_cafe(bundle, config)
    results = sw_ndcg(model, bundle, is_cafe=True)

    assert set(results.keys()) == set(K_VALUES), "Missing K values in SW-NDCG results"
    for k, score in results.items():
        assert 0.0 <= score <= 1.0, f"CAFE SW-NDCG@{k}={score} out of [0,1]"
        assert not (score != score), f"CAFE SW-NDCG@{k} is NaN"

    print_sw_ndcg_table(results, "CAFE-LightFM")
    print("[PASS] (4a) CAFE-LightFM: all SW-NDCG@K values finite and in [0,1]")


def test_baseline_sw_ndcg_in_range() -> None:
    bundle, config = _setup()
    model = _build_and_train_baseline(bundle, config)
    results = sw_ndcg(model, bundle, is_cafe=False)

    assert set(results.keys()) == set(K_VALUES)
    for k, score in results.items():
        assert 0.0 <= score <= 1.0, f"Baseline SW-NDCG@{k}={score} out of [0,1]"
        assert not (score != score), f"Baseline SW-NDCG@{k} is NaN"

    print_sw_ndcg_table(results, "LightFM-Baseline")
    print("[PASS] (4b) LightFM-Baseline: all SW-NDCG@K values finite and in [0,1]")


def test_sw_ndcg_entry_points_consistent() -> None:
    """sw_ndcg() (end-to-end) must equal
    compute_sw_ndcg_from_stage_results(stage_wise_ndcg(...)) (manual
    two-step call) exactly, for the same model/bundle."""
    bundle, config = _setup()
    model = _build_and_train_cafe(bundle, config)

    direct = sw_ndcg(model, bundle, is_cafe=True)
    stage_results = stage_wise_ndcg(model, bundle, is_cafe=True)
    manual = compute_sw_ndcg_from_stage_results(stage_results)

    for k in K_VALUES:
        assert abs(direct[k] - manual[k]) < 1e-9, (
            f"sw_ndcg() and manual two-step composition disagree at K={k}: "
            f"{direct[k]} != {manual[k]}"
        )
    print("[PASS] (5) sw_ndcg() and two-step composition are numerically consistent")


if __name__ == "__main__":
    test_hand_computed_weighted_sum()
    test_weight_validation_rejects_bad_sum()
    test_weight_validation_rejects_bad_keys()
    test_default_weights_match_locked_config()
    test_cafe_sw_ndcg_in_range()
    test_baseline_sw_ndcg_in_range()
    test_sw_ndcg_entry_points_consistent()
    print("=== ALL PHASE 3 STEP 2 SW-NDCG SMOKE TESTS PASSED ===")
