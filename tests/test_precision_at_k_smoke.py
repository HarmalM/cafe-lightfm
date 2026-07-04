"""
test_precision_at_k_smoke.py

Phase 3, Step 3: smoke tests for stage-stratified Precision@K.

Test plan (confirmed 2026-07-04):
    1. Perfect ranking      -> Precision@K = min(num_positives, K) / K
    2. Worst ranking        -> Precision@K = 0.0
    3. No relevant items    -> Precision@K = 0.0
    4. Integration: CAFE-LightFM on synthetic_generator_v3 (unsaturated)
    5. Integration: LightFM baseline on synthetic_generator_v3
    6. All returned values finite and in [0, 1]
    7. Stage coverage confirmed exactly {S1, S2, S3} x {5, 10, 20}

Dataset scope: synthetic_generator_v3.py (unsaturated), per the binding
Phase 3 Step 1.5 scope boundary. Not v2.

Validation-only: no scientific claim is drawn from the numeric results
of these tests; they establish correctness and numerical stability of
the metric implementation only.
"""

from __future__ import annotations

import math

import pytest

from experiments.ndcg import K_VALUES, STAGE_ORDER
from experiments.precision_at_k import (
    compute_precision_at_k,
    stage_wise_precision,
)


# --------------------------------------------------------------------------- #
# 1. Perfect ranking
# --------------------------------------------------------------------------- #

def test_precision_at_k_perfect_ranking():
    """
    All relevant items ranked first. For K <= num_positives, Precision@K
    must equal 1.0. For K > num_positives, Precision@K must equal
    num_positives / K (fixed denominator, confirmed 2026-07-04).
    """
    # 3 relevant items ranked first, then 7 irrelevant.
    ranked_relevance = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    assert compute_precision_at_k(ranked_relevance, 3) == pytest.approx(1.0)
    for k in (5, 10):
        expected = min(3, k) / k
        assert compute_precision_at_k(ranked_relevance, k) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 2. Worst ranking
# --------------------------------------------------------------------------- #

def test_precision_at_k_worst_ranking():
    """All relevant items ranked last -> Precision@K = 0.0 for small K."""
    ranked_relevance = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    for k in (3, 5):
        assert compute_precision_at_k(ranked_relevance, k) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. No relevant items
# --------------------------------------------------------------------------- #

def test_precision_at_k_no_relevant_items():
    """No relevant items anywhere -> Precision@K = 0.0 for all K, no error."""
    ranked_relevance = [0] * 10
    for k in K_VALUES:
        assert compute_precision_at_k(ranked_relevance, k) == pytest.approx(0.0)


def test_precision_at_k_rejects_non_positive_k():
    """k <= 0 must raise ValueError (fixed denominator cannot be <= 0)."""
    with pytest.raises(ValueError):
        compute_precision_at_k([1, 0, 1], 0)


# --------------------------------------------------------------------------- #
# 4-7. Integration tests (require a trained model + v3 bundle)
#
# These fixtures ("cafe_model_v3", "baseline_model_v3", "bundle_v3") are
# assumed to be provided by the project's existing conftest.py / fixture
# module used for Phase 3 Step 1/2 integration tests. They are referenced
# here by name only; no ranking or training logic is duplicated in this
# file. If these fixtures do not yet exist under this name, they must be
# added (or pointed at the existing v3 fixtures) before this test module
# can run end-to-end.
# --------------------------------------------------------------------------- #

def test_precision_at_k_cafe_lightfm_v3(cafe_model_v3, bundle_v3):
    """Integration: CAFE-LightFM on the unsaturated v3 dataset."""
    results = stage_wise_precision(cafe_model_v3, bundle_v3, is_cafe=True)
    _assert_valid_results_shape(results)


def test_precision_at_k_baseline_v3(baseline_model_v3, bundle_v3):
    """Integration: LightFM baseline on the unsaturated v3 dataset."""
    results = stage_wise_precision(baseline_model_v3, bundle_v3, is_cafe=False)
    _assert_valid_results_shape(results)


# --------------------------------------------------------------------------- #
# Shared shape/range/coverage assertion (used by both integration tests)
# --------------------------------------------------------------------------- #

def _assert_valid_results_shape(results):
    """
    Validates:
      - stage coverage is exactly {S1, S2, S3}
      - K coverage per stage is exactly {5, 10, 20}
      - every value is finite and in [0, 1]
    """
    assert set(results.keys()) == set(STAGE_ORDER)
    for stage_key in STAGE_ORDER:
        assert set(results[stage_key].keys()) == set(K_VALUES)
        for k in K_VALUES:
            value = results[stage_key][k]
            assert math.isfinite(value)
            assert 0.0 <= value <= 1.0
