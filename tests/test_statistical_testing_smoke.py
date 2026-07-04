"""
tests/test_statistical_testing_smoke.py

Phase 3, Step 4 — Smoke tests for experiments/statistical_testing.py.

Self-contained per the pattern established in test_precision_at_k_smoke.py
(2026-07-04 session): no dependency on conftest.py fixtures. All test
inputs are synthetic numpy arrays constructed directly in this file, with
the project master seed (42) fixed for reproducibility per config.py.

These tests validate the STATISTICAL PRIMITIVES ONLY (paired t-test,
Cohen's d_z, Bonferroni correction). They do not evaluate CAFE-LightFM or
the LightFM baseline directly — that integration (feeding real per-user
NDCG@K / Precision@K arrays from experiments/ndcg.py and
experiments/precision_at_k.py into these functions) is a separate,
not-yet-confirmed step (see module docstring "zero re-implementation"
note and the outstanding integration-layer question raised alongside this
delivery).

Run with:
    pytest tests/test_statistical_testing_smoke.py -v
"""

import numpy as np
import pytest

from experiments.statistical_testing import (
    PairedTestResult,
    bonferroni_correction,
    cohens_dz,
    paired_t_test,
    summarize_paired_results,
)

SEED = 42  # project master seed (config.py)


# ---------------------------------------------------------------------------
# 1. cohens_dz
# ---------------------------------------------------------------------------

def test_cohens_dz_zero_for_identical_scores():
    """Identical paired scores -> zero variance in differences -> d_z = 0.0
    (not NaN)."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0, 1, size=20)
    b = a.copy()
    assert cohens_dz(a, b) == 0.0


def test_cohens_dz_matches_hand_calculation():
    """Hand-computed example: A = [2, 4, 6, 8], B = [1, 2, 3, 4].
    D = [1, 2, 3, 4] -> mean(D) = 2.5, std(D, ddof=1) = 1.290994...
    d_z = 2.5 / 1.290994 = 1.936492...
    """
    a = np.array([2.0, 4.0, 6.0, 8.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.mean(a - b) / np.std(a - b, ddof=1)
    result = cohens_dz(a, b)
    assert np.isclose(result, expected, atol=1e-9)
    assert np.isclose(result, 1.936492, atol=1e-5)


def test_cohens_dz_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        cohens_dz(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_cohens_dz_rejects_too_few_observations():
    with pytest.raises(ValueError):
        cohens_dz(np.array([1.0]), np.array([2.0]))


# ---------------------------------------------------------------------------
# 2. paired_t_test
# ---------------------------------------------------------------------------

def test_paired_t_test_no_difference_high_p_value():
    """When A and B are identical, t should be 0 (or nan under zero
    variance) and the differences carry no evidence against H0. We assert
    the mean_diff is exactly 0.0 and cohens_dz is 0.0, avoiding any
    assumption about scipy's NaN-handling convention for the t-statistic
    itself under zero variance."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0, 1, size=15)
    b = a.copy()
    result = paired_t_test(a, b, metric_name="identical_case")
    assert result.mean_diff == 0.0
    assert result.cohens_dz == 0.0
    assert result.n_pairs == 15
    assert result.degrees_of_freedom == 14


def test_paired_t_test_zero_variance_diff_returns_valid_p_not_nan():
    """Regression test (found during Step 4 integration testing,
    2026-07-05): when every paired difference is identical (here: all
    zero), scipy.stats.ttest_rel computes a 0/0 NaN p-value. This must
    be reported as t=0.0, p=1.0 instead, so the result remains a valid
    input to bonferroni_correction (which rejects NaN/out-of-range
    p-values by design)."""
    a = np.array([0.5, 0.3, 0.7, 0.1])
    b = np.array([0.5, 0.3, 0.7, 0.1])
    result = paired_t_test(a, b, metric_name="zero_variance_case")
    assert result.t_statistic == 0.0
    assert result.p_value == 1.0
    assert not np.isnan(result.p_value)
    assert result.cohens_dz == 0.0


def test_paired_t_test_detects_known_shift():
    """A is B plus a constant positive shift with small added noise ->
    expect a strongly significant two-tailed p-value (p < 0.001) and a
    positive t-statistic (A > B)."""
    rng = np.random.default_rng(SEED)
    b = rng.uniform(0.3, 0.5, size=50)
    noise = rng.normal(0, 0.01, size=50)
    a = b + 0.2 + noise  # consistent positive shift, small noise
    result = paired_t_test(a, b, metric_name="shifted_case")
    assert result.p_value < 0.001
    assert result.t_statistic > 0
    assert result.mean_diff > 0
    assert result.cohens_dz > 0


def test_paired_t_test_two_tailed_symmetry():
    """Swapping A and B should flip the sign of t and mean_diff but leave
    the two-tailed p-value unchanged (Decision B: two-tailed test)."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0.4, 0.9, size=30)
    b = rng.uniform(0.1, 0.6, size=30)
    result_ab = paired_t_test(a, b, metric_name="ab")
    result_ba = paired_t_test(b, a, metric_name="ba")
    assert np.isclose(result_ab.p_value, result_ba.p_value, atol=1e-9)
    assert np.isclose(result_ab.t_statistic, -result_ba.t_statistic, atol=1e-9)
    assert np.isclose(result_ab.mean_diff, -result_ba.mean_diff, atol=1e-9)


def test_paired_t_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_t_test(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_paired_t_test_rejects_too_few_observations():
    with pytest.raises(ValueError):
        paired_t_test(np.array([1.0]), np.array([2.0]))


# ---------------------------------------------------------------------------
# 3. bonferroni_correction (Decision C: dynamic M, with override)
# ---------------------------------------------------------------------------

def test_bonferroni_dynamic_m_equals_number_of_comparisons():
    """With n_comparisons=None, M must equal len(p_values) (Decision C:
    dynamic default)."""
    p_values = {"m1": 0.01, "m2": 0.20, "m3": 0.049}
    result = bonferroni_correction(p_values, alpha=0.05)
    expected_alpha_corrected = 0.05 / 3
    for name in p_values:
        assert np.isclose(
            result[name]["alpha_corrected"], expected_alpha_corrected
        )


def test_bonferroni_override_locks_to_63():
    """Passing n_comparisons=63 (the locked full-study value, proposal
    Section 5.4) must override the dynamic default regardless of how many
    p-values are actually supplied in this call."""
    p_values = {"m1": 0.0005, "m2": 0.02}
    result = bonferroni_correction(p_values, alpha=0.05, n_comparisons=63)
    expected_alpha_corrected = 0.05 / 63
    for name in p_values:
        assert np.isclose(
            result[name]["alpha_corrected"], expected_alpha_corrected
        )
    # 0.0005 < 0.05/63 (=0.000794) -> significant
    assert result["m1"]["significant"] is True
    # 0.02 > 0.05/63 -> not significant
    assert result["m2"]["significant"] is False


def test_bonferroni_rejects_empty_input():
    with pytest.raises(ValueError):
        bonferroni_correction({}, alpha=0.05)


def test_bonferroni_rejects_invalid_p_value_range():
    with pytest.raises(ValueError):
        bonferroni_correction({"m1": 1.5}, alpha=0.05)


def test_bonferroni_rejects_non_positive_n_comparisons():
    with pytest.raises(ValueError):
        bonferroni_correction({"m1": 0.01}, alpha=0.05, n_comparisons=0)


# ---------------------------------------------------------------------------
# 4. summarize_paired_results (aggregation, zero re-implementation check)
# ---------------------------------------------------------------------------

def test_summarize_paired_results_dynamic_m_matches_batch_size():
    rng = np.random.default_rng(SEED)
    results = []
    for i, shift in enumerate([0.3, 0.0, -0.1]):
        b = rng.uniform(0.2, 0.5, size=20)
        a = b + shift + rng.normal(0, 0.01, size=20)
        results.append(paired_t_test(a, b, metric_name=f"metric_{i}"))

    summary = summarize_paired_results(results, alpha=0.05)
    expected_alpha_corrected = 0.05 / 3  # dynamic M = 3
    for name in summary:
        assert np.isclose(summary[name]["alpha_corrected"], expected_alpha_corrected)
    # The strongly positive-shift metric should be significant.
    assert summary["metric_0"]["significant"] is True


def test_summarize_paired_results_rejects_duplicate_metric_names():
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0, 1, size=10)
    b = rng.uniform(0, 1, size=10)
    r1 = paired_t_test(a, b, metric_name="dup")
    r2 = paired_t_test(a, b, metric_name="dup")
    with pytest.raises(ValueError):
        summarize_paired_results([r1, r2])


def test_summarize_paired_results_rejects_empty_list():
    with pytest.raises(ValueError):
        summarize_paired_results([])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
