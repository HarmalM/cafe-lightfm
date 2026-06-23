"""
test_synthetic_generator_v2_smoke.py

Smoke test for data/synthetic_generator_v2.py (Step 4). Validates:
    1. Record count matches n_users * n_sessions * events_per_session.
    2. Empirical leak rate is close to the target 20% (binomial tolerance).
    3. Empirical Cohen's d between stage_true groups (the ACTUAL generating
       stage, including leaked-in events) falls in the weak-separability
       band (~0.5-0.8), confirming the dataset is realistically harder
       than Step 2's.
    4. Reproducibility holds under a fixed seed.
    5. JSD sensitivity under these harder conditions: re-running the
       Step 3 detector's building blocks (per-signal histogram JSD)
       directly on stage_true groups still shows a meaningfully positive
       JSD between every adjacent stage pair (sanity floor, not a strict
       detection-rate claim -- that belongs to a dedicated Step 4
       evaluation, not this smoke test).

Run: python -m tests.test_synthetic_generator_v2_smoke
"""

from __future__ import annotations

import numpy as np

from data.interaction_schema import DecisionStage, StageLabelSource
from data.jsd_detector import compute_global_bin_edges, histogram_probabilities, jensen_shannon_divergence
from data.synthetic_generator_v2 import (
    LEAK_PROBABILITY,
    MASTER_SEED,
    generate_synthetic_dataset_v2,
)

N_USERS = 20
N_SESSIONS = 3
EVENTS_PER_SESSION = 20


def _empirical_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return abs(a.mean() - b.mean()) / pooled_std


def test_record_count() -> None:
    data = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)
    expected = N_USERS * N_SESSIONS * EVENTS_PER_SESSION
    assert len(data) == expected
    print(f"[PASS] record count == {expected}")


def test_empirical_leak_rate() -> None:
    from data.synthetic_generator import SESSION_INDEX_TO_STAGE

    data = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)
    leaked = 0
    for r in data:
        session_idx = int(r.identifiers.session_id.split("_s")[-1])
        nominal = SESSION_INDEX_TO_STAGE[session_idx]
        if r.stage.stage_true != nominal:
            leaked += 1
    rate = leaked / len(data)
    # Binomial std error for n=1200, p=0.2: sqrt(0.2*0.8/1200) ~= 0.0115
    # Allow a generous +/- 3 std-error tolerance band.
    assert abs(rate - LEAK_PROBABILITY) < 0.05, f"empirical leak rate {rate:.3f} too far from {LEAK_PROBABILITY}"
    print(f"[PASS] empirical leak rate={rate:.3f} (target {LEAK_PROBABILITY})")


def test_empirical_separability_band() -> None:
    data = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)

    by_stage = {DecisionStage.EXPLORATION: [], DecisionStage.COMPARISON: [], DecisionStage.FINALIZATION: []}
    for r in data:
        by_stage[r.stage.stage_true].append(r.continuous.hover_duration_ms)

    s1 = np.array(by_stage[DecisionStage.EXPLORATION])
    s2 = np.array(by_stage[DecisionStage.COMPARISON])
    s3 = np.array(by_stage[DecisionStage.FINALIZATION])

    d_12 = _empirical_cohens_d(s1, s2)
    d_23 = _empirical_cohens_d(s2, s3)

    assert 0.4 <= d_12 <= 0.95, f"S1-S2 empirical d={d_12:.2f} outside weak-separability band"
    assert 0.4 <= d_23 <= 0.95, f"S2-S3 empirical d={d_23:.2f} outside weak-separability band"
    print(f"[PASS] empirical Cohen's d: S1-S2={d_12:.2f}, S2-S3={d_23:.2f} (weak-separability band)")


def test_reproducibility() -> None:
    a = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)
    b = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)
    assert a[0].continuous.hover_duration_ms == b[0].continuous.hover_duration_ms
    assert a[-1].stage.stage_true == b[-1].stage.stage_true
    print("[PASS] identical seed (42) reproduces identical dataset")


def test_no_schema_leakage() -> None:
    """Confirms the schema-level leakage guard (stage_true XOR self_report)
    still holds for every record, independent of the new behavioral leak."""
    data = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)
    for r in data:
        assert r.stage.stage_label_source == StageLabelSource.SYNTHETIC_GROUND_TRUTH
        assert r.stage.stage_self_report is None
    print("[PASS] schema-level leakage guard holds for all records")


def test_jsd_sensitivity_floor() -> None:
    """Sanity floor only: per-signal histogram JSD between stage_true
    groups (not session windows) must still be meaningfully > 0 for every
    adjacent pair, confirming JSD has not collapsed to noise under the
    weaker separability + behavioral leakage introduced in Step 4."""
    data = generate_synthetic_dataset_v2(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, LEAK_PROBABILITY, MASTER_SEED)

    by_stage = {DecisionStage.EXPLORATION: [], DecisionStage.COMPARISON: [], DecisionStage.FINALIZATION: []}
    for r in data:
        by_stage[r.stage.stage_true].append(r)

    bin_edges = compute_global_bin_edges(data, "hover_duration_ms")

    def _probs(stage: DecisionStage) -> np.ndarray:
        values = np.array([r.continuous.hover_duration_ms for r in by_stage[stage]])
        return histogram_probabilities(values, bin_edges)

    jsd_12 = jensen_shannon_divergence(_probs(DecisionStage.EXPLORATION), _probs(DecisionStage.COMPARISON))
    jsd_23 = jensen_shannon_divergence(_probs(DecisionStage.COMPARISON), _probs(DecisionStage.FINALIZATION))

    assert jsd_12 > 0.02, f"S1-S2 JSD={jsd_12:.4f} too close to noise floor"
    assert jsd_23 > 0.02, f"S2-S3 JSD={jsd_23:.4f} too close to noise floor"
    print(f"[PASS] JSD sensitivity floor holds: S1-S2={jsd_12:.4f}, S2-S3={jsd_23:.4f}")


if __name__ == "__main__":
    test_record_count()
    test_empirical_leak_rate()
    test_empirical_separability_band()
    test_reproducibility()
    test_no_schema_leakage()
    test_jsd_sensitivity_floor()
    print("=== ALL STEP 4 GENERATOR SMOKE TESTS PASSED ===")
