"""
test_synthetic_generator_smoke.py

Smoke test for data/synthetic_generator.py. Validates:
    1. Record count matches n_users * n_sessions * events_per_session.
    2. Every record's stage_true matches the deterministic
       SESSION_INDEX_TO_STAGE mapping for its session.
    3. No record carries stage_self_report (leakage guard holds for
       every synthetic record, not just the schema-level unit test).
    4. Empirical Cohen's d (computed from the GENERATED samples, not the
       generator's input parameters) is >= 1.5 between adjacent stages
       for hover_duration_ms, confirming the dataset is, in practice,
       strongly separable as intended.

Run: python -m tests.test_synthetic_generator_smoke
"""

from __future__ import annotations

import numpy as np

from data.interaction_schema import DecisionStage, StageLabelSource
from data.synthetic_generator import (
    MASTER_SEED,
    SESSION_INDEX_TO_STAGE,
    generate_synthetic_dataset,
)

N_USERS = 5
N_SESSIONS = 3
EVENTS_PER_SESSION = 10


def _empirical_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (Cohen, 1988)."""
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return abs(a.mean() - b.mean()) / pooled_std


def test_record_count() -> None:
    data = generate_synthetic_dataset(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, MASTER_SEED)
    expected = N_USERS * N_SESSIONS * EVENTS_PER_SESSION
    assert len(data) == expected, f"expected {expected}, got {len(data)}"
    print(f"[PASS] record count == {expected}")


def test_deterministic_stage_assignment() -> None:
    data = generate_synthetic_dataset(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, MASTER_SEED)
    for record in data:
        session_idx = int(record.identifiers.session_id.split("_s")[-1])
        expected_stage = SESSION_INDEX_TO_STAGE[session_idx]
        assert record.stage.stage_true == expected_stage
    print("[PASS] every record's stage_true matches its session index")


def test_no_leakage() -> None:
    data = generate_synthetic_dataset(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, MASTER_SEED)
    for record in data:
        assert record.stage.stage_label_source == StageLabelSource.SYNTHETIC_GROUND_TRUTH
        assert record.stage.stage_self_report is None
    print("[PASS] no record carries stage_self_report")


def test_empirical_separability() -> None:
    data = generate_synthetic_dataset(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, MASTER_SEED)

    by_stage = {DecisionStage.EXPLORATION: [], DecisionStage.COMPARISON: [], DecisionStage.FINALIZATION: []}
    for record in data:
        by_stage[record.stage.stage_true].append(record.continuous.hover_duration_ms)

    s1 = np.array(by_stage[DecisionStage.EXPLORATION])
    s2 = np.array(by_stage[DecisionStage.COMPARISON])
    s3 = np.array(by_stage[DecisionStage.FINALIZATION])

    d_12 = _empirical_cohens_d(s1, s2)
    d_23 = _empirical_cohens_d(s2, s3)

    assert d_12 >= 1.5, f"S1-S2 empirical d={d_12:.2f} < 1.5"
    assert d_23 >= 1.5, f"S2-S3 empirical d={d_23:.2f} < 1.5"
    print(f"[PASS] empirical Cohen's d: S1-S2={d_12:.2f}, S2-S3={d_23:.2f} (both >= 1.5)")


def test_reproducibility() -> None:
    data_a = generate_synthetic_dataset(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, MASTER_SEED)
    data_b = generate_synthetic_dataset(N_USERS, N_SESSIONS, EVENTS_PER_SESSION, MASTER_SEED)
    assert data_a[0].continuous.hover_duration_ms == data_b[0].continuous.hover_duration_ms
    assert data_a[-1].identifiers.item_id == data_b[-1].identifiers.item_id
    print("[PASS] identical seed (42) reproduces identical dataset")


if __name__ == "__main__":
    test_record_count()
    test_deterministic_stage_assignment()
    test_no_leakage()
    test_empirical_separability()
    test_reproducibility()
    print("=== ALL SYNTHETIC GENERATOR SMOKE TESTS PASSED ===")
