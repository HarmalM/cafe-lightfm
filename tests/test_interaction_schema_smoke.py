"""
test_interaction_schema_smoke.py

Smoke test for interaction_schema.py. Validates:
    1. A synthetic record builds correctly (stage_true set, others None).
    2. A real-participant (pilot) record builds correctly (stage_self_report
       set, stage_true forced None).
    3. The leakage guard raises ValueError if stage_true is set on a
       SELF_REPORT_PILOT record (i.e. real data is never allowed to carry
       synthetic-only ground truth).
    4. to_flat_dict() flattens without error for both record types.

Run: python test_interaction_schema_smoke.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from data.interaction_schema import (
    CategoricalFeatures,
    ContinuousSignals,
    DecisionStage,
    Identifiers,
    InteractionRecord,
    InteractionType,
    StageLabels,
    StageLabelSource,
    TimeFields,
)


def _build_identifiers(suffix: str) -> Identifiers:
    return Identifiers(
        user_id=f"u_{suffix}",
        item_id=f"i_{suffix}",
        session_id=f"sess_{suffix}",
        event_id=f"evt_{suffix}",
    )


def _build_categorical() -> CategoricalFeatures:
    return CategoricalFeatures(
        item_category="university",
        program_type="MSc_CS",
        interaction_type=InteractionType.HOVER,
    )


def _build_continuous() -> ContinuousSignals:
    return ContinuousSignals(
        hover_duration_ms=1200.0,
        dwell_time_ms=4300.0,
        revisit_count=2,
        scroll_depth_pct=65.0,
    )


def _build_time(start: datetime, idx: int) -> TimeFields:
    ts = start + timedelta(seconds=idx * 30)
    return TimeFields(
        timestamp=ts,
        event_index=idx,
        time_since_start_s=float(idx * 30),
        time_since_previous_event_s=None if idx == 0 else 30.0,
    )


def test_synthetic_record_builds() -> None:
    record = InteractionRecord(
        identifiers=_build_identifiers("syn001"),
        categorical=_build_categorical(),
        continuous=_build_continuous(),
        time=_build_time(datetime(2026, 6, 24), 0),
        stage=StageLabels(
            stage_true=DecisionStage.COMPARISON,
            stage_label_source=StageLabelSource.SYNTHETIC_GROUND_TRUTH,
        ),
    )
    assert record.is_synthetic()
    assert record.stage.stage_true == DecisionStage.COMPARISON
    assert record.stage.stage_self_report is None
    print("[PASS] synthetic record builds, stage_true set, self_report None")


def test_pilot_record_builds() -> None:
    record = InteractionRecord(
        identifiers=_build_identifiers("pilot001"),
        categorical=_build_categorical(),
        continuous=_build_continuous(),
        time=_build_time(datetime(2026, 6, 24), 1),
        stage=StageLabels(
            stage_self_report=DecisionStage.EXPLORATION,
            stage_label_source=StageLabelSource.SELF_REPORT_PILOT,
        ),
    )
    assert not record.is_synthetic()
    assert record.stage.stage_true is None
    assert record.stage.stage_self_report == DecisionStage.EXPLORATION
    print("[PASS] pilot record builds, stage_true None, self_report set")


def test_leakage_guard_raises() -> None:
    try:
        StageLabels(
            stage_true=DecisionStage.FINALIZATION,       # forbidden here
            stage_self_report=DecisionStage.EXPLORATION,
            stage_label_source=StageLabelSource.SELF_REPORT_PILOT,
        )
    except ValueError as exc:
        print(f"[PASS] leakage guard raised as expected: {exc}")
        return
    raise AssertionError(
        "Expected ValueError: stage_true must not be set on a "
        "SELF_REPORT_PILOT record."
    )


def test_flatten() -> None:
    record = InteractionRecord(
        identifiers=_build_identifiers("flat001"),
        categorical=_build_categorical(),
        continuous=_build_continuous(),
        time=_build_time(datetime(2026, 6, 24), 2),
        stage=StageLabels(
            stage_inferred_jsd=DecisionStage.COMPARISON,
            stage_label_source=StageLabelSource.JSD_INFERRED,
        ),
    )
    flat = record.to_flat_dict()
    assert flat["id__user_id"] == "u_flat001"
    assert flat["cat__interaction_type"] == "hover"
    assert flat["stage__stage_inferred_jsd"] == "S2"
    print("[PASS] to_flat_dict() flattens correctly with provenance prefixes")


if __name__ == "__main__":
    test_synthetic_record_builds()
    test_pilot_record_builds()
    test_leakage_guard_raises()
    test_flatten()
    print("=== ALL SCHEMA SMOKE TESTS PASSED ===")
