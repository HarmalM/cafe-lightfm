"""
test_synthetic_generator_v3_smoke.py

Smoke tests for data/synthetic_generator_v3.py (Phase 3, Step 1.5).

Validates the 10 required conditions from the Step 1.5 decision log.
Tests 1-9 are self-contained (schema + interaction_matrix only). Test 10
trains the LightFM baseline via experiments.training_loop.train_baseline
(confirmed API, Phase 2 Step 4) and asserts its loss is not identically
0.0 across all 10 epochs -- the empirical confirmation that the
saturation fix from Step 1 actually resolves the training-blocking
issue, not just the static negative-count guarantees checked by tests
5-7.

Run: python -m tests.test_synthetic_generator_v3_smoke
"""

from __future__ import annotations

from data.interaction_schema import StageLabelSource
from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v3 import (
    COVERAGE_FRACTION,
    LEAK_PROBABILITY,
    MASTER_SEED,
    N_ITEMS_V3,
    generate_synthetic_dataset_v3,
)

N_USERS = 20
N_SESSIONS = 3
EVENTS_PER_SESSION = 20


def test_reproducibility() -> None:
    a = generate_synthetic_dataset_v3(N_USERS, N_SESSIONS, EVENTS_PER_SESSION,
                                       LEAK_PROBABILITY, COVERAGE_FRACTION, N_ITEMS_V3, MASTER_SEED)
    b = generate_synthetic_dataset_v3(N_USERS, N_SESSIONS, EVENTS_PER_SESSION,
                                       LEAK_PROBABILITY, COVERAGE_FRACTION, N_ITEMS_V3, MASTER_SEED)
    assert a[0].identifiers.item_id == b[0].identifiers.item_id
    assert a[-1].stage.stage_true == b[-1].stage.stage_true
    print("[PASS] (1) seed=42 reproducibility holds")


def test_uses_interaction_record_schema() -> None:
    data = generate_synthetic_dataset_v3()
    r = data[0]
    assert hasattr(r, "identifiers") and hasattr(r, "categorical")
    assert hasattr(r, "continuous") and hasattr(r, "time") and hasattr(r, "stage")
    assert r.stage.stage_label_source == StageLabelSource.SYNTHETIC_GROUND_TRUTH
    print("[PASS] (2) records conform to InteractionRecord schema")


def test_stages_exactly_s1_s2_s3() -> None:
    data = generate_synthetic_dataset_v3()
    stages = {r.stage.stage_true.value for r in data}
    assert stages == {"S1", "S2", "S3"}
    print("[PASS] (3) stages present are exactly {S1, S2, S3}")


def test_leak_rate_approx_020() -> None:
    from data.synthetic_generator import SESSION_INDEX_TO_STAGE

    data = generate_synthetic_dataset_v3()
    leaked = 0
    for r in data:
        session_idx = int(r.identifiers.session_id.split("_s")[-1])
        nominal = SESSION_INDEX_TO_STAGE[session_idx]
        if r.stage.stage_true != nominal:
            leaked += 1
    rate = leaked / len(data)
    assert abs(rate - LEAK_PROBABILITY) < 0.05, f"leak rate {rate:.3f} too far from {LEAK_PROBABILITY}"
    print(f"[PASS] (4) empirical leak rate={rate:.3f} (target {LEAK_PROBABILITY})")


def test_users_not_globally_saturated() -> None:
    data = generate_synthetic_dataset_v3()
    bundle = build_interaction_matrix(data)
    for u_idx in range(bundle.n_users):
        items_touched = {i for (u, i) in bundle.positive_pairs if u == u_idx}
        assert len(items_touched) < bundle.n_items, (
            f"user_idx={u_idx} touched {len(items_touched)}/{bundle.n_items} items "
            "(globally saturated)"
        )
    print("[PASS] (5) no user is globally saturated (unique items < catalog size)")


def test_every_user_has_global_negative() -> None:
    data = generate_synthetic_dataset_v3()
    bundle = build_interaction_matrix(data)
    for u_idx in range(bundle.n_users):
        items_touched = {i for (u, i) in bundle.positive_pairs if u == u_idx}
        n_negatives = bundle.n_items - len(items_touched)
        assert n_negatives >= 1, f"user_idx={u_idx} has zero valid global negatives"
    print("[PASS] (6) every user has >=1 valid global negative item")


def test_every_user_stage_pair_has_stage_negative() -> None:
    data = generate_synthetic_dataset_v3()
    bundle = build_interaction_matrix(data)
    for stage, pairs in bundle.positive_pairs_by_stage.items():
        by_user = {}
        for (u, i) in pairs:
            by_user.setdefault(u, set()).add(i)
        for u_idx, items_touched in by_user.items():
            n_negatives = bundle.n_items - len(items_touched)
            assert n_negatives >= 1, f"user_idx={u_idx}, stage={stage} has zero stage negatives"
    print("[PASS] (7) every (user, stage) pair has >=1 valid stage-specific negative")


def test_full_catalog_represented() -> None:
    """Confirms the 30-item catalog is actually represented across the
    generated dataset. build_interaction_matrix() only indexes OBSERVED
    items, so this check is not implied by N_ITEMS_V3 alone -- it must be
    verified against the actual records and the resulting bundle."""
    data = generate_synthetic_dataset_v3()
    observed_items = {r.identifiers.item_id for r in data}
    assert len(observed_items) == N_ITEMS_V3, (
        f"observed {len(observed_items)} distinct items, expected all {N_ITEMS_V3} "
        "to appear at least once across the full dataset"
    )
    bundle = build_interaction_matrix(data)
    assert bundle.n_items == N_ITEMS_V3, (
        f"bundle.n_items={bundle.n_items} != N_ITEMS_V3={N_ITEMS_V3}"
    )
    print(f"[PASS] (2b) full {N_ITEMS_V3}-item catalog is represented in the dataset and bundle")


def test_positive_pairs_not_full_catalog() -> None:
    data = generate_synthetic_dataset_v3()
    bundle = build_interaction_matrix(data)
    max_possible = bundle.n_users * bundle.n_items
    assert len(bundle.positive_pairs) < max_possible, (
        f"positive_pairs ({len(bundle.positive_pairs)}) covers the full "
        f"user-item catalog ({max_possible})"
    )
    print(f"[PASS] (8) positive_pairs={len(bundle.positive_pairs)} < full catalog={max_possible}")


def test_positive_pairs_by_stage_nonempty_all_stages() -> None:
    data = generate_synthetic_dataset_v3()
    bundle = build_interaction_matrix(data)
    assert set(bundle.positive_pairs_by_stage.keys()) == {"S1", "S2", "S3"}
    for stage, pairs in bundle.positive_pairs_by_stage.items():
        assert len(pairs) > 0, f"stage={stage} has empty positive_pairs_by_stage"
    print("[PASS] (9) positive_pairs_by_stage is non-empty for all of {S1, S2, S3}")


def test_baseline_loss_not_always_zero() -> None:
    """
    Confirms the structural fix actually resolves the Step 1 finding: with
    negatives now guaranteed (see tests 5-7), the LightFM baseline's WARP
    loss must NOT be identically 0.0 across all 10 epochs, as it was on
    the fully-saturated Step 4 (v2) dataset.
    """
    from models.baselines.lightfm_pytorch import LightFMPyTorch
    from experiments.training_loop import TrainingConfig, train_baseline

    data = generate_synthetic_dataset_v3()
    bundle = build_interaction_matrix(data)
    config = TrainingConfig(n_epochs=10, seed=MASTER_SEED)

    model = LightFMPyTorch(
        bundle.n_users,
        bundle.n_items,
        bundle.n_categories,
        bundle.n_programs,
        config.embedding_dim,
    )
    log = train_baseline(model, bundle, config)
    losses = [r.mean_loss for r in log.epochs]

    assert len(losses) == 10, f"expected 10 epoch results, got {len(losses)}"
    assert any(l > 1e-8 for l in losses), (
        "baseline loss remained ~0.0 across all epochs -- saturation fix failed"
    )
    print(
        f"[PASS] (10) baseline loss not identically 0.0 across epochs "
        f"(losses={[round(l, 4) for l in losses]})"
    )


if __name__ == "__main__":
    test_reproducibility()
    test_uses_interaction_record_schema()
    test_stages_exactly_s1_s2_s3()
    test_leak_rate_approx_020()
    test_full_catalog_represented()
    test_users_not_globally_saturated()
    test_every_user_has_global_negative()
    test_every_user_stage_pair_has_stage_negative()
    test_positive_pairs_not_full_catalog()
    test_positive_pairs_by_stage_nonempty_all_stages()
    test_baseline_loss_not_always_zero()
    print("=== ALL 10 REQUIRED CHECKS PASSED (+1 supplementary catalog-coverage check) ===")
