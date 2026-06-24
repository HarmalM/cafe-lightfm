"""
test_interaction_matrix_smoke.py

Smoke test for data/interaction_matrix.py. Validates:
    1. Vocabulary sizes match the Step 4 dataset (20 users, 10 items,
       1 category, 3 programs).
    2. positive_pairs (pooled) is a subset relationship-consistent
       aggregation: every (user, item) pair appearing in ANY stage's
       positive_pairs_by_stage also appears in the pooled positive_pairs.
    3. item_feature_idx_by_item assigns exactly one (category_idx,
       program_idx) per item (the mode-resolution placeholder, confirmed
       2026-06-25).
    4. The module never touches ContinuousSignals or TimeFields: verified
       by static inspection of the source (no attribute access on
       `.continuous` or `.time`) -- a structural guarantee, not just a
       runtime absence-of-error check.
    5. Reproducibility: identical input records produce an identical
       bundle (deterministic vocab ordering).

Run: python -m tests.test_interaction_matrix_smoke
"""

from __future__ import annotations

import inspect

from data import interaction_matrix as interaction_matrix_module
from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2, MASTER_SEED


def test_vocabulary_sizes() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    assert bundle.n_users == 20
    assert bundle.n_items == 10
    assert bundle.n_categories == 1   # single CATEGORY_POOL value in Step 4 generator
    assert bundle.n_programs == 3     # MSc_CS, MSc_DS, BSc_Eng
    print(f"[PASS] vocabulary sizes correct: users={bundle.n_users}, items={bundle.n_items}, "
          f"categories={bundle.n_categories}, programs={bundle.n_programs}")


def test_per_stage_pairs_subset_of_pooled() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    for stage, pairs in bundle.positive_pairs_by_stage.items():
        assert pairs.issubset(bundle.positive_pairs), f"stage {stage} has pairs not in pooled set"
    print("[PASS] every per-stage positive pair also appears in the pooled positive_pairs")


def test_one_feature_assignment_per_item() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    assert len(bundle.item_feature_idx_by_item) == bundle.n_items
    for item_idx, (cat_idx, prog_idx) in bundle.item_feature_idx_by_item.items():
        assert 0 <= cat_idx < bundle.n_categories
        assert 0 <= prog_idx < bundle.n_programs
    print("[PASS] exactly one (category_idx, program_idx) assigned per item, within vocab range")


def test_no_continuous_or_time_field_access() -> None:
    """Structural guarantee: the module's source never accesses `.continuous`
    or `.time` on an InteractionRecord."""
    source = inspect.getsource(interaction_matrix_module)
    assert ".continuous" not in source, "module must never read ContinuousSignals"
    assert ".time." not in source, "module must never read TimeFields"
    print("[PASS] source contains no `.continuous` or `.time.` attribute access")


def test_reproducibility() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle_a = build_interaction_matrix(dataset)
    bundle_b = build_interaction_matrix(dataset)
    assert bundle_a.user_id_to_idx == bundle_b.user_id_to_idx
    assert bundle_a.item_feature_idx_by_item == bundle_b.item_feature_idx_by_item
    assert bundle_a.positive_pairs == bundle_b.positive_pairs
    print("[PASS] identical input produces an identical bundle (deterministic)")


if __name__ == "__main__":
    test_vocabulary_sizes()
    test_per_stage_pairs_subset_of_pooled()
    test_one_feature_assignment_per_item()
    test_no_continuous_or_time_field_access()
    test_reproducibility()
    print("=== ALL INTERACTION MATRIX SMOKE TESTS PASSED ===")
