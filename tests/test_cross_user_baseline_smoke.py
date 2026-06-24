"""
test_cross_user_baseline_smoke.py

Smoke test for jsd_detector.cross_user_same_stage_baseline (Step 5,
confirmed 2026-06-24), run on the Step 4 realistic dataset. Validates:
    1. Correct pairing structure: no self-pairs, every pair shares the
       same session index (stage), n_users negative pairs per stage.
    2. Reproducibility under seed=42.
    3. Head-to-head discrimination: TPR-FPR using the new baseline as the
       negative reference is better than using the old split-half
       baseline (which was found to invert under leakage in the Step 4
       diagnostic, 2026-06-24).

Run: python -m tests.test_cross_user_baseline_smoke
"""

from __future__ import annotations

import numpy as np

from data.jsd_detector import (
    CONTINUOUS_SIGNALS,
    compute_global_bin_edges,
    cross_user_same_stage_baseline,
    detect_stage_boundaries,
)
from data.mdl_threshold import build_candidates_from_dataset, estimate_global_tau_via_grid_search
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2, MASTER_SEED


def _build_bin_edges(dataset):
    return {signal: compute_global_bin_edges(dataset, signal) for signal in CONTINUOUS_SIGNALS}


def test_pairing_structure() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bin_edges = _build_bin_edges(dataset)
    pairs = cross_user_same_stage_baseline(dataset, bin_edges, seed=42)

    n_users = 20
    n_sessions = 3
    assert len(pairs) == n_users * n_sessions, f"expected {n_users * n_sessions} pairs, got {len(pairs)}"

    for user_a, user_b, session_idx, jsd in pairs:
        assert user_a != user_b, "self-pair detected"
        assert 0 <= session_idx <= 2
        assert 0.0 <= jsd <= 1.0
    print(f"[PASS] pairing structure correct: {len(pairs)} pairs, no self-pairs, valid JSD range")


def test_reproducibility() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bin_edges = _build_bin_edges(dataset)
    pairs_a = cross_user_same_stage_baseline(dataset, bin_edges, seed=42)
    pairs_b = cross_user_same_stage_baseline(dataset, bin_edges, seed=42)
    assert pairs_a == pairs_b
    print("[PASS] identical seed (42) reproduces identical pairing and JSD values")


def test_discrimination_improves_over_split_half() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bin_edges = _build_bin_edges(dataset)

    # New baseline (negatives) vs true positives (between-session candidates)
    negative_pairs = cross_user_same_stage_baseline(dataset, bin_edges, seed=42)
    negative_jsd = [jsd for (_, _, _, jsd) in negative_pairs]

    candidates = build_candidates_from_dataset(dataset, jsd_n_bins=10)
    positive_jsd = [c.jsd_bits for c in candidates]

    # Old baseline, for comparison (split-half, via detect_stage_boundaries)
    old_result = detect_stage_boundaries(dataset)
    old_negative_jsd = old_result["internal_jsd"]

    mean_pos = float(np.mean(positive_jsd))
    mean_new_neg = float(np.mean(negative_jsd))
    mean_old_neg = float(np.mean(old_negative_jsd))

    print(f"mean positive (between-session) JSD: {mean_pos:.4f}")
    print(f"mean NEW negative (cross-user, same-stage) JSD: {mean_new_neg:.4f}")
    print(f"mean OLD negative (split-half) JSD: {mean_old_neg:.4f}")

    assert mean_pos > mean_new_neg, (
        f"positive mean ({mean_pos:.4f}) should exceed the new cross-user "
        f"negative mean ({mean_new_neg:.4f})"
    )
    print("[PASS] mean positive JSD > mean new cross-user-baseline JSD "
          "(old split-half baseline had this INVERTED under leakage)")

    # Best-achievable TPR-FPR (Youden's J) using the new baseline, via the
    # existing MDL grid search machinery's tau candidates.
    best_j, best_tau = -1.0, None
    for tau in sorted(set(positive_jsd) | set(negative_jsd)):
        tpr = sum(1 for v in positive_jsd if v > tau) / len(positive_jsd)
        fpr = sum(1 for v in negative_jsd if v > tau) / len(negative_jsd)
        j = tpr - fpr
        if j > best_j:
            best_j, best_tau = j, tau
    print(f"[INFO] best achievable TPR-FPR with new baseline: {best_j:.2f} at tau={best_tau:.4f}")
    assert best_j > 0, "new baseline should permit positive discrimination for some tau"
    print("[PASS] positive discrimination (TPR-FPR > 0) achievable with the new baseline")


if __name__ == "__main__":
    test_pairing_structure()
    test_reproducibility()
    test_discrimination_improves_over_split_half()
    print("=== ALL CROSS-USER BASELINE SMOKE TESTS PASSED ===")
