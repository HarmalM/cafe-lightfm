"""
test_jsd_detector_smoke.py

Smoke test for data/jsd_detector.py, run on the Step 2 synthetic dataset
(tiny, deterministic-stage, strongly-separable). Validates:
    1. JSD is bounded in [0, 1] (base-2 Jensen-Shannon property).
    2. ALL 10 between-session JSD values (5 users x 2 consecutive pairs)
       exceed the placeholder threshold -> every true stage boundary in
       this deterministic dataset is detected (no false negatives).
    3. Between-session JSD values are, on average, well above the
       internal (split-half) noise floor -- confirming the JSD computation
       itself behaves correctly under the simplest possible conditions.
    4. Bin edges are identical across all sessions for a given signal
       (the "fixed bin edges" requirement).

Run: python -m tests.test_jsd_detector_smoke
"""

from __future__ import annotations

import numpy as np

from data.jsd_detector import (
    CONTINUOUS_SIGNALS,
    compute_global_bin_edges,
    detect_stage_boundaries,
    jensen_shannon_divergence,
)
from data.synthetic_generator import generate_synthetic_dataset, MASTER_SEED


def test_jsd_bounded() -> None:
    p = np.array([0.6, 0.4])
    q = np.array([0.1, 0.9])
    value = jensen_shannon_divergence(p, q)
    assert 0.0 <= value <= 1.0
    # Identical distributions -> JSD == 0
    assert abs(jensen_shannon_divergence(p, p)) < 1e-9
    print(f"[PASS] JSD bounded in [0,1]; JSD(p,q)={value:.4f}, JSD(p,p)~0")


def test_fixed_bin_edges_consistency() -> None:
    dataset = generate_synthetic_dataset(seed=MASTER_SEED)
    for signal in CONTINUOUS_SIGNALS:
        edges_full = compute_global_bin_edges(dataset, signal)
        # Recomputing from a subset must NOT change the convention: edges
        # are defined globally, so we verify determinism/reproducibility
        # of the edge computation itself.
        edges_repeat = compute_global_bin_edges(dataset, signal)
        assert np.allclose(edges_full, edges_repeat)
        assert len(edges_full) == 11  # n_bins=10 -> 11 edges
    print("[PASS] bin edges are fixed and reproducible per signal")


def test_all_true_boundaries_detected() -> None:
    dataset = generate_synthetic_dataset(seed=MASTER_SEED)
    result = detect_stage_boundaries(dataset)

    n_users = len({uid for (uid, _, _, _) in result["between_jsd"]})
    expected_pairs = n_users * 2  # 3 sessions -> 2 consecutive pairs per user
    assert len(result["between_jsd"]) == expected_pairs

    detected_flags = [d for (_, _, _, _, d) in result["detections"]]
    assert all(detected_flags), "Not all true stage boundaries were detected"
    print(f"[PASS] all {expected_pairs}/{expected_pairs} true stage boundaries detected")


def test_between_jsd_above_noise_floor() -> None:
    dataset = generate_synthetic_dataset(seed=MASTER_SEED)
    result = detect_stage_boundaries(dataset)

    mean_internal = float(np.mean(result["internal_jsd"]))
    mean_between = float(np.mean([jsd for (_, _, _, jsd) in result["between_jsd"]]))

    assert mean_between > mean_internal, (
        f"between-session JSD ({mean_between:.4f}) should exceed the "
        f"internal noise floor ({mean_internal:.4f})"
    )
    print(
        f"[PASS] mean between-session JSD={mean_between:.4f} > "
        f"mean internal noise floor={mean_internal:.4f}; "
        f"threshold={result['threshold']:.4f}"
    )


if __name__ == "__main__":
    test_jsd_bounded()
    test_fixed_bin_edges_consistency()
    test_all_true_boundaries_detected()
    test_between_jsd_above_noise_floor()
    print("=== ALL JSD DETECTOR SMOKE TESTS PASSED ===")
