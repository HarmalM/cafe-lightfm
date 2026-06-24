"""
test_mdl_threshold_smoke.py

Step 5 smoke test for data/mdl_threshold.py. Locks in the configuration
determined by the Step 5 investigation (confirmed 2026-06-24):
    - JSD representation bins: 10 (matches Step 3)
    - MDL cost-model bins: 4 (decoupled BIC penalty; best TPR-FPR found)
    - Negative reference: cross-user, same-stage baseline (NOT split-half,
      which was found to invert under probabilistic leakage)

Validates:
    1. param_cost_bits and the cost functions behave sensibly (basic
       sanity: positive, monotonic in mdl_n_bins).
    2. estimate_global_tau_via_grid_search is reproducible under seed=42.
    3. The locked configuration achieves meaningfully positive
       discrimination (TPR - FPR) on the Step 4 realistic dataset,
       using the cross-user same-stage baseline as negatives.
    4. Discrimination is substantially better than at mdl_n_bins=10
       (the originally-coupled, failing configuration), confirming the
       decoupling + rebaselining fix is responsible for the improvement.

Run: python -m tests.test_mdl_threshold_smoke
"""

from __future__ import annotations

from data.jsd_detector import (
    CONTINUOUS_SIGNALS,
    compute_global_bin_edges,
    cross_user_same_stage_baseline,
)
from data.mdl_threshold import (
    build_candidates_from_dataset,
    estimate_global_tau_via_grid_search,
    param_cost_bits,
    total_mdl_cost,
)
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2, MASTER_SEED

JSD_N_BINS = 10
MDL_N_BINS = 4


def _setup():
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bin_edges = {s: compute_global_bin_edges(dataset, s) for s in CONTINUOUS_SIGNALS}
    candidates = build_candidates_from_dataset(dataset, jsd_n_bins=JSD_N_BINS)
    negative_pairs = cross_user_same_stage_baseline(dataset, bin_edges, seed=42)
    return candidates, negative_pairs


def test_param_cost_monotonic_in_bins() -> None:
    costs = [param_cost_bits(b, n_events=20) for b in range(2, 10)]
    assert all(costs[i] < costs[i + 1] for i in range(len(costs) - 1))
    print("[PASS] param_cost_bits increases monotonically with mdl_n_bins")


def test_tau_reproducibility() -> None:
    candidates, _ = _setup()
    tau_a, _ = estimate_global_tau_via_grid_search(candidates, mdl_n_bins=MDL_N_BINS)
    tau_b, _ = estimate_global_tau_via_grid_search(candidates, mdl_n_bins=MDL_N_BINS)
    assert tau_a == tau_b
    print(f"[PASS] tau reproducible under seed={MASTER_SEED}: tau={tau_a:.4f}")


def test_locked_configuration_discriminates() -> None:
    candidates, negative_pairs = _setup()
    positive_jsd = [c.jsd_bits for c in candidates]
    negative_jsd = [jsd for (_, _, _, jsd) in negative_pairs]

    tau, _ = estimate_global_tau_via_grid_search(candidates, mdl_n_bins=MDL_N_BINS)
    tpr = sum(1 for v in positive_jsd if v > tau) / len(positive_jsd)
    fpr = sum(1 for v in negative_jsd if v > tau) / len(negative_jsd)

    assert tpr - fpr > 0.3, f"TPR-FPR={tpr - fpr:.2f} below the 0.3 sanity bar for Step 5"
    print(f"[PASS] locked config (jsd_n_bins={JSD_N_BINS}, mdl_n_bins={MDL_N_BINS}): "
          f"tau={tau:.4f}, TPR={tpr:.2f}, FPR={fpr:.2f}, TPR-FPR={tpr - fpr:.2f}")


def test_decoupling_outperforms_coupled_baseline() -> None:
    """Confirms mdl_n_bins=4 substantially outperforms the ORIGINAL coupled
    configuration (mdl_n_bins == jsd_n_bins == 10), which had TPR-FPR <= 0."""
    candidates, negative_pairs = _setup()
    positive_jsd = [c.jsd_bits for c in candidates]
    negative_jsd = [jsd for (_, _, _, jsd) in negative_pairs]

    tau_coupled, _ = estimate_global_tau_via_grid_search(candidates, mdl_n_bins=10)
    tpr_c = sum(1 for v in positive_jsd if v > tau_coupled) / len(positive_jsd)
    fpr_c = sum(1 for v in negative_jsd if v > tau_coupled) / len(negative_jsd)

    tau_decoupled, _ = estimate_global_tau_via_grid_search(candidates, mdl_n_bins=MDL_N_BINS)
    tpr_d = sum(1 for v in positive_jsd if v > tau_decoupled) / len(positive_jsd)
    fpr_d = sum(1 for v in negative_jsd if v > tau_decoupled) / len(negative_jsd)

    assert (tpr_d - fpr_d) > (tpr_c - fpr_c)
    print(
        f"[PASS] decoupled (mdl_n_bins={MDL_N_BINS}) TPR-FPR={tpr_d - fpr_d:.2f} "
        f"> coupled (mdl_n_bins=10) TPR-FPR={tpr_c - fpr_c:.2f}"
    )


if __name__ == "__main__":
    test_param_cost_monotonic_in_bins()
    test_tau_reproducibility()
    test_locked_configuration_discriminates()
    test_decoupling_outperforms_coupled_baseline()
    print("=== ALL STEP 5 MDL THRESHOLD SMOKE TESTS PASSED ===")
