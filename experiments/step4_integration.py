"""
experiments/step4_integration.py

Phase 3, Step 4 (integration layer) -- connects the per-user ranking
outputs of experiments/ndcg.py and experiments/precision_at_k.py to the
paired-testing primitives in experiments/statistical_testing.py.

ZERO RE-IMPLEMENTATION: this module does not recompute DCG/IDCG, ranking
generation, or Precision@K. It imports and reuses, unchanged:
    - ranked_relevance_for_user()      (experiments/ndcg.py)
    - compute_ndcg_at_k_for_user()     (experiments/ndcg.py)
    - compute_precision_at_k()         (experiments/precision_at_k.py)
    - paired_t_test(), summarize_paired_results()
                                        (experiments/statistical_testing.py)
This module's only job is: (a) extract a per-user metric value for each
(model, stage, K, metric) combination, and (b) align the CAFE vs.
baseline per-user dictionaries into paired numpy arrays before handing
them to paired_t_test.

SCOPE (binding, per Steps 1-3 scope notes): when run against
data/synthetic_generator_v3.py, this is a pipeline-validation scaffold
only. No scientific claim of CAFE-LightFM superiority is made from these
results. That claim is reserved for the pilot (N=50) / full (N=300)
Prolific studies (proposal Section 5.3).

Excluded per Decision A2 (confirmed 2026-07-05): SW-NDCG is NOT included
in this per-user paired-testing driver. SW-NDCG (experiments/sw_ndcg.py)
has no natural per-user value unless a user has positives in all three
stages simultaneously, which synthetic_generator_v3.py does not
guarantee. Only per-stage, per-K NDCG@K and Precision@K are compared
here; SW-NDCG paired testing is deferred to the Prolific dataset.

Comparison family size for THIS scaffold (dynamic M, Decision C):
    |STAGE_ORDER| x |K_VALUES| x 2 metrics (NDCG, Precision)
    = 3 x 3 x 2 = 18 comparisons, evaluated against a single baseline
    (LightFM-PyTorch) on the v3 synthetic dataset.
This is NOT the locked full-study M=63 (proposal Section 5.4, which
additionally spans the full 7-baseline suite). Pass
`n_comparisons_override=63` to `run_ndcg_precision_paired_comparison`
once the full baseline suite and Prolific dataset are in place, to
reproduce the locked full-study correction without changing this
module's logic.

ASSUMPTION (explicit, labeled): bundle.positive_pairs_by_stage[stage_key]
is assumed identical for both the CAFE-LightFM and baseline evaluation
calls, since both models are evaluated against the SAME bundle/ground
truth (only the ranking each model produces differs). Rationale: this is
how experiments/ndcg.py's own stage_wise_ndcg() and
experiments/precision_at_k.py's stage_wise_precision() are structured
(single `bundle` argument shared across both `is_cafe` branches).
Impact if violated: align_paired_arrays() raises ValueError rather than
silently comparing mismatched user populations.

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418
[2] Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for
    Scientific Computing in Python. Nature Methods, 17, 261-272.
    https://doi.org/10.1038/s41592-019-0686-2
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.interaction_matrix import InteractionMatrixBundle
from experiments.ndcg import (
    K_VALUES,
    STAGE_ORDER,
    compute_ndcg_at_k_for_user,
    ranked_relevance_for_user,
)
from experiments.precision_at_k import compute_precision_at_k
from experiments.statistical_testing import (
    PairedTestResult,
    paired_t_test,
    summarize_paired_results,
)

# (metric_label, metric_fn) pairs. metric_fn signature must match
# (ranked_relevance: List[int], k: int) -> float, per compute_ndcg_at_k_for_user
# and compute_precision_at_k (both reused unmodified, zero re-implementation).
METRIC_SPECS: Tuple[Tuple[str, Callable[[List[int], int], float]], ...] = (
    ("NDCG", compute_ndcg_at_k_for_user),
    ("Precision", compute_precision_at_k),
)


def per_user_metric(
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    is_cafe: bool,
    stage_key: str,
    k: int,
    metric_fn: Callable[[List[int], int], float],
) -> Dict[int, float]:
    """
    Computes a per-user metric value (NDCG@K or Precision@K) for every
    user with >=1 positive interaction in `stage_key`, using that
    model's own ranking of the full item catalog for that user/stage.

    Parameters
    ----------
    model      : trained LightFMPyTorch (is_cafe=False) or CAFELightFM
                 (is_cafe=True).
    bundle     : InteractionMatrixBundle providing
                 `positive_pairs_by_stage`.
    is_cafe    : forwarded to `ranked_relevance_for_user` /
                 `generate_user_ranking` to select the model call
                 signature (stage-conditioned vs. stage-blind).
    stage_key  : one of 'S1', 'S2', 'S3'.
    k          : cutoff rank for the metric.
    metric_fn  : (ranked_relevance, k) -> float. Pass
                 `compute_ndcg_at_k_for_user` or `compute_precision_at_k`.

    Returns
    -------
    Dict[user_idx -> metric value]. Empty dict if the stage has no
    positive pairs (mirrors the empty-stage handling in
    `stage_wise_ndcg` / `stage_wise_precision`).
    """
    positive_set = bundle.positive_pairs_by_stage.get(stage_key, set())
    users_in_stage = sorted({u for (u, _) in positive_set})
    result: Dict[int, float] = {}
    for user_idx in users_in_stage:
        rel_list = ranked_relevance_for_user(
            user_idx, stage_key, model, bundle, positive_set, is_cafe
        )
        result[user_idx] = metric_fn(rel_list, k)
    return result


def align_paired_arrays(
    scores_a: Dict[int, float], scores_b: Dict[int, float]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Aligns two per-user metric dictionaries (CAFE vs. baseline) into
    paired numpy arrays ordered by ascending user_idx.

    Raises
    ------
    ValueError
        If the two dictionaries' user_idx key sets are not identical
        (see module-level ASSUMPTION note), or if empty.
    """
    keys_a = set(scores_a.keys())
    keys_b = set(scores_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            "CAFE and baseline per-user score dictionaries cover "
            f"different user sets (symmetric difference: "
            f"{keys_a.symmetric_difference(keys_b)}). Both models must "
            "be evaluated against the same bundle/stage."
        )
    if not keys_a:
        raise ValueError("No users found in either score dictionary.")
    common_users = sorted(keys_a)
    a_arr = np.array([scores_a[u] for u in common_users], dtype=float)
    b_arr = np.array([scores_b[u] for u in common_users], dtype=float)
    return a_arr, b_arr, common_users


def run_ndcg_precision_paired_comparison(
    cafe_model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    k_values: Tuple[int, ...] = K_VALUES,
    stage_order: Tuple[str, ...] = STAGE_ORDER,
    n_comparisons_override: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Driver: runs a two-tailed paired t-test (CAFE vs. baseline; Decision
    B) for every (stage, K, metric) combination across NDCG@K and
    Precision@K, then applies Bonferroni correction (Decision C: dynamic
    M by default -- see module docstring for this scaffold's M=18).

    Excludes SW-NDCG per Decision A2 (see module docstring).

    Parameters
    ----------
    cafe_model, baseline_model : trained models, duck-typed per
        experiments.ndcg.generate_user_ranking's `is_cafe` branch.
    bundle : InteractionMatrixBundle shared by both models.
    k_values, stage_order : evaluation grid (defaults match
        experiments.ndcg.K_VALUES / STAGE_ORDER).
    n_comparisons_override : Optional[int]. None -> dynamic M = number
        of comparisons actually produced (typically 18 for this
        3-stage x 3-K x 2-metric scaffold). Pass 63 to reproduce the
        locked full-study Bonferroni correction (proposal Section 5.4).

    Returns
    -------
    Dict keyed by f"{metric}@{k}_{stage}" -> per-comparison summary
    (n_pairs, mean_a, mean_b, mean_diff, t_statistic, cohens_dz,
    degrees_of_freedom, p_value, alpha_corrected, significant).

    Raises
    ------
    ValueError
        If no valid (stage, K, metric) comparison could be formed (e.g.
        every stage is empty, or every stage has <2 users) -- surfaced
        rather than silently returning an empty result.
    """
    results: List[PairedTestResult] = []

    for stage_key in stage_order:
        for k in k_values:
            for metric_name, metric_fn in METRIC_SPECS:
                cafe_scores = per_user_metric(
                    cafe_model, bundle, True, stage_key, k, metric_fn
                )
                baseline_scores = per_user_metric(
                    baseline_model, bundle, False, stage_key, k, metric_fn
                )
                if not cafe_scores or not baseline_scores:
                    # Empty stage (no positives) -- skip rather than
                    # raise, mirroring stage_wise_ndcg's all-zero
                    # fallback for empty stages.
                    continue
                a_arr, b_arr, _ = align_paired_arrays(cafe_scores, baseline_scores)
                if a_arr.size < 2:
                    # paired_t_test / cohens_dz require >=2 observations;
                    # skip degenerate single-user stages rather than crash.
                    continue
                label = f"{metric_name}@{k}_{stage_key}"
                results.append(paired_t_test(a_arr, b_arr, metric_name=label))

    if not results:
        raise ValueError(
            "No valid paired comparisons were produced. Check that "
            "bundle.positive_pairs_by_stage contains at least one stage "
            "with >=2 users having positive interactions."
        )

    return summarize_paired_results(
        results, alpha=0.05, n_comparisons=n_comparisons_override
    )


def print_comparison_table(summary: Dict[str, Dict[str, float]]) -> None:
    """Prints a formatted summary of the paired-comparison results."""
    header = (
        f"{'Comparison':<16}{'n':>4}{'mean_A':>9}{'mean_B':>9}"
        f"{'diff':>9}{'t':>8}{'p':>10}{'d_z':>8}{'alpha*':>10}{'sig':>6}"
    )
    print("\n=== Step 4: Paired t-test Summary (CAFE vs. Baseline) ===")
    print(header)
    print("-" * len(header))
    for name, r in summary.items():
        print(
            f"{name:<16}{r['n_pairs']:>4}{r['mean_a']:>9.4f}"
            f"{r['mean_b']:>9.4f}{r['mean_diff']:>9.4f}"
            f"{r['t_statistic']:>8.3f}{r['p_value']:>10.6f}"
            f"{r['cohens_dz']:>8.3f}{r['alpha_corrected']:>10.6f}"
            f"{'Y' if r['significant'] else 'N':>6}"
        )
    print()
