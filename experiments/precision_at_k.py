"""
precision_at_k.py

Phase 3, Step 3: stage-stratified Precision@K evaluation for CAFE-LightFM
and the LightFM baseline.

IMPORTANT SCOPE NOTE (confirmed 2026-07-04):
    This module implements and validates the Precision@K computation
    itself, evaluated on the unsaturated synthetic validation dataset
    (`data/synthetic_generator_v3.py`, Phase 3 Step 1.5), per the
    binding scope boundary established in that step. As with Step 1/2,
    this is VALIDATION-ONLY: no scientific performance claim about
    CAFE-LightFM's superiority is drawn from these figures. That claim
    is reserved for the pilot (N=50) and full (N=300) Prolific studies
    (proposal Section 5.3).

Precision@K definition (fixed denominator, confirmed by PI 2026-07-04):
    Precision@K = |{relevant items in top-K}| / K

    The denominator is always K, regardless of how many relevant items
    actually exist for a given (user, stage) pair. If the number of true
    positives for a user/stage is < K, the maximum attainable
    Precision@K is correspondingly < 1.0 -- this is expected behavior,
    not an implementation defect, and mirrors the evaluation protocol
    used in the SASRec (Kang & McAuley, 2018) [2] and BERT4Rec
    (Sun et al., 2019) [3] baselines this model will be compared
    against in Phase 3 Step 5.

Zero re-implementation: this module performs NO re-implementation of
ranking or scoring logic. It imports `ranked_relevance_for_user()`
directly from `experiments/ndcg.py` (Phase 3, Step 1) as its sole
relevance source; `ranked_relevance_for_user()` itself calls
`generate_user_ranking()` internally, so ranking generation is reused
transitively without being imported directly here. This mirrors the
reuse pattern established in `experiments/sw_ndcg.py` (Phase 3, Step 2).

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418
    (Precision@K as a standard fixed-denominator top-K IR accuracy
    metric is used alongside NDCG@K throughout this and related work.)
[2] Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential
    Recommendation. Proceedings of the 2018 IEEE International
    Conference on Data Mining (ICDM), 197-206. arXiv:1808.09781.
    https://arxiv.org/abs/1808.09781
[3] Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P.
    (2019). BERT4Rec: Sequential Recommendation with Bidirectional
    Encoder Representations from Transformer. Proceedings of the 28th
    ACM CIKM, 1441-1450. https://doi.org/10.1145/3357384.3357895
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from data.interaction_matrix import InteractionMatrixBundle
from experiments.ndcg import (
    K_VALUES,
    STAGE_ORDER,
    ranked_relevance_for_user,
)


# --------------------------------------------------------------------------- #
# Core Precision@K computation
# --------------------------------------------------------------------------- #

def compute_precision_at_k(ranked_relevance: List[int], k: int) -> float:
    """
    Computes Precision@K for a single ranked list of binary relevance
    labels, using a fixed denominator K (Jarvelin & Kekalainen, 2002) [1].

        Precision@K = |{relevant items in top-K}| / K

    Parameters
    ----------
    ranked_relevance : binary relevance labels (1 = relevant, 0 = not),
                        already ordered by the model's ranking
                        (descending score), as produced by
                        `ranked_relevance_for_user()`.
    k                : cutoff rank.

    Returns
    -------
    float in [0, 1]. If len(ranked_relevance) < k, the numerator is
    computed over the available items but the denominator remains k
    (fixed-K convention, confirmed 2026-07-04) -- the result will be
    < 1.0 even for a "perfect" partial ranking. This is expected
    behavior, not a division-by-zero risk, since k > 0 always.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    top_k = ranked_relevance[:k]
    return sum(top_k) / k


# --------------------------------------------------------------------------- #
# Stage-stratified Precision@K
# --------------------------------------------------------------------------- #

def stage_wise_precision(
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    is_cafe: bool,
    k_values: Tuple[int, ...] = K_VALUES,
) -> Dict[str, Dict[int, float]]:
    """
    Computes mean Precision@K for each stage and each K, averaged over
    all users who have at least one positive item in that stage.

    Mirrors the structure of `experiments.ndcg.stage_wise_ndcg()`
    exactly, reusing the same ranking/relevance generation so that
    NDCG@K and Precision@K are computed over identical rankings for
    every (user, stage) pair -- required for the metrics to be
    comparable within the same evaluation protocol (proposal Section 5.4).

    Parameters
    ----------
    model     : trained LightFMPyTorch or CAFELightFM
    bundle    : InteractionMatrixBundle
    is_cafe   : True for CAFE-LightFM, False for the baseline
    k_values  : K values to evaluate (default (5, 10, 20))

    Returns
    -------
    Dict[stage_key -> Dict[K -> mean_Precision@K]]
    """
    model.eval()
    results: Dict[str, Dict[int, float]] = {}

    for stage_key in STAGE_ORDER:
        positive_set = bundle.positive_pairs_by_stage.get(stage_key, set())
        if not positive_set:
            results[stage_key] = {k: 0.0 for k in k_values}
            continue

        users_in_stage = sorted({u for (u, _) in positive_set})
        precision_by_k: Dict[int, List[float]] = {k: [] for k in k_values}

        for user_idx in users_in_stage:
            rel_list = ranked_relevance_for_user(
                user_idx, stage_key, model, bundle, positive_set, is_cafe
            )
            for k in k_values:
                precision_by_k[k].append(compute_precision_at_k(rel_list, k))

        results[stage_key] = {
            k: sum(scores) / len(scores)
            for k, scores in precision_by_k.items()
        }

    return results


def print_precision_table(
    results: Dict[str, Dict[int, float]],
    model_name: str,
    k_values: Tuple[int, ...] = K_VALUES,
) -> None:
    """Prints a formatted Precision@K results table."""
    header = f"{'Stage':>6}" + "".join(f"  Prec@{k:>2}" for k in k_values)
    print(f"\n=== {model_name} | Stage-wise Precision@K ===")
    print(header)
    print("-" * len(header))
    for stage_key in STAGE_ORDER:
        row = f"{stage_key:>6}"
        for k in k_values:
            row += f"  {results[stage_key][k]:>8.4f}"
        print(row)
    print()
