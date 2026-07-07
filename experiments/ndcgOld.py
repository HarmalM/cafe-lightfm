"""
ndcg.py

Phase 3, Step 1: stage-stratified NDCG@K evaluation for CAFE-LightFM
and the LightFM baseline.

IMPORTANT SCOPE NOTE (confirmed 2026-06-26):
    This module implements and validates the NDCG@K computation itself.
    All evaluation in this step is IN-TRAINING (relevance derived from
    the same positive_pairs_by_stage used during training) and is
    conducted SOLELY to verify that the metric implementation returns
    correct, finite values in [0,1]. No scientific performance claims
    should be drawn from these scores until a proper held-out test
    split (real Prolific data) is available.

NDCG@K formula (Jarvelin & Kekalainen, 2002) [1]:
    DCG@K  = sum_{k=1}^{K} (2^{rel_k} - 1) / log2(k + 1)
    NDCG@K = DCG@K / IDCG@K
    where IDCG@K is the DCG of a perfect ranking, and rel_k in {0, 1}
    for binary implicit feedback.

Modularity: `compute_ndcg_at_k_for_user()` and
`generate_user_ranking()` are exposed as public functions so that
SW-NDCG (Phase 3, Step 2) can import and reuse them without
re-implementing the core ranking and discount logic.

SW-NDCG weights and Bonferroni-corrected alpha are NOT applied here
(deferred to Step 2). The exact number of statistical comparisons must
be fixed after baseline selection is finalised before computing the
corrected threshold.

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418
[2] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop. CEUR-WS, Vol. 1448, 14-21.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import torch

from data.interaction_matrix import InteractionMatrixBundle
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

K_VALUES: Tuple[int, ...] = (5, 10, 20)
STAGE_ORDER: Tuple[str, ...] = ("S1", "S2", "S3")
STAGE_TO_IDX: Dict[str, int] = {"S1": 0, "S2": 1, "S3": 2}


# --------------------------------------------------------------------------- #
# Core NDCG computation (exposed for SW-NDCG reuse in Step 2)
# --------------------------------------------------------------------------- #

def compute_dcg_at_k(ranked_relevance: List[int], k: int) -> float:
    """
    Computes DCG@K for a single ranked list of binary relevance labels.
    DCG@K = sum_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)  [1]
    """
    dcg = 0.0
    for rank, rel in enumerate(ranked_relevance[:k], start=1):
        dcg += (2 ** rel - 1) / math.log2(rank + 1)
    return dcg


def compute_ndcg_at_k_for_user(
    ranked_relevance: List[int], k: int
) -> float:
    """
    Computes NDCG@K for a single user given a ranked binary relevance list.
    Returns 0.0 if IDCG == 0 (no relevant items in the list) to avoid
    division by zero.
    Exposed publicly for reuse by SW-NDCG (Phase 3, Step 2).
    """
    ideal = sorted(ranked_relevance, reverse=True)
    idcg = compute_dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return compute_dcg_at_k(ranked_relevance, k) / idcg


# --------------------------------------------------------------------------- #
# Ranking generation
# --------------------------------------------------------------------------- #

def generate_user_ranking(
    user_idx: int,
    stage_key: str,
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    is_cafe: bool,
) -> List[Tuple[float, int]]:
    """
    Generates a ranked list of (score, item_idx) for a given user and
    stage, using the model's scoring function.

    Exposed publicly for reuse by SW-NDCG (Phase 3, Step 2).

    Parameters
    ----------
    user_idx   : integer index of the user (from bundle.user_id_to_idx)
    stage_key  : one of 'S1', 'S2', 'S3'
    model      : LightFMPyTorch or CAFELightFM (duck-typed via is_cafe)
    bundle     : InteractionMatrixBundle
    is_cafe    : if True, passes stage_idx to the model; if False, uses
                 the stage-blind baseline signature.

    Returns
    -------
    List of (score, item_idx) sorted descending by score. Ties broken by
    item_idx ascending (deterministic under any seed).
    """
    n_items = bundle.n_items
    user_tensor = torch.full((n_items,), user_idx, dtype=torch.long)
    item_tensor = torch.arange(n_items, dtype=torch.long)

    cat_list, prog_list = [], []
    for i in range(n_items):
        cat_idx, prog_idx = bundle.item_feature_idx_by_item[i]
        cat_list.append(cat_idx)
        prog_list.append(prog_idx)
    category_tensor = torch.tensor(cat_list, dtype=torch.long)
    program_tensor = torch.tensor(prog_list, dtype=torch.long)

    with torch.no_grad():
        if is_cafe:
            stage_idx = STAGE_TO_IDX[stage_key]
            stage_tensor = torch.full((n_items,), stage_idx, dtype=torch.long)
            scores = model(
                user_tensor, item_tensor, category_tensor,
                program_tensor, stage_tensor,
            )
        else:
            scores = model(
                user_tensor, item_tensor, category_tensor, program_tensor
            )

    scored_items = sorted(
        zip(scores.tolist(), range(n_items)),
        key=lambda x: (-x[0], x[1]),  # descending score, ascending item_idx for ties
    )
    return scored_items


def ranked_relevance_for_user(
    user_idx: int,
    stage_key: str,
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    positive_set: Set[Tuple[int, int]],
    is_cafe: bool,
) -> List[int]:
    """
    Returns a binary relevance list aligned with the model's ranking for
    a given user/stage. rel=1 if (user_idx, item_idx) in positive_set.
    """
    ranking = generate_user_ranking(user_idx, stage_key, model, bundle, is_cafe)
    positive_items = {item for (u, item) in positive_set if u == user_idx}
    return [1 if item_idx in positive_items else 0 for (_, item_idx) in ranking]


# --------------------------------------------------------------------------- #
# Stage-stratified NDCG@K
# --------------------------------------------------------------------------- #

def stage_wise_ndcg(
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    is_cafe: bool,
    k_values: Tuple[int, ...] = K_VALUES,
) -> Dict[str, Dict[int, float]]:
    """
    Computes mean NDCG@K for each stage and each K, averaged over all
    users who have at least one positive item in that stage.

    Parameters
    ----------
    model     : trained LightFMPyTorch or CAFELightFM
    bundle    : InteractionMatrixBundle
    is_cafe   : True for CAFE-LightFM, False for the baseline
    k_values  : K values to evaluate (default (5, 10, 20))

    Returns
    -------
    Dict[stage_key -> Dict[K -> mean_NDCG@K]]
    """
    model.eval()
    results: Dict[str, Dict[int, float]] = {}

    for stage_key in STAGE_ORDER:
        positive_set = bundle.positive_pairs_by_stage.get(stage_key, set())
        if not positive_set:
            results[stage_key] = {k: 0.0 for k in k_values}
            continue

        users_in_stage = sorted({u for (u, _) in positive_set})
        ndcg_by_k: Dict[int, List[float]] = {k: [] for k in k_values}

        for user_idx in users_in_stage:
            rel_list = ranked_relevance_for_user(
                user_idx, stage_key, model, bundle, positive_set, is_cafe
            )
            for k in k_values:
                ndcg_by_k[k].append(compute_ndcg_at_k_for_user(rel_list, k))

        results[stage_key] = {
            k: sum(scores) / len(scores)
            for k, scores in ndcg_by_k.items()
        }

    return results


def print_ndcg_table(
    results: Dict[str, Dict[int, float]],
    model_name: str,
    k_values: Tuple[int, ...] = K_VALUES,
) -> None:
    """Prints a formatted NDCG results table."""
    header = f"{'Stage':>6}" + "".join(f"  NDCG@{k:>2}" for k in k_values)
    print(f"\n=== {model_name} | Stage-wise NDCG ===")
    print(header)
    print("-" * len(header))
    for stage_key in STAGE_ORDER:
        row = f"{stage_key:>6}"
        for k in k_values:
            row += f"  {results[stage_key][k]:>8.4f}"
        print(row)
    print()
