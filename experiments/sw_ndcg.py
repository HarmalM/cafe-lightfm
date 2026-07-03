"""
sw_ndcg.py

Phase 3, Step 2: Stage-Weighted NDCG@K (SW-NDCG) for CAFE-LightFM
evaluation, per proposal Section 5.4 ("Stage-Weighted NDCG: a composite
metric weighting stage-specific performance by the decision salience of
each stage, with higher weight assigned to finalization-stage
performance").

FORMAL DEFINITION (Definition III.2)
-------------------------------------
For model M, cutoff K, and stage weights {w_s1, w_s2, w_s3} with
sum_j w_sj = 1:

    SW-NDCG@K(M) = sum_{j=1}^{3} w_sj * NDCG@K(s_j; M)

where NDCG@K(s_j; M) is the mean stage-stratified NDCG@K produced by
`experiments.ndcg.stage_wise_ndcg()` (Phase 3, Step 1) [1].

LOCKED WEIGHTS (confirmed, project configuration):
    w_S1 = 0.20 (Exploration)
    w_S2 = 0.30 (Comparison)
    w_S3 = 0.50 (Finalization)
Finalization receives the highest weight because it is the
decision-consequential stage (proposal Section 5.1): a user's terminal
choice is made at S3, so recommendation quality there has the greatest
practical impact on outcomes.

SCOPE NOTE (inherited from Step 1 and Step 1.5, still binding):
    Evaluation in this step uses IN-TRAINING relevance on either the
    Step 4 (saturated, v2) or Step 1.5 (unsaturated, v3) synthetic
    dataset. This validates the SW-NDCG *implementation* only. No
    scientific performance claims are made until real Prolific data is
    available (Section 5.3, proposal).

This module performs ZERO re-implementation of DCG/IDCG/ranking logic:
it consumes `stage_wise_ndcg()` from `experiments/ndcg.py` (Step 1) as
its sole per-stage NDCG source, per that module's documented reuse
contract.

References
----------
[1] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation
    of IR techniques. ACM Transactions on Information Systems, 20(4),
    422-446. https://doi.org/10.1145/582415.582418
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from data.interaction_matrix import InteractionMatrixBundle
from experiments.ndcg import K_VALUES, STAGE_ORDER, stage_wise_ndcg

# Locked stage weights (confirmed, proposal Section 5.4 + project config).
DEFAULT_STAGE_WEIGHTS: Dict[str, float] = {"S1": 0.20, "S2": 0.30, "S3": 0.50}

_WEIGHT_SUM_TOLERANCE = 1e-9


def _validate_weights(weights: Dict[str, float]) -> None:
    """
    Validates that `weights` covers exactly {S1, S2, S3} and sums to 1.0
    within floating-point tolerance. Raises ValueError otherwise -- fails
    fast rather than silently producing a mis-scaled composite metric.
    """
    if set(weights.keys()) != set(STAGE_ORDER):
        raise ValueError(
            f"weights must cover exactly {set(STAGE_ORDER)}, got {set(weights.keys())}"
        )
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"stage weights must sum to 1.0, got {total}")


def compute_sw_ndcg_from_stage_results(
    stage_results: Dict[str, Dict[int, float]],
    weights: Dict[str, float] = DEFAULT_STAGE_WEIGHTS,
    k_values: Tuple[int, ...] = K_VALUES,
) -> Dict[int, float]:
    """
    Combines an existing stage_wise_ndcg() output into SW-NDCG@K per
    Definition III.2. Pure function -- no model or bundle needed -- so it
    is directly unit-testable against hand-computed values.

    Parameters
    ----------
    stage_results : output of experiments.ndcg.stage_wise_ndcg()
                    (Dict[stage_key -> Dict[K -> mean NDCG@K]])
    weights       : stage weights, must sum to 1.0 (default: locked
                    project weights S1=0.20/S2=0.30/S3=0.50)
    k_values      : K values to compute SW-NDCG for (default (5,10,20))

    Returns
    -------
    Dict[K -> SW-NDCG@K]
    """
    _validate_weights(weights)
    missing_stages = set(STAGE_ORDER) - set(stage_results.keys())
    if missing_stages:
        raise ValueError(f"stage_results missing required stages: {missing_stages}")

    sw_ndcg_by_k: Dict[int, float] = {}
    for k in k_values:
        sw_ndcg_by_k[k] = sum(
            weights[stage] * stage_results[stage][k] for stage in STAGE_ORDER
        )
    return sw_ndcg_by_k


def sw_ndcg(
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    is_cafe: bool,
    weights: Dict[str, float] = DEFAULT_STAGE_WEIGHTS,
    k_values: Tuple[int, ...] = K_VALUES,
) -> Dict[int, float]:
    """
    End-to-end SW-NDCG@K: runs stage_wise_ndcg() (Step 1) then combines
    with locked stage weights (Definition III.2).

    Parameters
    ----------
    model    : trained LightFMPyTorch or CAFELightFM
    bundle   : InteractionMatrixBundle
    is_cafe  : True for CAFE-LightFM, False for the stage-blind baseline
    weights  : stage weights, must sum to 1.0 (default: locked weights)
    k_values : K values to evaluate (default (5, 10, 20))

    Returns
    -------
    Dict[K -> SW-NDCG@K]
    """
    stage_results = stage_wise_ndcg(model, bundle, is_cafe, k_values)
    return compute_sw_ndcg_from_stage_results(stage_results, weights, k_values)


def print_sw_ndcg_table(
    sw_results: Dict[int, float],
    model_name: str,
    weights: Dict[str, float] = DEFAULT_STAGE_WEIGHTS,
    k_values: Tuple[int, ...] = K_VALUES,
) -> None:
    """Prints a formatted SW-NDCG results table with the weights used."""
    weight_str = ", ".join(f"{s}={weights[s]:.2f}" for s in STAGE_ORDER)
    print(f"\n=== {model_name} | SW-NDCG (weights: {weight_str}) ===")
    header = "".join(f"  SW-NDCG@{k:<2}" for k in k_values)
    print(header)
    row = "".join(f"  {sw_results[k]:>10.4f}" for k in k_values)
    print(row)
    print()
