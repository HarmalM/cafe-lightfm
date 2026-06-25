"""
warp_loss.py

Phase 2, Step 3: WARP (Weighted Approximate-Rank Pairwise) loss [1],
matching LightFM's training objective (Kula, 2015) [2] and the proposal's
Section 5.2.

SINGLE generic implementation, parameterized by:
    - a `score_fn(user_idx, item_idx) -> scores` callable
    - a per-sample exclusion set used during negative sampling

This lets the SAME warp_loss() serve both training regimes (confirmed
2026-06-25):
    - Step 1 (stage-blind baseline): use `baseline_scorer()` and
      `build_user_positive_sets(bundle.positive_pairs)` -- negatives are
      items the user has not interacted with GLOBALLY.
    - Step 2 (CAFE-LightFM / SCA): use `cafe_scorer(..., stage_idx)` and
      `build_user_positive_sets(bundle.positive_pairs_by_stage[stage])`
      -- negatives are items the user has not interacted with WITHIN
      THE CURRENT STAGE. The SAME item may therefore be excluded
      (positive) for one stage's call and a valid negative candidate for
      another stage's call, for the SAME user -- this is the entire
      point of stage-conditioned training.

Negative sampling: rejection sampling up to `max_sampled` trials per
positive, matching the practical default used by the `lightfm` package
(not the theoretical n_items upper bound from the original WARP paper
[1]) -- chosen for fidelity to "faithfully replicating LightFM" (Phase 0)
and for tractability on larger future item catalogs.

Approximate rank weighting: rank_estimate = max(1, (n_items - 1) //
n_trials_until_violation); weight = log(rank_estimate + 1) [1, 2].

Deliberately OUT OF SCOPE for this module (deferred to Step 4):
    - the training loop / optimizer / epoch iteration

References
----------
[1] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to
    Large Vocabulary Image Annotation. Proceedings of the 22nd
    International Joint Conference on Artificial Intelligence (IJCAI),
    2764-2770.
[2] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop on New Trends in CBRS.
    CEUR-WS, Vol. 1448, 14-21.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import torch

from data.interaction_matrix import InteractionMatrixBundle
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_user_positive_sets(positive_pairs: Set[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """Inverts a (user_idx, item_idx) pair set into {user_idx: {item_idx, ...}},
    for O(1) exclusion-set lookup during negative sampling."""
    result: Dict[int, Set[int]] = defaultdict(set)
    for u, i in positive_pairs:
        result[u].add(i)
    return dict(result)


def gather_item_features(
    bundle: InteractionMatrixBundle, item_idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Looks up (category_idx, program_idx) for a batch of item indices."""
    cat_list, prog_list = [], []
    for idx in item_idx.tolist():
        cat_idx, prog_idx = bundle.item_feature_idx_by_item[idx]
        cat_list.append(cat_idx)
        prog_list.append(prog_idx)
    return (
        torch.tensor(cat_list, dtype=torch.long),
        torch.tensor(prog_list, dtype=torch.long),
    )


def baseline_scorer(model: LightFMPyTorch, bundle: InteractionMatrixBundle) -> ScoreFn:
    """Scorer for the Step 1 stage-blind baseline (no stage argument)."""

    def score(user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        category_idx, program_idx = gather_item_features(bundle, item_idx)
        return model(user_idx, item_idx, category_idx, program_idx)

    return score


def cafe_scorer(
    model: CAFELightFM, bundle: InteractionMatrixBundle, stage_idx: int
) -> ScoreFn:
    """Scorer for Step 2's CAFE-LightFM, fixed at a given stage_idx."""

    def score(user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        category_idx, program_idx = gather_item_features(bundle, item_idx)
        stage_tensor = torch.full_like(item_idx, stage_idx)
        return model(user_idx, item_idx, category_idx, program_idx, stage_tensor)

    return score


def warp_loss(
    score_fn: ScoreFn,
    user_idx: torch.Tensor,
    positive_item_idx: torch.Tensor,
    user_positive_sets: Dict[int, Set[int]],
    n_items: int,
    max_sampled: int = 10,
    margin: float = 1.0,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """
    Computes the mean WARP loss over a batch of (user, positive_item)
    pairs, via rejection-sampled negatives.

    Parameters
    ----------
    score_fn            : callable, (user_idx, item_idx) -> scores. Already
                           closes over any fixed arguments (item metadata
                           lookups, stage_idx for CAFE-LightFM).
    user_idx            : LongTensor, shape (batch,)
    positive_item_idx   : LongTensor, shape (batch,)
    user_positive_sets  : {user_idx: {item_idx, ...}} -- the exclusion
                           scope for negative sampling. Pass a GLOBAL set
                           (from bundle.positive_pairs) for the baseline,
                           or a STAGE-SPECIFIC set (from
                           bundle.positive_pairs_by_stage[stage]) for
                           CAFE-LightFM.
    n_items             : total number of items (for rank approximation
                           and candidate sampling range).
    max_sampled         : maximum negative-sampling trials per positive
                           before giving up (default 10).
    margin              : hinge margin (default 1.0).
    rng                 : optional random.Random instance for
                           reproducible sampling (project convention:
                           seed=42). Defaults to the global `random`
                           module if not provided.

    Returns
    -------
    torch.Tensor, scalar -- mean loss over the batch. Positives for which
    no violating negative was found within max_sampled trials contribute
    zero loss (standard WARP behavior: already well-ranked, no signal).
    """
    if rng is None:
        rng = random.Random()

    batch_size = user_idx.shape[0]
    pos_scores = score_fn(user_idx, positive_item_idx)  # (batch,), vectorized

    losses = []
    for b in range(batch_size):
        u = int(user_idx[b].item())
        pos_score = pos_scores[b]
        excluded = user_positive_sets.get(u, set())

        n_trials = 0
        violator_score = None
        while n_trials < max_sampled:
            n_trials += 1
            candidate = rng.randrange(n_items)
            if candidate in excluded:
                continue
            candidate_tensor = torch.tensor([candidate], dtype=torch.long)
            user_tensor = torch.tensor([u], dtype=torch.long)
            candidate_score = score_fn(user_tensor, candidate_tensor)[0]
            if candidate_score > pos_score - margin:
                violator_score = candidate_score
                break

        if violator_score is not None:
            rank_estimate = max(1, (n_items - 1) // n_trials)
            weight = math.log(rank_estimate + 1)
            hinge = torch.clamp(margin - pos_score + violator_score, min=0.0)
            losses.append(weight * hinge)
        else:
            losses.append(torch.zeros((), dtype=pos_scores.dtype))

    return torch.stack(losses).mean()
