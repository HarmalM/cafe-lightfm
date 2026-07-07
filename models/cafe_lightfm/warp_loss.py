"""
warp_loss.py

Phase 2, Step 3: WARP (Weighted Approximate-Rank Pairwise) loss [1],
matching LightFM's training objective (Kula, 2015) [2] and the proposal's
Section 5.2.

SINGLE generic implementation, parameterized by:
    - a `score_fn(user_idx, item_idx) -> scores` callable
    - a per-sample exclusion set used during negative sampling

This lets the SAME warp_loss() serve both training regimes:
    - Step 1 (stage-blind baseline): use `baseline_scorer()`.
    - Step 2 (CAFE-LightFM / SCA): use `cafe_scorer(..., stage_idx)`.

PHASE 3, STEP 7 ADDITION (2026-07-06, confirmed by PI):
    `cafe_scorer` gains an optional `freeze_uniform: bool = False`
    parameter, passed straight through to `model(...)` inside the
    closure. This is the sole change needed to support the
    "CAFE-LightFM-noAttention" ablation at the loss-computation layer --
    `baseline_scorer` and `warp_loss()` itself are attention/stage-
    agnostic and are intentionally left untouched. Default False
    preserves Step 1-6 call sites exactly.

Negative sampling: rejection sampling up to `max_sampled` trials per
positive, matching the practical default used by the `lightfm` package.

Approximate rank weighting: rank_estimate = max(1, (n_items - 1) //
n_trials_until_violation); weight = log(rank_estimate + 1) [1, 2].

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
from typing import Callable, Dict, Optional, Set, Tuple

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
    model: CAFELightFM,
    bundle: InteractionMatrixBundle,
    stage_idx: int,
    freeze_uniform: bool = False,
) -> ScoreFn:
    """Scorer for Step 2's CAFE-LightFM, fixed at a given stage_idx.

    Parameters
    ----------
    freeze_uniform : bool, default False
        Phase 3, Step 7 (noAttention ablation). Passed straight to
        `model(...)`. Default preserves Step 1-6 behavior exactly.
    """

    def score(user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        category_idx, program_idx = gather_item_features(bundle, item_idx)
        stage_tensor = torch.full_like(item_idx, stage_idx)
        return model(
            user_idx, item_idx, category_idx, program_idx, stage_tensor,
            freeze_uniform=freeze_uniform,
        )

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
    pairs, via rejection-sampled negatives. Unchanged from Phase 2, Step
    3 -- attention/stage-agnostic by design, so no Step 7 modification
    is needed here.

    Parameters
    ----------
    score_fn            : callable, (user_idx, item_idx) -> scores. Already
                           closes over any fixed arguments (item metadata
                           lookups, stage_idx / freeze_uniform for
                           CAFE-LightFM).
    user_idx            : LongTensor, shape (batch,)
    positive_item_idx   : LongTensor, shape (batch,)
    user_positive_sets  : {user_idx: {item_idx, ...}} -- exclusion scope
                           for negative sampling.
    n_items             : total number of items.
    max_sampled         : maximum negative-sampling trials per positive
                           (default 10).
    margin              : hinge margin (default 1.0).
    rng                 : optional random.Random instance for
                           reproducible sampling (project convention:
                           seed=42).

    Returns
    -------
    torch.Tensor, scalar -- mean loss over the batch.
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
            # See Phase 2, Step 3 fix (2026-06-26): pos_score * 0.0 keeps
            # the tensor in the autograd graph so loss.backward() never
            # raises, even when every sample in a batch finds no
            # violator.
            losses.append(pos_score * 0.0)

    return torch.stack(losses).mean()


if __name__ == "__main__":
    # Self-contained smoke test for the Step 7 addition only (cafe_scorer
    # freeze_uniform passthrough). Uses a duck-typed stub model rather
    # than the real CAFELightFM, per project test-pattern convention
    # (minimal duck-typed stubs, no conftest dependency).
    class _StubBundle:
        def __init__(self):
            self.item_feature_idx_by_item = {0: (0, 0), 1: (0, 1)}

    class _StubModel:
        """Records the freeze_uniform value it was called with."""

        def __init__(self):
            self.last_freeze_uniform = None

        def __call__(self, user_idx, item_idx, category_idx, program_idx, stage_idx, freeze_uniform=False):
            self.last_freeze_uniform = freeze_uniform
            return torch.zeros(item_idx.shape[0])

    def _setup():
        return _StubModel(), _StubBundle()

    model, bundle = _setup()

    # 1. Default: freeze_uniform not passed -> should default to False.
    score_fn_default = cafe_scorer(model, bundle, stage_idx=1)
    score_fn_default(torch.tensor([0]), torch.tensor([0]))
    assert model.last_freeze_uniform is False, "Default cafe_scorer must pass freeze_uniform=False."

    # 2. Explicit freeze_uniform=True must reach the model call.
    score_fn_frozen = cafe_scorer(model, bundle, stage_idx=1, freeze_uniform=True)
    score_fn_frozen(torch.tensor([0]), torch.tensor([1]))
    assert model.last_freeze_uniform is True, "cafe_scorer must pass through freeze_uniform=True."

    print("warp_loss.py smoke test: all checks passed.")
