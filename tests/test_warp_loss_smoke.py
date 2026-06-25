"""
test_warp_loss_smoke.py

Smoke test for models/cafe_lightfm/warp_loss.py (Phase 2, Step 3).
Validates:
    1. The key design property (confirmed 2026-06-25): global vs.
       stage-specific exclusion sets genuinely differ on real data --
       an item positive for a user in one stage but not another IS
       excluded under the global scope and IS a valid negative under
       the stage-specific scope.
    2. WARP loss gradient flow through the Step 1 baseline scorer.
    3. WARP loss gradient flow through the Step 2 CAFE-LightFM scorer
       (including SCA's w_base / w_stage parameters).
    4. Reproducibility under a seeded rng.
    5. Rank-weight sanity check, using a hand-controlled deterministic
       RNG sequence (no real model needed): a violator found QUICKLY
       must receive a LARGER weight (implying a worse-ranked positive)
       than a violator found only after many trials.
    6. The "no violator found" path contributes exactly zero loss.

REQUIRES PyTorch for tests 2-3. Tests 1, 4, 5, 6 use only plain Python /
a hand-rolled scorer and run without a real model. Run in Colab:
    python -m tests.test_warp_loss_smoke
"""

from __future__ import annotations

import math
import random

import torch

from data.interaction_matrix import build_interaction_matrix
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2, MASTER_SEED
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM
from models.cafe_lightfm.warp_loss import (
    baseline_scorer,
    build_user_positive_sets,
    cafe_scorer,
    warp_loss,
)

N_CATEGORIES, N_PROGRAMS, N_STAGES = 1, 3, 3
EMBEDDING_DIM = 64
SEED = 42


class _SequenceRNG:
    """Deterministic stand-in for random.Random: cycles through a fixed
    sequence of `randrange` return values, for controlled rank-weight tests."""

    def __init__(self, sequence):
        self._sequence = sequence
        self._i = 0

    def randrange(self, n: int) -> int:
        value = self._sequence[self._i % len(self._sequence)]
        self._i += 1
        return value


def test_exclusion_sets_differ_by_scope() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)

    global_sets = build_user_positive_sets(bundle.positive_pairs)
    stage_sets = {
        stage: build_user_positive_sets(pairs) for stage, pairs in bundle.positive_pairs_by_stage.items()
    }

    found_example = False
    for u, global_items in global_sets.items():
        for stage_items_by_user in stage_sets.values():
            stage_items = stage_items_by_user.get(u, set())
            if global_items - stage_items:
                found_example = True
                break
        if found_example:
            break

    assert found_example, "no user/item found where global and stage-specific exclusion sets differ"
    print("[PASS] global and stage-specific exclusion sets genuinely differ on real data "
          "(an item positive in one stage need not be excluded as a negative in another)")


def test_gradient_flow_baseline_scorer() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)

    torch.manual_seed(SEED)
    model = LightFMPyTorch(bundle.n_users, bundle.n_items, N_CATEGORIES, N_PROGRAMS, EMBEDDING_DIM)
    score_fn = baseline_scorer(model, bundle)
    user_positive_sets = build_user_positive_sets(bundle.positive_pairs)

    pairs = list(bundle.positive_pairs)[:8]
    user_idx = torch.tensor([p[0] for p in pairs])
    item_idx = torch.tensor([p[1] for p in pairs])

    loss = warp_loss(
        score_fn, user_idx, item_idx, user_positive_sets, bundle.n_items,
        max_sampled=10, margin=1.0, rng=random.Random(SEED),
    )
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.item_embedding.weight.grad is not None
    print(f"[PASS] WARP loss (baseline scorer) is finite ({loss.item():.4f}) and gradients flow")


def test_gradient_flow_cafe_scorer() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)

    torch.manual_seed(SEED)
    model = CAFELightFM(bundle.n_users, bundle.n_items, N_CATEGORIES, N_PROGRAMS, N_STAGES, EMBEDDING_DIM)
    stage = "S1"
    score_fn = cafe_scorer(model, bundle, stage_idx=0)
    user_positive_sets = build_user_positive_sets(bundle.positive_pairs_by_stage[stage])

    pairs = list(bundle.positive_pairs_by_stage[stage])[:8]
    user_idx = torch.tensor([p[0] for p in pairs])
    item_idx = torch.tensor([p[1] for p in pairs])

    loss = warp_loss(
        score_fn, user_idx, item_idx, user_positive_sets, bundle.n_items,
        max_sampled=10, margin=1.0, rng=random.Random(SEED),
    )
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.sca.w_base.grad is not None
    assert model.sca.w_stage.weight.grad is not None
    print(f"[PASS] WARP loss (CAFE-LightFM scorer, stage={stage}) is finite ({loss.item():.4f}); "
          f"gradients reach SCA's w_base and w_stage")


def test_reproducibility() -> None:
    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)

    torch.manual_seed(SEED)
    model_a = LightFMPyTorch(bundle.n_users, bundle.n_items, N_CATEGORIES, N_PROGRAMS, EMBEDDING_DIM)
    torch.manual_seed(SEED)
    model_b = LightFMPyTorch(bundle.n_users, bundle.n_items, N_CATEGORIES, N_PROGRAMS, EMBEDDING_DIM)

    user_positive_sets = build_user_positive_sets(bundle.positive_pairs)
    pairs = list(bundle.positive_pairs)[:8]
    user_idx = torch.tensor([p[0] for p in pairs])
    item_idx = torch.tensor([p[1] for p in pairs])

    loss_a = warp_loss(
        baseline_scorer(model_a, bundle), user_idx, item_idx, user_positive_sets,
        bundle.n_items, max_sampled=10, margin=1.0, rng=random.Random(SEED),
    )
    loss_b = warp_loss(
        baseline_scorer(model_b, bundle), user_idx, item_idx, user_positive_sets,
        bundle.n_items, max_sampled=10, margin=1.0, rng=random.Random(SEED),
    )
    assert torch.allclose(loss_a, loss_b)
    print(f"[PASS] identical seed ({SEED}) reproduces identical WARP loss: {loss_a.item():.4f}")


def test_rank_weight_sanity_fast_vs_slow_violator() -> None:
    """No real model needed: a hand-controlled scorer and a hand-controlled
    RNG sequence isolate the rank-weighting logic directly."""
    item_scores = {0: -10.0, 1: -10.0, 2: -10.0, 3: -10.0, 4: 10.0}
    POS_ITEM = 99

    def manual_score_fn(user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [0.0 if idx == POS_ITEM else item_scores[idx] for idx in item_idx.tolist()]
        )

    user_idx = torch.tensor([0])
    positive_item_idx = torch.tensor([POS_ITEM])

    # Fast: violator (index 4) found on the FIRST trial.
    loss_fast = warp_loss(
        manual_score_fn, user_idx, positive_item_idx, {}, n_items=5,
        max_sampled=10, margin=1.0, rng=_SequenceRNG([4, 0, 1, 2, 3]),
    )
    # Slow: violator (index 4) found only on the LAST (5th) trial.
    loss_slow = warp_loss(
        manual_score_fn, user_idx, positive_item_idx, {}, n_items=5,
        max_sampled=10, margin=1.0, rng=_SequenceRNG([0, 1, 2, 3, 4]),
    )

    assert loss_fast > loss_slow, (
        f"fast-found violator should yield a LARGER weight (worse-ranked positive) "
        f"than slow-found: loss_fast={loss_fast.item():.4f}, loss_slow={loss_slow.item():.4f}"
    )
    print(f"[PASS] rank-weight sanity: loss_fast={loss_fast.item():.4f} > loss_slow={loss_slow.item():.4f} "
          f"(quickly-found violator implies worse rank, correctly weighted higher)")


def test_no_violator_found_contributes_zero_loss() -> None:
    def all_low_score_fn(user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [0.0 if idx == 99 else -100.0 for idx in item_idx.tolist()]
        )

    user_idx = torch.tensor([0])
    positive_item_idx = torch.tensor([99])
    loss = warp_loss(
        all_low_score_fn, user_idx, positive_item_idx, {}, n_items=5,
        max_sampled=10, margin=1.0, rng=_SequenceRNG([0, 1, 2, 3, 4]),
    )
    assert loss.item() == 0.0
    print("[PASS] no violator found within max_sampled trials -> exactly zero loss")


if __name__ == "__main__":
    test_exclusion_sets_differ_by_scope()
    test_rank_weight_sanity_fast_vs_slow_violator()
    test_no_violator_found_contributes_zero_loss()
    test_gradient_flow_baseline_scorer()
    test_gradient_flow_cafe_scorer()
    test_reproducibility()
    print("=== ALL WARP LOSS (STEP 3) SMOKE TESTS PASSED ===")
