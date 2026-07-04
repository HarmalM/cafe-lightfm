"""
tests/test_step4_integration_smoke.py

Phase 3, Step 4 -- smoke tests for experiments/step4_integration.py.

SELF-CONTAINED STUBS (explicit assumption, labeled):
    This test file does NOT import the real InteractionMatrixBundle,
    CAFELightFM, or LightFMPyTorch classes -- those are not visible to
    this environment. Instead it defines minimal duck-typed stub classes
    (`_StubBundle`, `_StubCAFEModel`, `_StubBaselineModel`) that satisfy
    exactly the interface `experiments/ndcg.py::generate_user_ranking`
    actually calls:
        - bundle.n_items
        - bundle.item_feature_idx_by_item[i]        -> (cat_idx, prog_idx)
        - bundle.positive_pairs_by_stage[stage_key]  -> Set[(user, item)]
        - model.eval()
        - model(user_tensor, item_tensor, category_tensor, program_tensor
                [, stage_tensor if is_cafe])         -> 1D tensor of scores

    RATIONALE: these attribute/method names and call signatures were
    copied verbatim from the pasted experiments/ndcg.py source (Phase 3
    Step 1), so the stubs are faithful to the real duck-typed contract,
    not a guess. IMPACT: this validates the integration/alignment logic
    (experiments/step4_integration.py) end-to-end, but does NOT validate
    against the real trained CAFE-LightFM / LightFM-PyTorch models or the
    real synthetic_generator_v3.py dataset -- that verification must
    still happen in Colab against the actual repo objects.

Run with:
    pytest tests/test_step4_integration_smoke.py -v
"""

from typing import Dict, Set, Tuple

import numpy as np
import pytest
import torch

from experiments.step4_integration import (
    align_paired_arrays,
    per_user_metric,
    run_ndcg_precision_paired_comparison,
)
from experiments.ndcg import compute_ndcg_at_k_for_user
from experiments.precision_at_k import compute_precision_at_k

SEED = 42  # project master seed (config.py)


# ---------------------------------------------------------------------------
# Stub bundle / models (see module docstring)
# ---------------------------------------------------------------------------

class _StubBundle:
    """Minimal duck-typed stand-in for InteractionMatrixBundle."""

    def __init__(
        self,
        n_items: int,
        positive_pairs_by_stage: Dict[str, Set[Tuple[int, int]]],
    ):
        self.n_items = n_items
        self.positive_pairs_by_stage = positive_pairs_by_stage
        # Item features are unused by the stub models below; (0, 0) for all.
        self.item_feature_idx_by_item = {i: (0, 0) for i in range(n_items)}


class _StubCAFEModel:
    """
    Deterministic stub mimicking CAFELightFM's call signature. NOT a real
    model. Assigns a high score (0.9 + tiny noise) to each user's true
    positive items and a low random score to all other items, so that
    CAFE-vs-baseline paired tests have a known, strongly positive
    expected direction for validating the statistical pipeline.
    """

    def __init__(self, n_users: int, n_items: int, positive_set, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.n_items = n_items
        self.low_scores = rng.uniform(0.0, 0.2, size=(n_users, n_items))
        self.positive_by_user: Dict[int, Set[int]] = {}
        for (u, i) in positive_set:
            self.positive_by_user.setdefault(u, set()).add(i)

    def eval(self):
        pass

    def __call__(self, user_tensor, item_tensor, category_tensor, program_tensor, stage_tensor):
        u = int(user_tensor[0].item())
        pos_items = self.positive_by_user.get(u, set())
        scores = []
        for i in item_tensor.tolist():
            if i in pos_items:
                scores.append(0.9)
            else:
                scores.append(float(self.low_scores[u, i]))
        return torch.tensor(scores, dtype=torch.float32)


class _StubBaselineModel:
    """
    Deterministic stub mimicking LightFMPyTorch's call signature (no
    stage_tensor argument). Assigns uniform random scores irrespective of
    true positivity, simulating an untrained/uninformative baseline.
    """

    def __init__(self, n_users: int, n_items: int, seed: int = SEED + 1):
        rng = np.random.default_rng(seed)
        self.scores = rng.uniform(0.0, 1.0, size=(n_users, n_items))

    def eval(self):
        pass

    def __call__(self, user_tensor, item_tensor, category_tensor, program_tensor):
        u = int(user_tensor[0].item())
        scores = [float(self.scores[u, i]) for i in item_tensor.tolist()]
        return torch.tensor(scores, dtype=torch.float32)


def _build_stub_scenario(n_users=15, n_items=30, n_pos_per_user=2, seed=SEED):
    """Builds a fixed, reproducible stub bundle + CAFE/baseline model pair
    with 2 known positive items per user in stage S1 (S2/S3 left empty to
    also exercise the empty-stage skip path)."""
    rng = np.random.default_rng(seed)
    positive_set: Set[Tuple[int, int]] = set()
    for u in range(n_users):
        items = rng.choice(n_items, size=n_pos_per_user, replace=False)
        for i in items:
            positive_set.add((u, int(i)))

    bundle = _StubBundle(
        n_items=n_items,
        positive_pairs_by_stage={"S1": positive_set, "S2": set(), "S3": set()},
    )
    cafe_model = _StubCAFEModel(n_users, n_items, positive_set, seed=seed)
    baseline_model = _StubBaselineModel(n_users, n_items, seed=seed + 1)
    return bundle, cafe_model, baseline_model


# ---------------------------------------------------------------------------
# 1. per_user_metric
# ---------------------------------------------------------------------------

def test_per_user_metric_ndcg_returns_one_entry_per_positive_user():
    bundle, cafe_model, _ = _build_stub_scenario()
    scores = per_user_metric(
        cafe_model, bundle, True, "S1", 10, compute_ndcg_at_k_for_user
    )
    assert len(scores) == 15  # all 15 users have positives in S1
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_per_user_metric_empty_stage_returns_empty_dict():
    bundle, cafe_model, _ = _build_stub_scenario()
    scores = per_user_metric(
        cafe_model, bundle, True, "S2", 10, compute_ndcg_at_k_for_user
    )
    assert scores == {}


def test_per_user_metric_cafe_outscores_baseline_on_ndcg():
    """Sanity check on the stub design itself: CAFE (which ranks true
    positives near the top) should have a much higher mean NDCG@10 than
    the uninformative random baseline."""
    bundle, cafe_model, baseline_model = _build_stub_scenario()
    cafe_scores = per_user_metric(
        cafe_model, bundle, True, "S1", 10, compute_ndcg_at_k_for_user
    )
    baseline_scores = per_user_metric(
        baseline_model, bundle, False, "S1", 10, compute_ndcg_at_k_for_user
    )
    assert np.mean(list(cafe_scores.values())) > np.mean(list(baseline_scores.values()))


# ---------------------------------------------------------------------------
# 2. align_paired_arrays
# ---------------------------------------------------------------------------

def test_align_paired_arrays_matches_lengths_and_order():
    a = {2: 0.5, 0: 0.1, 1: 0.9}
    b = {2: 0.4, 0: 0.2, 1: 0.8}
    arr_a, arr_b, users = align_paired_arrays(a, b)
    assert users == [0, 1, 2]
    assert np.allclose(arr_a, [0.1, 0.9, 0.5])
    assert np.allclose(arr_b, [0.2, 0.8, 0.4])


def test_align_paired_arrays_rejects_mismatched_user_sets():
    a = {0: 0.5, 1: 0.3}
    b = {0: 0.5, 2: 0.3}
    with pytest.raises(ValueError):
        align_paired_arrays(a, b)


def test_align_paired_arrays_rejects_empty_input():
    with pytest.raises(ValueError):
        align_paired_arrays({}, {})


# ---------------------------------------------------------------------------
# 3. run_ndcg_precision_paired_comparison (end-to-end)
# ---------------------------------------------------------------------------

def test_run_comparison_end_to_end_dynamic_m():
    """With S2/S3 empty, only S1 x {5,10,20} x {NDCG, Precision} = 6
    comparisons should be produced -> dynamic M = 6, alpha* = 0.05/6.
    CAFE should be significantly better than the random baseline on all
    6 comparisons given the strong stub score separation."""
    bundle, cafe_model, baseline_model = _build_stub_scenario()
    summary = run_ndcg_precision_paired_comparison(cafe_model, baseline_model, bundle)

    assert len(summary) == 6
    expected_alpha_corrected = 0.05 / 6
    for name, r in summary.items():
        assert np.isclose(r["alpha_corrected"], expected_alpha_corrected)
        assert r["mean_diff"] > 0  # CAFE > baseline by stub construction
        assert r["significant"] is True


def test_run_comparison_n_comparisons_override_locks_to_63():
    bundle, cafe_model, baseline_model = _build_stub_scenario()
    summary = run_ndcg_precision_paired_comparison(
        cafe_model, baseline_model, bundle, n_comparisons_override=63
    )
    expected_alpha_corrected = 0.05 / 63
    for r in summary.values():
        assert np.isclose(r["alpha_corrected"], expected_alpha_corrected)


def test_run_comparison_raises_when_all_stages_empty():
    bundle = _StubBundle(n_items=10, positive_pairs_by_stage={"S1": set(), "S2": set(), "S3": set()})
    cafe_model = _StubCAFEModel(5, 10, set())
    baseline_model = _StubBaselineModel(5, 10)
    with pytest.raises(ValueError):
        run_ndcg_precision_paired_comparison(cafe_model, baseline_model, bundle)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
