"""
test_cafe_lightfm_smoke.py

Smoke test for models/cafe_lightfm/{sca_layer,cafe_lightfm}.py (Phase 2,
Step 2). Validates:
    1. Correct output shape.
    2. Successful gradient flow, including through SCA's w_base and
       w_stage parameters.
    3. Reproducibility with seed=42.
    4. Attention weights are a valid probability distribution (sum to 1,
       non-negative) over the two item metadata features.
    5. CRITICAL: equivalence to the Step 1 baseline AT INITIALIZATION --
       CAFELightFM's score must exactly match LightFMPyTorch's score for
       identical inputs and identical embedding weights, for EVERY stage,
       confirming the |F_i| scale correction works as intended (the
       central justification for that correction, confirmed 2026-06-25).
    6. Softmax shift-invariance, illustrating WHY the proposal's original
       stage-only bias b_{s_j} was correctly omitted (a small didactic
       unit check, not specific to this model).
    7. After breaking symmetry (manually perturbing w_stage), attention
       weights DIFFER across stages for the same item -- confirming the
       model is genuinely stage-aware once trained away from
       initialization (the functional opposite of Step 1's stage-blind
       guarantee).

REQUIRES PyTorch. Run in the Colab environment (Phase 0):
    python -m tests.test_cafe_lightfm_smoke
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

N_USERS, N_ITEMS, N_CATEGORIES, N_PROGRAMS, N_STAGES = 20, 10, 1, 3, 3
EMBEDDING_DIM = 64
SEED = 42


def _build_cafe() -> CAFELightFM:
    torch.manual_seed(SEED)
    return CAFELightFM(N_USERS, N_ITEMS, N_CATEGORIES, N_PROGRAMS, N_STAGES, EMBEDDING_DIM)


def _build_baseline() -> LightFMPyTorch:
    torch.manual_seed(SEED)
    return LightFMPyTorch(N_USERS, N_ITEMS, N_CATEGORIES, N_PROGRAMS, EMBEDDING_DIM)


def test_output_shape() -> None:
    model = _build_cafe()
    batch = 16
    user_idx = torch.randint(0, N_USERS, (batch,))
    item_idx = torch.randint(0, N_ITEMS, (batch,))
    category_idx = torch.randint(0, N_CATEGORIES, (batch,))
    program_idx = torch.randint(0, N_PROGRAMS, (batch,))
    stage_idx = torch.randint(0, N_STAGES, (batch,))

    scores = model(user_idx, item_idx, category_idx, program_idx, stage_idx)
    assert scores.shape == (batch,), f"expected shape ({batch},), got {scores.shape}"
    print(f"[PASS] output shape correct: {scores.shape}")


def test_gradient_flow() -> None:
    model = _build_cafe()
    batch = 16
    user_idx = torch.randint(0, N_USERS, (batch,))
    item_idx = torch.randint(0, N_ITEMS, (batch,))
    category_idx = torch.randint(0, N_CATEGORIES, (batch,))
    program_idx = torch.randint(0, N_PROGRAMS, (batch,))
    stage_idx = torch.randint(0, N_STAGES, (batch,))

    scores = model(user_idx, item_idx, category_idx, program_idx, stage_idx)
    loss = scores.pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
    print("[PASS] loss.backward() succeeds; every parameter (including SCA's w_base/w_stage) receives a gradient")


def test_reproducibility() -> None:
    model_a = _build_cafe()
    model_b = _build_cafe()

    user_idx = torch.tensor([0, 1, 2])
    item_idx = torch.tensor([0, 1, 2])
    category_idx = torch.tensor([0, 0, 0])
    program_idx = torch.tensor([0, 1, 2])
    stage_idx = torch.tensor([0, 1, 2])

    scores_a = model_a(user_idx, item_idx, category_idx, program_idx, stage_idx)
    scores_b = model_b(user_idx, item_idx, category_idx, program_idx, stage_idx)
    assert torch.allclose(scores_a, scores_b)
    print(f"[PASS] identical seed ({SEED}) reproduces identical scores")


def test_attention_is_valid_distribution() -> None:
    model = _build_cafe()
    item_idx = torch.tensor([0, 1, 2])
    category_idx = torch.tensor([0, 0, 0])
    program_idx = torch.tensor([0, 1, 2])
    stage_idx = torch.tensor([0, 1, 2])

    _, alpha = model.item_representation(item_idx, category_idx, program_idx, stage_idx)
    assert alpha.shape == (3, 2)
    assert torch.all(alpha >= 0.0)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(3), atol=1e-6)
    print(f"[PASS] attention weights are a valid distribution over 2 features: sums={alpha.sum(dim=-1).tolist()}")


def test_equivalence_to_step1_baseline_at_init() -> None:
    """The central justification for the |F_i| scale correction
    (confirmed 2026-06-25): at initialization, CAFELightFM must produce
    IDENTICAL scores to the Step 1 baseline, for every stage."""
    cafe = _build_cafe()
    baseline = _build_baseline()

    user_idx = torch.arange(N_USERS) % N_USERS
    item_idx = torch.arange(N_USERS) % N_ITEMS
    category_idx = torch.zeros(N_USERS, dtype=torch.long)
    program_idx = torch.arange(N_USERS) % N_PROGRAMS

    baseline_scores = baseline(user_idx, item_idx, category_idx, program_idx)

    for stage in range(N_STAGES):
        stage_idx = torch.full((N_USERS,), stage, dtype=torch.long)
        cafe_scores = cafe(user_idx, item_idx, category_idx, program_idx, stage_idx)
        assert torch.allclose(cafe_scores, baseline_scores, atol=1e-6), (
            f"CAFELightFM at init (stage={stage}) does not match Step 1 baseline: "
            f"max diff = {(cafe_scores - baseline_scores).abs().max().item():.2e}"
        )
    print(f"[PASS] CAFELightFM at initialization == Step 1 baseline, for all {N_STAGES} stages "
          f"(confirms the |F_i| scale correction)")


def test_softmax_shift_invariance_didactic() -> None:
    """Didactic check (not specific to this model): illustrates WHY the
    proposal's original stage-only bias b_{s_j} was correctly identified
    as inert and omitted -- adding the SAME constant to every logit
    leaves softmax unchanged."""
    logits = torch.tensor([0.3, -0.7, 1.2])
    constant = 5.0
    p1 = F.softmax(logits, dim=-1)
    p2 = F.softmax(logits + constant, dim=-1)
    assert torch.allclose(p1, p2, atol=1e-6)
    print("[PASS] softmax shift-invariance confirmed: a stage-only bias b_{s_j} would have no "
          "effect on alpha(f, s_j), justifying its removal")


def test_stage_sensitivity_after_symmetry_breaking() -> None:
    """After perturbing w_stage away from zero, attention weights for the
    SAME item must differ across stages -- confirming genuine
    stage-awareness once the model leaves initialization."""
    model = _build_cafe()
    with torch.no_grad():
        model.sca.w_stage.weight.copy_(torch.randn(N_STAGES, EMBEDDING_DIM) * 0.5)

    item_idx = torch.tensor([0])
    category_idx = torch.tensor([0])
    program_idx = torch.tensor([0])

    alphas = []
    for stage in range(N_STAGES):
        stage_idx = torch.tensor([stage])
        _, alpha = model.item_representation(item_idx, category_idx, program_idx, stage_idx)
        alphas.append(alpha.squeeze(0))

    distinct_pairs = sum(
        1
        for i in range(N_STAGES)
        for j in range(i + 1, N_STAGES)
        if not torch.allclose(alphas[i], alphas[j], atol=1e-4)
    )
    assert distinct_pairs > 0, "attention weights are identical across all stages after symmetry breaking"
    print(f"[PASS] attention weights differ across stages for the same item after symmetry breaking "
          f"({distinct_pairs} distinct stage-pairs)")


if __name__ == "__main__":
    test_output_shape()
    test_gradient_flow()
    test_reproducibility()
    test_attention_is_valid_distribution()
    test_equivalence_to_step1_baseline_at_init()
    test_softmax_shift_invariance_didactic()
    test_stage_sensitivity_after_symmetry_breaking()
    print("=== ALL CAFE-LIGHTFM (STEP 2) SMOKE TESTS PASSED ===")
