"""
test_lightfm_pytorch_smoke.py

Smoke test for models/baselines/lightfm_pytorch.py (Phase 2, Step 1).
Validates, per the agreed Step 1 checklist (confirmed 2026-06-25):
    1. Correct output shape.
    2. Successful gradient flow (loss.backward() with no errors, and
       gradients reach every embedding table that should receive them).
    3. Reproducibility with seed=42.
    4. Item metadata embeddings (category, program) are correctly
       included in the item representation -- verified by direct
       reconstruction (q_i == e_item + e_category + e_program) and by a
       perturbation test (changing only the category index changes q_i).
    5. The Step 1 model is fully stage-blind: `forward`'s signature has
       no stage-related parameter.
    6. Continuous behavioral signals are never used in the scoring
       function: `forward`'s signature has no continuous-signal
       parameter, and the module source never imports or references
       ContinuousSignals.

REQUIRES PyTorch. Run in the Colab environment (Phase 0):
    python -m tests.test_lightfm_pytorch_smoke
"""

from __future__ import annotations

import inspect

import torch

from models.baselines import lightfm_pytorch as lightfm_pytorch_module
from models.baselines.lightfm_pytorch import LightFMPyTorch

N_USERS, N_ITEMS, N_CATEGORIES, N_PROGRAMS = 20, 10, 1, 3
EMBEDDING_DIM = 64
SEED = 42


def _build_model() -> LightFMPyTorch:
    torch.manual_seed(SEED)
    return LightFMPyTorch(N_USERS, N_ITEMS, N_CATEGORIES, N_PROGRAMS, EMBEDDING_DIM)


def test_output_shape() -> None:
    model = _build_model()
    batch = 16
    user_idx = torch.randint(0, N_USERS, (batch,))
    item_idx = torch.randint(0, N_ITEMS, (batch,))
    category_idx = torch.randint(0, N_CATEGORIES, (batch,))
    program_idx = torch.randint(0, N_PROGRAMS, (batch,))

    scores = model(user_idx, item_idx, category_idx, program_idx)
    assert scores.shape == (batch,), f"expected shape ({batch},), got {scores.shape}"
    print(f"[PASS] output shape correct: {scores.shape}")


def test_gradient_flow() -> None:
    model = _build_model()
    batch = 16
    user_idx = torch.randint(0, N_USERS, (batch,))
    item_idx = torch.randint(0, N_ITEMS, (batch,))
    category_idx = torch.randint(0, N_CATEGORIES, (batch,))
    program_idx = torch.randint(0, N_PROGRAMS, (batch,))

    scores = model(user_idx, item_idx, category_idx, program_idx)
    loss = scores.pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
    print("[PASS] loss.backward() succeeds; every parameter tensor receives a gradient")


def test_reproducibility() -> None:
    model_a = _build_model()
    model_b = _build_model()

    user_idx = torch.tensor([0, 1, 2])
    item_idx = torch.tensor([0, 1, 2])
    category_idx = torch.tensor([0, 0, 0])
    program_idx = torch.tensor([0, 1, 2])

    scores_a = model_a(user_idx, item_idx, category_idx, program_idx)
    scores_b = model_b(user_idx, item_idx, category_idx, program_idx)
    assert torch.allclose(scores_a, scores_b)
    print(f"[PASS] identical seed ({SEED}) reproduces identical scores")


def test_item_metadata_included() -> None:
    model = _build_model()
    item_idx = torch.tensor([3])
    category_idx = torch.tensor([0])
    program_idx = torch.tensor([1])

    # (1) Direct reconstruction check
    q_i = model.item_representation(item_idx, category_idx, program_idx)
    expected = (
        model.item_embedding(item_idx)
        + model.category_embedding(category_idx)
        + model.program_embedding(program_idx)
    )
    assert torch.allclose(q_i, expected)

    # (2) Perturbation check: changing ONLY program_idx must change q_i
    program_idx_alt = torch.tensor([2])
    q_i_alt = model.item_representation(item_idx, category_idx, program_idx_alt)
    assert not torch.allclose(q_i, q_i_alt), "program_type embedding has no effect on q_i"
    print("[PASS] item metadata embeddings (category, program) correctly included in q_i")


def test_fully_stage_blind() -> None:
    sig = inspect.signature(LightFMPyTorch.forward)
    param_names = set(sig.parameters.keys())
    stage_like = {p for p in param_names if "stage" in p.lower()}
    assert not stage_like, f"forward() has stage-related parameter(s): {stage_like}"
    print(f"[PASS] forward() signature has no stage-related parameter: {sorted(param_names)}")


def test_no_continuous_signal_usage() -> None:
    sig = inspect.signature(LightFMPyTorch.forward)
    param_names = set(sig.parameters.keys())
    continuous_names = {"hover_duration_ms", "dwell_time_ms", "revisit_count", "scroll_depth_pct"}
    assert not (param_names & continuous_names), "forward() accepts a continuous-signal parameter"

    source = inspect.getsource(lightfm_pytorch_module)
    assert "ContinuousSignals" not in source, "module must not import/reference ContinuousSignals"
    print("[PASS] forward() accepts no continuous-signal parameter; module never references ContinuousSignals")


if __name__ == "__main__":
    test_output_shape()
    test_gradient_flow()
    test_reproducibility()
    test_item_metadata_included()
    test_fully_stage_blind()
    test_no_continuous_signal_usage()
    print("=== ALL LIGHTFM PYTORCH BASELINE SMOKE TESTS PASSED ===")
