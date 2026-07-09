"""
Phase 3, Step 7b — CAFE-LightFM-fixed-stage-weights ablation.

Purpose (validation-only, synthetic-v3 pipeline check — no scientific claim):
    Tests whether the empirical behavior of Full (learned stage-conditioned
    attention, W_base/W_s trained end-to-end via gradient descent) differs
    from a variant in which the SCA attention weights are FROZEN, from
    initialization, at the exact per-stage values already discovered by the
    trained Full checkpoint (Phase 3, Step 6 aggregated alpha matrix). Only
    p_u, q_i embeddings and b_u, b_i biases are trained in this variant; the
    attention distribution alpha(f, s) is a constant, non-learned tensor.

    This isolates: does *learning* the stage-conditioned attention weights
    end-to-end contribute anything beyond simply *having* a correct,
    stage-differentiated static reweighting? A null result here (Full and
    FixedStageWeights performing similarly) would suggest the benefit (if
    any) of CAFE-LightFM derives from the stage-differentiated reweighting
    itself, not specifically from gradient-based learning of it. A positive
    result (Full outperforming FixedStageWeights) would suggest end-to-end
    co-adaptation between attention and embeddings matters.

Design principle maintained (per Step 7 precedent): every new parameter
defaults to None, so all Steps 1-7 call sites remain byte-identical and
unaffected. This module is written against the DOCUMENTED interfaces of
sca_layer.py / cafe_lightfm.py / training_loop.py (per Phase 3 handoff
summaries); it has NOT been verified against the literal source files,
since those are not present in this sandbox (Colab-only environment,
per project convention). Before running in Colab:
    1. Confirm SCALayer's actual constructor signature and forward-pass
       matches the FIXED_ALPHA integration point assumed below.
    2. Confirm CAFELightFM threads `fixed_alpha` the same way it threads
       `freeze_uniform` (Step 7).
    3. Run test_step7b_fixed_stage_weights_smoke.py (bottom of this file)
       after integration, before trusting any ablation output.

Author: CAFE-LightFM project (Paper I). Seed = 42 throughout.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch

# --- Reused, not reimplemented -------------------------------------------
# These imports assume the Step 5-7 module layout documented in
# Phase3_Progress_Summary_Steps1_to_7.md. Zero re-implementation: this
# script must not duplicate ranking, NDCG, Precision, or statistical-testing
# logic — it only orchestrates training + evaluation of one new variant.
from data.synthetic_generator_v3 import generate_synthetic_dataset_v3
from data.interaction_matrix import build_interaction_matrix
from experiments.training_loop import train_cafe_lightfm
from experiments.step4_integration import run_ndcg_precision_paired_comparison
from models.cafe_lightfm.cafe_lightfm import CAFELightFM

SEED = 42

# Frozen per-stage attention weights, reused verbatim from the Phase 3
# Step 6 aggregated feature-stage matrix (checkpoint: cafe_lightfm_v3_seed42.pt).
# Order: [category, program] — matches the SCA layer's 2-feature attention
# scope confirmed in Step 6 source inspection of sca_layer.py.
FIXED_STAGE_ALPHA: Dict[str, torch.Tensor] = {
    "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
    "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
    "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
}


def _validate_fixed_alpha(fixed_alpha: Dict[str, torch.Tensor]) -> None:
    """Fail fast if the frozen attention distributions are malformed.

    Mirrors the fail-fast weight-validation pattern already established
    in experiments/sw_ndcg.py (_validate_weights), applied here to the
    frozen alpha rows instead of the SW-NDCG stage weights.
    """
    expected_stages = {"S1", "S2", "S3"}
    if set(fixed_alpha.keys()) != expected_stages:
        raise ValueError(
            f"fixed_alpha must cover exactly {expected_stages}, "
            f"got {set(fixed_alpha.keys())}"
        )
    for stage, vec in fixed_alpha.items():
        if vec.ndim != 1:
            raise ValueError(f"fixed_alpha[{stage}] must be 1-D, got shape {vec.shape}")
        if torch.any(vec < 0):
            raise ValueError(f"fixed_alpha[{stage}] contains a negative weight")
        total = vec.sum().item()
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"fixed_alpha[{stage}] must sum to 1.0 within tolerance, got {total}"
            )


def build_fixed_stage_weights_model(
    bundle,
    embedding_dim: int = 64,
    fixed_alpha: Optional[Dict[str, torch.Tensor]] = None,
) -> CAFELightFM:
    """Construct a CAFE-LightFM instance whose SCA attention is frozen.

    ASSUMPTION (labeled, pending source verification): CAFELightFM /
    SCALayer expose a `fixed_alpha` constructor argument, analogous in
    spirit to the `freeze_uniform: bool` flag added in Step 7, but holding
    a *stage-specific* tensor instead of a uniform 1/n_features constant.
    If the real sca_layer.py does not yet expose this hook, it must be
    added there first, following exactly the freeze_uniform precedent:
    default `fixed_alpha=None` (preserves Steps 1-7 behavior unchanged),
    and when provided, the forward pass indexes `fixed_alpha[stage]`
    directly instead of computing softmax(W_base @ e_f + W_s @ e_f).
    W_base and W_s remain in the module (for state_dict compatibility)
    but never enter the autograd graph when fixed_alpha is set — mirroring
    how W_stage[1]/W_stage[2] stay zero-initialized and gradient-free in
    the noStage variant.
    """
    fixed_alpha = fixed_alpha or FIXED_STAGE_ALPHA
    _validate_fixed_alpha(fixed_alpha)

    model = CAFELightFM(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        n_categories=bundle.n_categories,
        n_programs=bundle.n_programs,
        n_stages=3,
        embedding_dim=embedding_dim,
        fixed_alpha=fixed_alpha,  # NEW hook — see assumption note above
    )
    return model


def run_fixed_stage_weights_ablation(seed: int = SEED) -> dict:
    """Train + evaluate the fixed-stage-weights variant on synthetic-v3.

    Reuses, unchanged: dataset generation (Step 1.5), bundle construction,
    the Step 5 training loop (train_cafe_lightfm), and the Step 4/5 paired
    NDCG/Precision comparison runner. This function ONLY wires the new
    frozen-attention model into that existing pipeline.
    """
    torch.manual_seed(seed)

    records = generate_synthetic_dataset_v3(seed=seed)
    bundle = build_interaction_matrix(records)

    model = build_fixed_stage_weights_model(bundle)

    # train_cafe_lightfm signature per Step 5/7 handoff docs: trains
    # p_u, q_i, b_u, b_i via WARP loss. Because fixed_alpha bypasses
    # W_base/W_s in the forward pass, no ablation-specific training-loop
    # change is required beyond passing the already-configured model in —
    # unlike noStage/noAttention, which needed new training_loop.py
    # parameters (stage_idx_override, freeze_uniform).
    trained_model = train_cafe_lightfm(
        model=model,
        bundle=bundle,
        n_epochs=10,
        seed=seed,
    )

    comparison = run_ndcg_precision_paired_comparison(
        full_model_checkpoint="outputs/checkpoints/cafe_lightfm_v3_seed42.pt",
        variant_model=trained_model,
        variant_name="fixed_stage_weights",
        bundle=bundle,
        k_values=(5, 10, 20),
    )

    comparison["scope_reminder"] = (
        "VALIDATION-ONLY: synthetic-v3 pipeline check. No scientific "
        "superiority or necessity claim is made. Formal claims are "
        "reserved for the Prolific pilot (N=50) / full study (N=300)."
    )
    return comparison


# --------------------------------------------------------------------- #
# Smoke tests (self-contained, no conftest.py dependency — project convention)
# --------------------------------------------------------------------- #

def test_fixed_alpha_validation_accepts_locked_values():
    """The locked Step-6-derived weights must pass validation unchanged."""
    _validate_fixed_alpha(FIXED_STAGE_ALPHA)  # should not raise


def test_fixed_alpha_validation_rejects_bad_stage_keys():
    bad = {"S1": torch.tensor([0.5, 0.5]), "S2": torch.tensor([0.5, 0.5])}
    try:
        _validate_fixed_alpha(bad)
        raise AssertionError("expected ValueError for missing S3")
    except ValueError:
        pass


def test_fixed_alpha_validation_rejects_non_unit_sum():
    bad = {
        "S1": torch.tensor([0.5, 0.6]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }
    try:
        _validate_fixed_alpha(bad)
        raise AssertionError("expected ValueError for row not summing to 1.0")
    except ValueError:
        pass


def test_fixed_alpha_validation_rejects_negative_weight():
    bad = {
        "S1": torch.tensor([1.2, -0.2]),
        "S2": torch.tensor([0.5, 0.5]),
        "S3": torch.tensor([0.5, 0.5]),
    }
    try:
        _validate_fixed_alpha(bad)
        raise AssertionError("expected ValueError for negative weight")
    except ValueError:
        pass


def test_locked_weights_match_step6_source_values():
    """Guards against silent drift between this file and the Step 6 report."""
    assert torch.allclose(
        FIXED_STAGE_ALPHA["S1"], torch.tensor([0.6144, 0.3856]), atol=1e-4
    )
    assert torch.allclose(
        FIXED_STAGE_ALPHA["S2"], torch.tensor([0.6035, 0.3965]), atol=1e-4
    )
    assert torch.allclose(
        FIXED_STAGE_ALPHA["S3"], torch.tensor([0.6724, 0.3276]), atol=1e-4
    )


if __name__ == "__main__":
    # Local smoke check (validation logic only — does not require the
    # actual repo modules, which are not present in this sandbox).
    test_fixed_alpha_validation_accepts_locked_values()
    test_fixed_alpha_validation_rejects_bad_stage_keys()
    test_fixed_alpha_validation_rejects_non_unit_sum()
    test_fixed_alpha_validation_rejects_negative_weight()
    test_locked_weights_match_step6_source_values()
    print("All standalone smoke checks passed (validation logic only).")
    print(
        "NOTE: run_fixed_stage_weights_ablation() requires the actual "
        "repo modules (data/, experiments/, models/) and the "
        "fixed_alpha hook added to sca_layer.py — verify in Colab "
        "before trusting any trained output."
    )
