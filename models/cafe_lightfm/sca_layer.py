"""
sca_layer.py

Phase 2, Step 2: Stage-Conditioned Attention (SCA) layer, extending the
Step 1 LightFM baseline (Kula, 2015) [1] with a per-stage feature-attention
mechanism (proposal Section 5.2).

Mathematical specification (confirmed 2026-06-25, refining the proposal's
original Section 5.2 equation):

    logit(f, s_j) = w_base . e_f + w_{s_j} . e_f
    alpha(f, s_j) = softmax_f( logit(f, s_j) )              over f in F_i
    weighted_sum  = |F_i| * sum_{f in F_i} alpha(f, s_j) * e_f

Two deliberate, confirmed departures from the proposal's literal equation:

    1. SCALE CORRECTION (the |F_i| factor). Softmax weights sum to 1, not
       |F_i|, so a naive sum_f alpha(f,s_j)*e_f shrinks the metadata
       contribution by a factor of |F_i| relative to the Step 1 baseline
       (q_i = e_item_id + sum_f e_f), even at "neutral" (near-uniform)
       attention. Multiplying by |F_i| restores exact equivalence to the
       Step 1 baseline at initialization.
    2. BIAS REMOVAL. The proposal's original "+ b_{s_j}" term is
       mathematically INERT under softmax (shift-invariance) and is
       omitted entirely -- see module history for full rationale.

Item-feature scope (confirmed 2026-06-25): attention is applied ONLY to
item_category and program_type (item_id excluded -- no shared vocabulary
across items).

PHASE 3, STEP 7 ADDITION (2026-07-06, confirmed by PI):
    `freeze_uniform: bool = False` added to `forward()`. This is the
    mechanism for the "CAFE-LightFM-noAttention" ablation (proposal
    Section 5.6): when True, the w_base/w_stage logit computation is
    SKIPPED entirely and alpha is set to a constant uniform tensor
    (1/n_features per feature). Default is `False`, so all existing
    (Step 1-6) call sites are unaffected.

PHASE 3, STEP 7b ADDITION:
    `fixed_alpha: Optional[Dict[str, torch.Tensor]] = None` added to
    `__init__()`. This supports the "CAFE-LightFM-fixed-stage-weights"
    ablation: when provided, alpha is not learned and is not computed
    through softmax. Instead, the layer directly uses a fixed per-stage
    attention vector. This preserves the original learned w_stage path
    unchanged when fixed_alpha is None.

References
----------
[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. Proceedings of the 2nd Workshop on New Trends in
    Content-Based Recommender Systems, RecSys 2015. CEUR-WS, Vol. 1448,
    14-21.
[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
    Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All
    You Need. Advances in Neural Information Processing Systems
    (NeurIPS), 30.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fixed_alpha_to_tensor(
    fixed_alpha: Dict[str, torch.Tensor],
    n_stages: int,
) -> torch.Tensor:
    """Validate and convert fixed per-stage alpha weights to a tensor.

    Expected input format:
        {
            "S1": tensor([category_weight, program_weight]),
            "S2": tensor([category_weight, program_weight]),
            "S3": tensor([category_weight, program_weight]),
        }

    Returns
    -------
    torch.Tensor, shape (n_stages, n_features)
        Rows ordered as S1, S2, S3, ...
    """
    expected_keys = {f"S{i + 1}" for i in range(n_stages)}
    actual_keys = set(fixed_alpha.keys())

    if actual_keys != expected_keys:
        raise ValueError(
            f"fixed_alpha must cover exactly {expected_keys}, got {actual_keys}"
        )

    rows = []
    expected_shape = None

    for i in range(n_stages):
        stage_name = f"S{i + 1}"
        vec = fixed_alpha[stage_name]

        if not torch.is_tensor(vec):
            vec = torch.tensor(vec, dtype=torch.float32)

        vec = vec.detach().clone().to(dtype=torch.float32)

        if vec.ndim != 1:
            raise ValueError(
                f"fixed_alpha[{stage_name}] must be 1-D, got shape {tuple(vec.shape)}"
            )

        if expected_shape is None:
            expected_shape = vec.shape
        elif vec.shape != expected_shape:
            raise ValueError(
                f"All fixed_alpha rows must have the same shape. "
                f"Expected {tuple(expected_shape)}, got {tuple(vec.shape)} "
                f"for {stage_name}."
            )

        if not torch.isfinite(vec).all():
            raise ValueError(f"fixed_alpha[{stage_name}] contains non-finite values")

        if torch.any(vec < 0):
            raise ValueError(f"fixed_alpha[{stage_name}] contains negative weights")

        total = vec.sum().item()
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"fixed_alpha[{stage_name}] must sum to 1.0 within tolerance, "
                f"got {total}"
            )

        rows.append(vec)

    return torch.stack(rows, dim=0)


class StageConditionedAttention(nn.Module):
    """Computes alpha(f, s_j) and the scale-corrected, stage-weighted sum
    of item metadata feature embeddings."""

    def __init__(
        self,
        n_stages: int,
        embedding_dim: int,
        fixed_alpha: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.embedding_dim = embedding_dim

        # w_base: single vector shared across all stages (Section 5.2).
        # Zero-initialized: at init, logits are all 0 regardless of input,
        # so alpha is exactly uniform (1/|F_i| each) -- required for the
        # Step 1 equivalence property (see module docstring, point 1).
        self.w_base = nn.Parameter(torch.zeros(embedding_dim))

        # w_{s_j}: one vector per stage. Also zero-initialized for the
        # same reason.
        self.w_stage = nn.Embedding(n_stages, embedding_dim)
        nn.init.zeros_(self.w_stage.weight)

        # NOTE: no bias parameter here -- see module docstring, point 2
        # (the proposal's stage-only bias b_{s_j} cancels under softmax).

        # NEW: Optional fixed per-stage alpha matrix for Step 7b.
        # When None, all previous learned-attention behavior is unchanged.
        if fixed_alpha is None:
            self.register_buffer("fixed_alpha_tensor", None)
        else:
            fixed_alpha_tensor = _fixed_alpha_to_tensor(fixed_alpha, n_stages)
            self.register_buffer("fixed_alpha_tensor", fixed_alpha_tensor)

    def forward(
        self,
        feature_embeddings: torch.Tensor,
        stage_idx: torch.Tensor,
        freeze_uniform: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature_embeddings : torch.Tensor, shape (batch, n_features, d)
        stage_idx           : torch.Tensor, shape (batch,)
        freeze_uniform       : bool, default False
            Phase 3, Step 7 (noAttention ablation). When True, bypasses
            the w_base/w_stage logit computation and softmax entirely,
            returning a constant uniform alpha = 1/n_features. w_base
            and w_stage receive no gradient in this branch. Default
            False preserves all Step 1-6 behavior exactly.

        Returns
        -------
        weighted_sum : torch.Tensor, shape (batch, d)
            |F_i| * sum_f alpha(f, s_j) * e_f  (scale-corrected).
        alpha        : torch.Tensor, shape (batch, n_features)
            Attention weights, summing to 1 across the feature axis,
            retained for interpretability analysis (proposal Section 6.2).
        """
        n_features = feature_embeddings.shape[1]

        # NEW: Step 7b fixed-stage-weights ablation.
        # If fixed_alpha is provided, bypass learned softmax attention.
        # This branch does NOT touch w_base or w_stage, so their gradients
        # remain None/zero for this forward path.
        if self.fixed_alpha_tensor is not None:
            if freeze_uniform:
                raise ValueError(
                    "fixed_alpha and freeze_uniform are mutually exclusive. "
                    "Use only one fixed-attention mode at a time."
                )

            if stage_idx.ndim != 1:
                raise ValueError(
                    f"stage_idx must be 1-D with shape (batch,), got {tuple(stage_idx.shape)}"
                )

            if self.fixed_alpha_tensor.shape[1] != n_features:
                raise ValueError(
                    f"fixed_alpha feature dimension mismatch: "
                    f"fixed_alpha has {self.fixed_alpha_tensor.shape[1]} features, "
                    f"but feature_embeddings has {n_features} features."
                )

            if torch.any(stage_idx < 0) or torch.any(stage_idx >= self.n_stages):
                raise ValueError(
                    f"stage_idx contains values outside valid range [0, {self.n_stages - 1}]"
                )

            stage_idx_for_buffer = stage_idx.to(
                device=self.fixed_alpha_tensor.device,
                dtype=torch.long,
            )

            alpha = self.fixed_alpha_tensor[stage_idx_for_buffer].to(
                dtype=feature_embeddings.dtype,
                device=feature_embeddings.device,
            )

            weighted = torch.einsum("bf,bfd->bd", alpha, feature_embeddings)
            return n_features * weighted, alpha

        if freeze_uniform:
            batch = feature_embeddings.shape[0]
            alpha = torch.full(
                (batch, n_features),
                1.0 / n_features,
                dtype=feature_embeddings.dtype,
                device=feature_embeddings.device,
            )
            weighted = torch.einsum("bf,bfd->bd", alpha, feature_embeddings)
            return n_features * weighted, alpha

        # ORIGINAL LEARNED PATH — unchanged.
        w_s = self.w_stage(stage_idx)  # (batch, d)

        logit_base = torch.einsum("bfd,d->bf", feature_embeddings, self.w_base)
        logit_stage = torch.einsum("bfd,bd->bf", feature_embeddings, w_s)
        logits = logit_base + logit_stage  # (batch, n_features)

        alpha = F.softmax(logits, dim=-1)  # sums to 1 over the feature axis
        weighted = torch.einsum("bf,bfd->bd", alpha, feature_embeddings)  # (batch, d)

        return n_features * weighted, alpha


if __name__ == "__main__":
    # Self-contained smoke test (no conftest dependency), per project
    # test-pattern convention.
    def _setup():
        torch.manual_seed(42)
        layer = StageConditionedAttention(n_stages=3, embedding_dim=4)
        feats = torch.randn(5, 2, 4)  # batch=5, n_features=2, d=4
        stage_idx = torch.tensor([0, 1, 2, 0, 1])
        return layer, feats, stage_idx

    layer, feats, stage_idx = _setup()

    # 1. Default (freeze_uniform=False) at zero-init must be exactly
    #    uniform (1/n_features) -- pre-existing equivalence property.
    weighted_default, alpha_default = layer(feats, stage_idx)
    assert torch.allclose(
        alpha_default,
        torch.full_like(alpha_default, 0.5),
        atol=1e-6,
    ), "Zero-init default forward should be exactly uniform alpha."

    # 2. freeze_uniform=True must produce identical alpha to (1).
    weighted_frozen, alpha_frozen = layer(feats, stage_idx, freeze_uniform=True)
    assert torch.allclose(
        alpha_frozen,
        alpha_default,
        atol=1e-6,
    ), "freeze_uniform output should match the zero-init uniform case."

    assert torch.allclose(
        weighted_frozen,
        weighted_default,
        atol=1e-6,
    ), "freeze_uniform weighted sum should match zero-init weighted sum."

    # 3. freeze_uniform=True must NOT route gradient into w_base/w_stage.
    feats_for_grad = feats.detach().clone().requires_grad_(True)
    weighted_frozen_grad, _ = layer(
        feats_for_grad,
        stage_idx,
        freeze_uniform=True,
    )
    loss = weighted_frozen_grad.sum()
    loss.backward()

    assert layer.w_base.grad is None or torch.all(layer.w_base.grad == 0), (
        "w_base must receive zero/None gradient when freeze_uniform=True."
    )
    assert layer.w_stage.weight.grad is None or torch.all(layer.w_stage.weight.grad == 0), (
        "w_stage must receive zero/None gradient when freeze_uniform=True."
    )

    # 4. alpha rows must sum to 1 in both default and freeze_uniform modes.
    assert torch.allclose(alpha_default.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(alpha_frozen.sum(dim=-1), torch.ones(5), atol=1e-6)

    # 5. fixed_alpha must return the exact per-stage alpha values.
    fixed_alpha = {
        "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
        "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
        "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
    }
    fixed_layer = StageConditionedAttention(
        n_stages=3,
        embedding_dim=4,
        fixed_alpha=fixed_alpha,
    )

    weighted_fixed, alpha_fixed = fixed_layer(feats, stage_idx)
    expected_alpha = torch.stack(
        [
            fixed_alpha["S1"],
            fixed_alpha["S2"],
            fixed_alpha["S3"],
            fixed_alpha["S1"],
            fixed_alpha["S2"],
        ],
        dim=0,
    )

    assert torch.allclose(alpha_fixed, expected_alpha, atol=1e-6), (
        "fixed_alpha mode must return the exact locked per-stage values."
    )
    assert torch.allclose(alpha_fixed.sum(dim=-1), torch.ones(5), atol=1e-6), (
        "fixed_alpha rows must sum to 1."
    )

    # 6. fixed_alpha must NOT route gradient into w_base/w_stage.
    feats_fixed_grad = feats.detach().clone().requires_grad_(True)
    weighted_fixed_grad, _ = fixed_layer(feats_fixed_grad, stage_idx)
    fixed_loss = weighted_fixed_grad.sum()
    fixed_loss.backward()

    assert fixed_layer.w_base.grad is None or torch.all(fixed_layer.w_base.grad == 0), (
        "w_base must receive zero/None gradient when fixed_alpha is used."
    )
    assert (
        fixed_layer.w_stage.weight.grad is None
        or torch.all(fixed_layer.w_stage.weight.grad == 0)
    ), "w_stage must receive zero/None gradient when fixed_alpha is used."

    # 7. fixed_alpha and freeze_uniform must be mutually exclusive.
    try:
        fixed_layer(feats, stage_idx, freeze_uniform=True)
        raise AssertionError("Expected ValueError when fixed_alpha and freeze_uniform are both used.")
    except ValueError:
        pass

    print("sca_layer.py smoke test: all checks passed.")
