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
    (1/n_features per feature). This guarantees w_base and w_stage never
    enter the autograd graph for that forward call -- they receive zero
    gradient and remain at their zero-initialized values throughout
    training, which is exactly the "stage-specific embeddings without
    attention weighting" condition the proposal specifies. Because the
    layer is already verified to be numerically equivalent to the Step 1
    baseline at zero-init (see module docstring, point 1, and
    `test_equivalence_to_step1_baseline_at_init`), this flag reproduces
    that same equivalence but HOLDS it fixed for the entire training run,
    rather than allowing it to be an artifact of initialization only.
    Default is `False`, so all existing (Step 1-6) call sites are
    unaffected -- this is an additive, backward-compatible change.

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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageConditionedAttention(nn.Module):
    """Computes alpha(f, s_j) and the scale-corrected, stage-weighted sum
    of item metadata feature embeddings."""

    def __init__(self, n_stages: int, embedding_dim: int) -> None:
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
    assert torch.allclose(alpha_default, torch.full_like(alpha_default, 0.5), atol=1e-6), (
        "Zero-init default forward should be exactly uniform alpha."
    )

    # 2. freeze_uniform=True must produce identical alpha to (1), and
    #    must NOT route gradient into w_base/w_stage.
    weighted_frozen, alpha_frozen = layer(feats, stage_idx, freeze_uniform=True)
    assert torch.allclose(alpha_frozen, alpha_default, atol=1e-6), (
        "freeze_uniform output should match the zero-init uniform case."
    )
    assert torch.allclose(weighted_frozen, weighted_default, atol=1e-6), (
        "freeze_uniform weighted sum should match zero-init weighted sum."
    )
    loss = weighted_frozen.sum()
    loss.backward()
    assert layer.w_base.grad is None or torch.all(layer.w_base.grad == 0), (
        "w_base must receive zero/None gradient when freeze_uniform=True."
    )
    assert layer.w_stage.weight.grad is None or torch.all(layer.w_stage.weight.grad == 0), (
        "w_stage must receive zero/None gradient when freeze_uniform=True."
    )

    # 3. alpha rows must sum to 1 in both modes.
    assert torch.allclose(alpha_default.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(alpha_frozen.sum(dim=-1), torch.ones(5), atol=1e-6)

    print("sca_layer.py smoke test: all checks passed.")
