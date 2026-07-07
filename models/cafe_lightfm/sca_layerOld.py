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
       Step 1 baseline at initialization. This matters for two reasons:
       (a) the proposal's own "CAFE-LightFM-noStage" ablation (Section
           5.6) is defined as "uniform attention weights" and should
           reproduce the Step 1/LightFM baseline NUMERICALLY, not merely
           qualitatively;
       (b) training stability benefits from starting at a point
           equivalent to a known-working baseline rather than an
           arbitrarily rescaled one.

    2. BIAS REMOVAL. The proposal's original "+ b_{s_j}" term (a bias
       depending on stage ALONE, not on the feature f) is mathematically
       INERT under softmax: softmax is shift-invariant
       (softmax(x + c) = softmax(x) for any constant c added to every
       coordinate), and b_{s_j} is exactly such a constant across every
       f in F_i for fixed s_j. It therefore has ZERO effect on
       alpha(f, s_j) in the original formula as written -- this is a
       genuine correction to the proposal's Section 5.2 equation, not a
       simplification choice. The bias is omitted entirely here (rather
       than replaced with the feature-and-stage-specific alternative
       b_{f,s_j}, which does NOT cancel): with only two item metadata
       features (item_category, program_type) at this stage, b_{f,s_j}
       would add parameters without a clear expressivity need. Revisit
       if/when the Prolific catalog provides a richer item-feature set.

Item-feature scope (confirmed 2026-06-25, consistent with the Step 0
interface, Phase2_Step0_Model_Interface_Freeze.md): attention is applied
ONLY to item_category and program_type. item_id is excluded -- it has no
shared vocabulary across items, so "attention" over a single, item-unique
embedding has no meaningful interpretation as feature salience.

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
        self, feature_embeddings: torch.Tensor, stage_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature_embeddings : torch.Tensor, shape (batch, n_features, d)
        stage_idx           : torch.Tensor, shape (batch,)

        Returns
        -------
        weighted_sum : torch.Tensor, shape (batch, d)
            |F_i| * sum_f alpha(f, s_j) * e_f  (scale-corrected).
        alpha        : torch.Tensor, shape (batch, n_features)
            Attention weights, summing to 1 across the feature axis,
            retained for interpretability analysis (proposal Section 6.2).
        """
        n_features = feature_embeddings.shape[1]
        w_s = self.w_stage(stage_idx)  # (batch, d)

        logit_base = torch.einsum("bfd,d->bf", feature_embeddings, self.w_base)
        logit_stage = torch.einsum("bfd,bd->bf", feature_embeddings, w_s)
        logits = logit_base + logit_stage  # (batch, n_features)

        alpha = F.softmax(logits, dim=-1)  # sums to 1 over the feature axis
        weighted = torch.einsum("bf,bfd->bd", alpha, feature_embeddings)  # (batch, d)

        return n_features * weighted, alpha
