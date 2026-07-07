"""
cafe_lightfm.py

Phase 2, Step 2: full CAFE-LightFM model -- the Step 1 LightFM baseline's
embedding structure, with item metadata aggregation replaced by the
Stage-Conditioned Attention (SCA) layer (sca_layer.py).

Scoring function:
    r_hat(u, i, s_j) = q_u . q_i(s_j) + b_u + b_i
    q_u    = e_user[u]                                  (unchanged from Step 1)
    q_i(s_j) = e_item[i] + SCA(e_category[cat(i)], e_program[program(i)]; s_j)

At initialization (SCA's w_base, w_stage both zero), q_i(s_j) is IDENTICAL
to the Step 1 baseline's q_i for every s_j -- verified in
test_cafe_lightfm_smoke.py's `test_equivalence_to_step1_baseline_at_init`.
This is a deliberate design property (see sca_layer.py docstring), not a
coincidence.

References
----------
[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop on New Trends in CBRS.
    CEUR-WS, Vol. 1448, 14-21.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models.cafe_lightfm.sca_layer import StageConditionedAttention


class CAFELightFM(nn.Module):
    """Stage-Conditioned Attention Framework for hybrid matrix
    factorization (proposal Section 5.2)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_categories: int,
        n_programs: int,
        n_stages: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Identical embedding structure to Step 1's LightFMPyTorch,
        # constructed in the SAME ORDER, so that seeding the RNG
        # identically before each model's construction yields identical
        # embedding weights (required for the Step 1 equivalence test).
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        self.program_embedding = nn.Embedding(n_programs, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self._init_weights()

        # SCA's own parameters are deterministically zero-initialized
        # (see sca_layer.py) and therefore consume no RNG state, so
        # placing this after the embeddings above does not disturb the
        # RNG-alignment property used by the equivalence test.
        self.sca = StageConditionedAttention(n_stages, embedding_dim)

    def _init_weights(self) -> None:
        for emb in (
            self.user_embedding,
            self.item_embedding,
            self.category_embedding,
            self.program_embedding,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def user_representation(self, user_idx: torch.Tensor) -> torch.Tensor:
        """q_u = e_user (ID-only; unchanged from Step 1)."""
        return self.user_embedding(user_idx)

    def item_representation(
        self,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        program_idx: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_i(s_j) = e_item + SCA(e_category, e_program; s_j)

        Returns (q_i, alpha) where alpha has shape (batch, 2) -- attention
        weights over [item_category, program_type], in that order.
        """
        cat_emb = self.category_embedding(category_idx)  # (batch, d)
        prog_emb = self.program_embedding(program_idx)    # (batch, d)
        feature_embeddings = torch.stack([cat_emb, prog_emb], dim=1)  # (batch, 2, d)

        weighted_sum, alpha = self.sca(feature_embeddings, stage_idx)
        q_i = self.item_embedding(item_idx) + weighted_sum
        return q_i, alpha

    def forward_with_attention(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        program_idx: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (score, alpha) -- use this when attention weights are
        needed for interpretability analysis (proposal Section 6.2)."""
        q_u = self.user_representation(user_idx)
        q_i, alpha = self.item_representation(item_idx, category_idx, program_idx, stage_idx)
        dot = (q_u * q_i).sum(dim=-1)
        b_u = self.user_bias(user_idx).squeeze(-1)
        b_i = self.item_bias(item_idx).squeeze(-1)
        return dot + b_u + b_i, alpha

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        program_idx: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Score only -- signature-compatible with how Step 3's WARP loss
        will consume the model (no attention weights threaded through)."""
        score, _alpha = self.forward_with_attention(
            user_idx, item_idx, category_idx, program_idx, stage_idx
        )
        return score
