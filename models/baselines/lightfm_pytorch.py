"""
lightfm_pytorch.py

Phase 2, Step 1: pure-PyTorch reimplementation of the LightFM hybrid
matrix-factorization scoring function [1], per the frozen Step 0 interface
(Phase2_Step0_Model_Interface_Freeze.md). STAGE-BLIND by construction: the
forward signature has no stage argument and no continuous-signal argument
-- this is a structural property of the class, not a runtime check.

Motivation: lightfm==1.17 cannot build under Python 3.12 (Phase 0 blocker
-- PyLongObject's ob_digit removal breaks LightFM's Cython extensions).
This module is the literature baseline named in the proposal's Section
5.5, reimplemented faithfully rather than substituted with a generic MF
model.

Scoring function (frozen interface, Section 5):
    r_hat(u, i) = q_u . q_i + b_u + b_i
    q_u = e_user[u]
    q_i = e_item[i] + e_category[cat(i)] + e_program[program(i)]

Deliberately OUT OF SCOPE for this module (deferred to later Phase 2
steps, confirmed 2026-06-25):
    - WARP loss (Step 3)
    - training loop (Step 4)
    - Stage-Conditioned Attention / alpha(f, s_j) (Step 2)

References
----------
[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. Proceedings of the 2nd Workshop on New Trends in
    Content-Based Recommender Systems, RecSys 2015. CEUR-WS, Vol. 1448,
    14-21.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LightFMPyTorch(nn.Module):
    """Stage-blind hybrid matrix-factorization baseline (Kula, 2015)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_categories: int,
        n_programs: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        self.program_embedding = nn.Embedding(n_programs, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self._init_weights()

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

    def item_representation(
        self,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        program_idx: torch.Tensor,
    ) -> torch.Tensor:
        """q_i = e_item + e_category + e_program (Kula, 2015, hybrid sum)."""
        return (
            self.item_embedding(item_idx)
            + self.category_embedding(category_idx)
            + self.program_embedding(program_idx)
        )

    def user_representation(self, user_idx: torch.Tensor) -> torch.Tensor:
        """q_u = e_user (ID-only; no user metadata in the current schema,
        per the Step 0 interface, Section 2)."""
        return self.user_embedding(user_idx)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        program_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        r_hat(u, i) = q_u . q_i + b_u + b_i

        All four arguments are LongTensors of identical shape (batch,).
        Returns a FloatTensor of shape (batch,).

        NOTE: this signature intentionally has NO stage argument and NO
        continuous-signal argument (hover_duration_ms, dwell_time_ms,
        revisit_count, scroll_depth_pct) -- enforced structurally by the
        absence of corresponding parameters and embedding tables, not by
        a runtime guard. Stage-conditioning is introduced only in Step 2
        (Stage-Conditioned Attention).
        """
        q_u = self.user_representation(user_idx)
        q_i = self.item_representation(item_idx, category_idx, program_idx)
        dot = (q_u * q_i).sum(dim=-1)
        b_u = self.user_bias(user_idx).squeeze(-1)
        b_i = self.item_bias(item_idx).squeeze(-1)
        return dot + b_u + b_i
