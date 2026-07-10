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

PHASE 3, STEP 7 ADDITION (2026-07-06, confirmed by PI):
    `freeze_uniform: bool = False` threaded through `item_representation`,
    `forward_with_attention`, and `forward`, passed straight to the SCA
    layer (sca_layer.py). Supports the "CAFE-LightFM-noAttention"
    ablation (proposal Section 5.6). Default False preserves Step 1-6
    behavior exactly (additive, backward-compatible).

PHASE 3, STEP 7b ADDITION:
    `fixed_alpha: Optional[Dict[str, torch.Tensor]] = None` added to
    `__init__()` and passed into StageConditionedAttention. This supports
    the "CAFE-LightFM-fixed-stage-weights" ablation. Default None preserves
    all previous Step 1-7 behavior.

References
----------
[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop on New Trends in CBRS.
    CEUR-WS, Vol. 1448, 14-21.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        fixed_alpha: Optional[Dict[str, torch.Tensor]] = None,
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
        #
        # NEW: fixed_alpha is passed through for Step 7b. When None,
        # StageConditionedAttention preserves the original learned path.
        self.sca = StageConditionedAttention(
            n_stages=n_stages,
            embedding_dim=embedding_dim,
            fixed_alpha=fixed_alpha,
        )

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
        freeze_uniform: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_i(s_j) = e_item + SCA(e_category, e_program; s_j)

        Parameters
        ----------
        freeze_uniform : bool, default False
            Phase 3, Step 7 (noAttention ablation). Passed straight to
            the SCA layer. See sca_layer.py for the exact semantics.

        Returns
        -------
        (q_i, alpha) where alpha has shape (batch, 2) -- attention
        weights over [item_category, program_type], in that order.
        """
        cat_emb = self.category_embedding(category_idx)    # (batch, d)
        prog_emb = self.program_embedding(program_idx)     # (batch, d)

        feature_embeddings = torch.stack(
            [cat_emb, prog_emb],
            dim=1,
        )  # (batch, 2, d)

        weighted_sum, alpha = self.sca(
            feature_embeddings,
            stage_idx,
            freeze_uniform=freeze_uniform,
        )

        q_i = self.item_embedding(item_idx) + weighted_sum
        return q_i, alpha

    def forward_with_attention(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        program_idx: torch.Tensor,
        stage_idx: torch.Tensor,
        freeze_uniform: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (score, alpha) -- use this when attention weights are
        needed for interpretability analysis (proposal Section 6.2)."""
        q_u = self.user_representation(user_idx)

        q_i, alpha = self.item_representation(
            item_idx,
            category_idx,
            program_idx,
            stage_idx,
            freeze_uniform=freeze_uniform,
        )

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
        freeze_uniform: bool = False,
    ) -> torch.Tensor:
        """Score only -- signature-compatible with how Step 3's WARP loss
        consumes the model (no attention weights threaded through).

        Parameters
        ----------
        freeze_uniform : bool, default False
            Phase 3, Step 7 (noAttention ablation). Default preserves
            Step 1-6 call sites exactly.
        """
        score, _alpha = self.forward_with_attention(
            user_idx,
            item_idx,
            category_idx,
            program_idx,
            stage_idx,
            freeze_uniform=freeze_uniform,
        )

        return score


if __name__ == "__main__":
    # Self-contained smoke test (no conftest dependency).
    def _setup(fixed_alpha=None):
        torch.manual_seed(42)

        model = CAFELightFM(
            n_users=4,
            n_items=6,
            n_categories=1,
            n_programs=3,
            n_stages=3,
            embedding_dim=8,
            fixed_alpha=fixed_alpha,
        )

        user_idx = torch.tensor([0, 1, 2, 3])
        item_idx = torch.tensor([0, 1, 2, 3])
        category_idx = torch.zeros(4, dtype=torch.long)
        program_idx = torch.tensor([0, 1, 2, 0])
        stage_idx = torch.tensor([0, 1, 2, 0])

        return model, user_idx, item_idx, category_idx, program_idx, stage_idx

    model, user_idx, item_idx, category_idx, program_idx, stage_idx = _setup()

    # 1. Default forward must still run and return a (batch,) score tensor.
    scores_default = model(
        user_idx,
        item_idx,
        category_idx,
        program_idx,
        stage_idx,
    )

    assert scores_default.shape == (4,), "Default forward output shape mismatch."

    # 2. freeze_uniform=True must also run, and at zero-init must produce
    #    identical scores to the default path.
    scores_frozen = model(
        user_idx,
        item_idx,
        category_idx,
        program_idx,
        stage_idx,
        freeze_uniform=True,
    )

    assert torch.allclose(scores_default, scores_frozen, atol=1e-6), (
        "At zero-init, freeze_uniform=True scores must match default scores."
    )

    # 3. forward_with_attention must expose alpha with the correct shape.
    _, alpha = model.forward_with_attention(
        user_idx,
        item_idx,
        category_idx,
        program_idx,
        stage_idx,
        freeze_uniform=True,
    )

    assert alpha.shape == (4, 2), "Alpha shape must be (batch, 2)."

    # 4. fixed_alpha model must run and return the expected per-stage alpha.
    fixed_alpha = {
        "S1": torch.tensor([0.6144, 0.3856], dtype=torch.float32),
        "S2": torch.tensor([0.6035, 0.3965], dtype=torch.float32),
        "S3": torch.tensor([0.6724, 0.3276], dtype=torch.float32),
    }

    fixed_model, user_idx, item_idx, category_idx, program_idx, stage_idx = _setup(
        fixed_alpha=fixed_alpha
    )

    fixed_scores, fixed_alpha_out = fixed_model.forward_with_attention(
        user_idx,
        item_idx,
        category_idx,
        program_idx,
        stage_idx,
    )

    assert fixed_scores.shape == (4,), "fixed_alpha forward output shape mismatch."
    assert fixed_alpha_out.shape == (4, 2), "fixed_alpha alpha shape must be (batch, 2)."

    expected_alpha = torch.stack(
        [
            fixed_alpha["S1"],
            fixed_alpha["S2"],
            fixed_alpha["S3"],
            fixed_alpha["S1"],
        ],
        dim=0,
    )

    assert torch.allclose(fixed_alpha_out, expected_alpha, atol=1e-6), (
        "fixed_alpha model must return the exact locked per-stage attention values."
    )

    # 5. fixed_alpha and freeze_uniform must be mutually exclusive.
    try:
        fixed_model(
            user_idx,
            item_idx,
            category_idx,
            program_idx,
            stage_idx,
            freeze_uniform=True,
        )

        raise AssertionError(
            "Expected ValueError when fixed_alpha and freeze_uniform are both used."
        )

    except ValueError:
        pass

    print("cafe_lightfm.py smoke test: all checks passed.")
