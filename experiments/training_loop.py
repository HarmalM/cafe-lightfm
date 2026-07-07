"""
training_loop.py

Phase 2, Step 4: full training loop for CAFE-LightFM and the LightFM
baseline, integrating Steps 1-3's components under a single reproducible
entry point.

Design:
    - Optimizer  : Adagrad lr=0.05, matching LightFM-style sparse updates.
    - Batching   : full positive-pair batch per stage/epoch.
    - Epochs     : 10 fixed epochs for smoke-test validation.
    - Reproducibility: seed=42, deterministic pair ordering via sorted().

PHASE 3, STEP 7 ADDITIONS (2026-07-06, confirmed by PI):
    `train_cafe_lightfm` gains four new optional parameters, all
    default-safe (i.e., all defaults reproduce Step 1-6 / Step 5
    behavior EXACTLY, byte-for-byte, seed=42):

        stage_idx_override : Optional[int] = None
            When set, every stage-loop iteration scores with this fixed
            stage_idx instead of the per-stage value from stage_to_idx.
            Supports the "CAFE-LightFM-noStage" ablation (proposal
            Section 5.6): the model still trains on S1/S2/S3 data
            separately (three per-epoch loss terms, exactly as before),
            but the SCA layer is always conditioned on the SAME stage
            index, so w_stage[stage_idx_override] is the only stage
            embedding that ever receives gradient -- disabling
            stage-conditional variation via a constant stage index (not
            a bias-freeze -- see sca_layer.py, no b_{s_j} exists in this
            implementation).

        stage_order : Optional[List[str]] = None
            Defaults to the module-level STAGE_ORDER (["S1","S2","S3"])
            when None. Overriding this lets the SAME loop iterate over a
            different stage-key partition -- e.g. ["S1","decision"] for
            the "CAFE-LightFM-2Stage" ablation -- without touching the
            loop body.

        stage_to_idx : Optional[Dict[str,int]] = None
            Defaults to the module-level STAGE_TO_IDX when None. Must be
            consistent with stage_order and with the n_stages the
            CAFELightFM instance was constructed with.

        freeze_uniform : bool = False
            Passed straight through to cafe_scorer(...) -> model(...).
            Supports the "CAFE-LightFM-noAttention" ablation. See
            sca_layer.py / cafe_lightfm.py / warp_loss.py for the full
            mechanism.

    `train_baseline` is UNCHANGED (the LightFM baseline has no stage or
    attention concept, so no ablation parameter applies to it).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.optim as optim

from data.interaction_matrix import InteractionMatrixBundle, build_interaction_matrix
from data.synthetic_generator_v2 import generate_synthetic_dataset_v2
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM
from models.cafe_lightfm.warp_loss import (
    baseline_scorer,
    build_user_positive_sets,
    cafe_scorer,
    warp_loss,
)

MASTER_SEED = 42
STAGE_ORDER = ["S1", "S2", "S3"]
STAGE_TO_IDX = {"S1": 0, "S2": 1, "S3": 2}


def _seed_all(seed: int) -> None:
    """
    Applies the project-wide seed.

    If the Phase 0 utility exists, use it. Otherwise, fall back to direct
    torch/random seeding so this file can still run in isolated smoke tests.
    """
    try:
        from utils.reproducibility import set_global_seed

        set_global_seed(seed)
    except ModuleNotFoundError:
        torch.manual_seed(seed)
        random.seed(seed)


@dataclass
class TrainingConfig:
    n_epochs: int = 10
    lr: float = 0.05
    max_sampled: int = 10
    margin: float = 1.0
    embedding_dim: int = 64
    n_stages: int = 3
    seed: int = MASTER_SEED


@dataclass
class EpochResult:
    epoch: int
    loss_by_stage: Dict[str, float]
    mean_loss: float
    elapsed_s: float


@dataclass
class TrainingLog:
    model_name: str
    config: TrainingConfig
    epochs: List[EpochResult] = field(default_factory=list)

    def append(self, result: EpochResult) -> None:
        self.epochs.append(result)

    def print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"Training log: {self.model_name}")
        print(
            f"{'Epoch':>6}  {'S1 loss':>10}  {'S2 loss':>10}  "
            f"{'S3 loss':>10}  {'Mean':>10}  {'Time(s)':>8}"
        )
        print(f"{'-' * 60}")

        for result in self.epochs:
            print(
                f"{result.epoch:>6}  "
                f"{result.loss_by_stage.get('S1', float('nan')):>10.4f}  "
                f"{result.loss_by_stage.get('S2', float('nan')):>10.4f}  "
                f"{result.loss_by_stage.get('S3', float('nan')):>10.4f}  "
                f"{result.mean_loss:>10.4f}  "
                f"{result.elapsed_s:>8.2f}"
            )

        print(f"{'=' * 60}\n")


def train_cafe_lightfm(
    model: CAFELightFM,
    bundle: InteractionMatrixBundle,
    config: TrainingConfig,
    stage_idx_override: Optional[int] = None,
    stage_order: Optional[List[str]] = None,
    stage_to_idx: Optional[Dict[str, int]] = None,
    freeze_uniform: bool = False,
) -> TrainingLog:
    """
    Trains CAFE-LightFM with stage-specific negative sampling.

    One epoch iterates over the stage keys in `stage_order` (default
    S1 -> S2 -> S3). For each stage, all positive pairs are consumed in
    deterministic sorted order.

    Parameters
    ----------
    stage_idx_override : Optional[int], default None
        Phase 3, Step 7 (noStage ablation). When not None, every
        stage-loop iteration is scored with this fixed stage_idx instead
        of stage_to_idx[stage_key]. Default None preserves Step 1-6
        behavior exactly.
    stage_order : Optional[List[str]], default None
        Phase 3, Step 7 (2Stage ablation support). Defaults to the
        module-level STAGE_ORDER when None.
    stage_to_idx : Optional[Dict[str, int]], default None
        Phase 3, Step 7 (2Stage ablation support). Defaults to the
        module-level STAGE_TO_IDX when None. Must stay consistent with
        stage_order and with the CAFELightFM instance's n_stages.
    freeze_uniform : bool, default False
        Phase 3, Step 7 (noAttention ablation). Passed straight through
        to cafe_scorer(...). Default False preserves Step 1-6 behavior
        exactly.

    Returns
    -------
    TrainingLog
    """
    _seed_all(config.seed)

    effective_stage_order = stage_order if stage_order is not None else STAGE_ORDER
    effective_stage_to_idx = stage_to_idx if stage_to_idx is not None else STAGE_TO_IDX

    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    log = TrainingLog(model_name="CAFE-LightFM", config=config)
    epoch_rng = random.Random(config.seed)

    for epoch in range(1, config.n_epochs + 1):
        start_time = time.time()
        loss_by_stage: Dict[str, float] = {}

        for stage_key in effective_stage_order:
            base_stage_idx = effective_stage_to_idx[stage_key]
            effective_stage_idx = (
                stage_idx_override if stage_idx_override is not None else base_stage_idx
            )
            pairs = sorted(bundle.positive_pairs_by_stage.get(stage_key, set()))

            if not pairs:
                loss_by_stage[stage_key] = 0.0
                continue

            user_idx = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
            item_idx = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)

            user_positive_sets = build_user_positive_sets(
                bundle.positive_pairs_by_stage[stage_key]
            )

            score_fn = cafe_scorer(
                model, bundle, stage_idx=effective_stage_idx, freeze_uniform=freeze_uniform
            )
            stage_rng = random.Random(epoch_rng.randint(0, 2**32))

            optimizer.zero_grad()

            loss = warp_loss(
                score_fn,
                user_idx,
                item_idx,
                user_positive_sets,
                bundle.n_items,
                max_sampled=config.max_sampled,
                margin=config.margin,
                rng=stage_rng,
            )

            loss.backward()
            optimizer.step()

            loss_by_stage[stage_key] = loss.item()

        mean_loss = sum(loss_by_stage.values()) / len(loss_by_stage)
        elapsed_s = time.time() - start_time

        result = EpochResult(
            epoch=epoch,
            loss_by_stage=loss_by_stage,
            mean_loss=mean_loss,
            elapsed_s=elapsed_s,
        )
        log.append(result)

        print(
            f"[CAFE-LightFM] epoch={epoch:>2}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in loss_by_stage.items())
            + f"  mean={mean_loss:.4f}  t={elapsed_s:.2f}s"
        )

    return log


def train_baseline(
    model: LightFMPyTorch,
    bundle: InteractionMatrixBundle,
    config: TrainingConfig,
) -> TrainingLog:
    """
    Trains the stage-blind LightFM baseline with global negative sampling.

    Unchanged from Phase 2, Step 4 -- the baseline has no stage or
    attention concept, so no Step 7 ablation parameter applies here.
    """
    _seed_all(config.seed)

    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    log = TrainingLog(model_name="LightFM-Baseline", config=config)

    user_positive_sets = build_user_positive_sets(bundle.positive_pairs)
    score_fn = baseline_scorer(model, bundle)
    epoch_rng = random.Random(config.seed)

    pairs = sorted(bundle.positive_pairs)

    user_idx = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    item_idx = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)

    for epoch in range(1, config.n_epochs + 1):
        start_time = time.time()
        epoch_rng_state = random.Random(epoch_rng.randint(0, 2**32))

        optimizer.zero_grad()

        loss = warp_loss(
            score_fn,
            user_idx,
            item_idx,
            user_positive_sets,
            bundle.n_items,
            max_sampled=config.max_sampled,
            margin=config.margin,
            rng=epoch_rng_state,
        )

        loss.backward()
        optimizer.step()

        elapsed_s = time.time() - start_time

        result = EpochResult(
            epoch=epoch,
            loss_by_stage={"global": loss.item()},
            mean_loss=loss.item(),
            elapsed_s=elapsed_s,
        )
        log.append(result)

        print(
            f"[LightFM-Baseline] epoch={epoch:>2}  "
            f"loss={loss.item():.4f}  "
            f"t={elapsed_s:.2f}s"
        )

    return log


if __name__ == "__main__":
    _seed_all(MASTER_SEED)

    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    config = TrainingConfig()

    print("=== Training CAFE-LightFM ===")
    _seed_all(MASTER_SEED)
    cafe_model = CAFELightFM(
        bundle.n_users,
        bundle.n_items,
        bundle.n_categories,
        bundle.n_programs,
        config.n_stages,
        config.embedding_dim,
    )
    cafe_log = train_cafe_lightfm(cafe_model, bundle, config)
    cafe_log.print_summary()

    print("=== Training LightFM Baseline ===")
    _seed_all(MASTER_SEED)
    baseline_model = LightFMPyTorch(
        bundle.n_users,
        bundle.n_items,
        bundle.n_categories,
        bundle.n_programs,
        config.embedding_dim,
    )
    baseline_log = train_baseline(baseline_model, bundle, config)
    baseline_log.print_summary()
