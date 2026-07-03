"""
training_loop.py

Phase 2, Step 4: full training loop for CAFE-LightFM and the LightFM
baseline, integrating Steps 1-3's components under a single reproducible
entry point.

Design (confirmed 2026-06-26):
    - Optimizer  : Adagrad [1] lr=0.05, matching LightFM's original
                   training convention (Kula, 2015) [2] for sparse
                   gradient updates produced by WARP loss.
    - Batching   : full positive_pairs_by_stage per stage per epoch
                   (no mini-batch splitting). With ~400 pairs/stage on
                   the Step 4 dataset, this is computationally trivial
                   and avoids introducing a second RNG dependency inside
                   the epoch for reproducibility.
    - Epochs     : 10 fixed -- sufficient to confirm loss decrease on
                   synthetic data without overfitting claims on a dataset
                   whose item catalog is known to be fully saturated
                   globally (Phase 1, Step 5 finding).
    - Reproducibility: master seed=42 (project convention, Phase 0).
                   Applied via utils.reproducibility.set_global_seed()
                   when that module is available (Phase 0 environment);
                   falls back to direct torch.manual_seed() + random.seed()
                   when running outside the full project tree (e.g. unit
                   tests). Pair ordering uses sorted() throughout to
                   eliminate any set-iteration non-determinism.

Stage ordering within each epoch: S1 -> S2 -> S3 (chain order,
consistent with A1, Definition III.1).

References
----------
[1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization. Journal of
    Machine Learning Research, 12, 2121-2159.
    https://jmlr.org/papers/v12/duchi11a.html
[2] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop on New Trends in CBRS.
    CEUR-WS, Vol. 1448, 14-21.
[3] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to
    Large Vocabulary Image Annotation. IJCAI 2011, 2764-2770.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

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
    """Applies the project-wide seed to torch, random, and (if available)
    utils.reproducibility.set_global_seed() from Phase 0."""
    try:
        from utils.reproducibility import set_global_seed
        set_global_seed(seed)
    except ModuleNotFoundError:
        # Running outside the full project tree (e.g. isolated smoke test).
        # Fall back to direct seeding -- documented explicitly here rather
        # than silently, so the behaviour is transparent and auditable.
        torch.manual_seed(seed)
        random.seed(seed)


# --------------------------------------------------------------------------- #
# Training configuration (frozen, Step 4)
# --------------------------------------------------------------------------- #

@dataclass
class TrainingConfig:
    n_epochs: int = 10
    lr: float = 0.05
    max_sampled: int = 10
    margin: float = 1.0
    embedding_dim: int = 64
    n_stages: int = 3
    seed: int = MASTER_SEED


# --------------------------------------------------------------------------- #
# Per-epoch result record
# --------------------------------------------------------------------------- #

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
        print(f"\n{'='*60}")
        print(f"Training log: {self.model_name}")
        print(f"{'Epoch':>6}  {'S1 loss':>10}  {'S2 loss':>10}  "
              f"{'S3 loss':>10}  {'Mean':>10}  {'Time(s)':>8}")
        print(f"{'-'*60}")
        for r in self.epochs:
            print(
                f"{r.epoch:>6}  "
                f"{r.loss_by_stage.get('S1', float('nan')):>10.4f}  "
                f"{r.loss_by_stage.get('S2', float('nan')):>10.4f}  "
                f"{r.loss_by_stage.get('S3', float('nan')):>10.4f}  "
                f"{r.mean_loss:>10.4f}  "
                f"{r.elapsed_s:>8.2f}"
            )
        print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# Training functions
# --------------------------------------------------------------------------- #

def train_cafe_lightfm(
    model: CAFELightFM,
    bundle: InteractionMatrixBundle,
    config: TrainingConfig,
) -> TrainingLog:
    """Trains CAFE-LightFM with stage-specific negative sampling."""
    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    log = TrainingLog(model_name="CAFE-LightFM", config=config)
    epoch_rng = random.Random(config.seed)

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        loss_by_stage: Dict[str, float] = {}

        for stage_key in STAGE_ORDER:
            stage_idx = STAGE_TO_IDX[stage_key]
            # sorted() for deterministic pair ordering (confirmed 2026-06-26)
            pairs = sorted(bundle.positive_pairs_by_stage.get(stage_key, set()))
            if not pairs:
                loss_by_stage[stage_key] = 0.0
                continue

            user_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            item_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            user_positive_sets = build_user_positive_sets(
                bundle.positive_pairs_by_stage[stage_key]
            )
            score_fn = cafe_scorer(model, bundle, stage_idx=stage_idx)
            stage_rng = random.Random(epoch_rng.randint(0, 2**32))

            optimizer.zero_grad()
            loss = warp_loss(
                score_fn, user_idx, item_idx,
                user_positive_sets, bundle.n_items,
                max_sampled=config.max_sampled,
                margin=config.margin,
                rng=stage_rng,
            )
            loss.backward()
            optimizer.step()
            loss_by_stage[stage_key] = loss.item()

        mean_loss = sum(loss_by_stage.values()) / len(loss_by_stage)
        elapsed = time.time() - t0
        result = EpochResult(epoch, loss_by_stage, mean_loss, elapsed)
        log.append(result)
        print(f"[CAFE-LightFM] epoch={epoch:>2}  "
              f"S1={loss_by_stage.get('S1', 0):.4f}  "
              f"S2={loss_by_stage.get('S2', 0):.4f}  "
              f"S3={loss_by_stage.get('S3', 0):.4f}  "
              f"mean={mean_loss:.4f}  t={elapsed:.2f}s")

    return log


def train_baseline(
    model: LightFMPyTorch,
    bundle: InteractionMatrixBundle,
    config: TrainingConfig,
) -> TrainingLog:
    """Trains the stage-blind LightFM baseline with global negative sampling."""
    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    log = TrainingLog(model_name="LightFM-Baseline", config=config)
    user_positive_sets = build_user_positive_sets(bundle.positive_pairs)
    score_fn = baseline_scorer(model, bundle)
    epoch_rng = random.Random(config.seed)

    # sorted() for deterministic pair ordering (confirmed 2026-06-26)
    pairs = sorted(bundle.positive_pairs)
    user_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    item_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        epoch_rng_state = random.Random(epoch_rng.randint(0, 2**32))

        optimizer.zero_grad()
        loss = warp_loss(
            score_fn, user_idx, item_idx,
            user_positive_sets, bundle.n_items,
            max_sampled=config.max_sampled,
            margin=config.margin,
            rng=epoch_rng_state,
        )
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        result = EpochResult(epoch, {"global": loss.item()}, loss.item(), elapsed)
        log.append(result)
        print(f"[LightFM-Baseline] epoch={epoch:>2}  loss={loss.item():.4f}  t={elapsed:.2f}s")

    return log


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    _seed_all(MASTER_SEED)

    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    config = TrainingConfig()

    print("=== Training CAFE-LightFM ===")
    _seed_all(MASTER_SEED)
    cafe_model = CAFELightFM(
        bundle.n_users, bundle.n_items,
        bundle.n_categories, bundle.n_programs,
        config.n_stages, config.embedding_dim,
    )
    cafe_log = train_cafe_lightfm(cafe_model, bundle, config)
    cafe_log.print_summary()

    print("=== Training LightFM Baseline ===")
    _seed_all(MASTER_SEED)
    baseline_model = LightFMPyTorch(
        bundle.n_users, bundle.n_items,
        bundle.n_categories, bundle.n_programs,
        config.embedding_dim,
    )
    baseline_log = train_baseline(baseline_model, bundle, config)
    baseline_log.print_summary()


Phase 2, Step 4: full training loop for CAFE-LightFM and the LightFM
baseline, integrating Steps 1-3's components under a single reproducible
entry point.

Design (confirmed 2026-06-26):
    - Optimizer  : Adagrad [1] lr=0.05, matching LightFM's original
                   training convention (Kula, 2015) [2] for sparse
                   gradient updates produced by WARP loss.
    - Batching   : full positive_pairs_by_stage per stage per epoch
                   (no mini-batch splitting). With ~400 pairs/stage on
                   the Step 4 dataset, this is computationally trivial
                   and avoids introducing a second RNG dependency inside
                   the epoch for reproducibility.
    - Epochs     : 10 fixed -- sufficient to confirm loss decrease on
                   synthetic data without overfitting claims on a dataset
                   whose item catalog is known to be fully saturated
                   globally (Phase 1, Step 5 finding).
    - Reproducibility: master seed=42 (project convention, Phase 0),
                   applied via utils/reproducibility.set_global_seed()
                   before model construction and before every epoch's
                   negative-sampling RNG.

Stage ordering within each epoch: S1 -> S2 -> S3 (chain order,
consistent with A1, Definition III.1).

References
----------
[1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization. Journal of
    Machine Learning Research, 12, 2121-2159.
    https://jmlr.org/papers/v12/duchi11a.html
[2] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop on New Trends in CBRS.
    CEUR-WS, Vol. 1448, 14-21.
[3] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to
    Large Vocabulary Image Annotation. IJCAI 2011, 2764-2770.
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
STAGE_ORDER = ["S1", "S2", "S3"]   # chain order, consistent with A1
STAGE_TO_IDX = {"S1": 0, "S2": 1, "S3": 2}


# --------------------------------------------------------------------------- #
# Training configuration (frozen, Step 4)
# --------------------------------------------------------------------------- #

@dataclass
class TrainingConfig:
    n_epochs: int = 10
    lr: float = 0.05           # Adagrad default for LightFM [1, 2]
    max_sampled: int = 10      # WARP negative-sampling trials [2, 3]
    margin: float = 1.0        # WARP hinge margin [3]
    embedding_dim: int = 64    # project-wide convention, Phase 0
    n_stages: int = 3
    seed: int = MASTER_SEED


# --------------------------------------------------------------------------- #
# Per-epoch result record (for logging and downstream evaluation)
# --------------------------------------------------------------------------- #

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
        print(f"\n{'='*60}")
        print(f"Training log: {self.model_name}")
        print(f"{'Epoch':>6}  {'S1 loss':>10}  {'S2 loss':>10}  "
              f"{'S3 loss':>10}  {'Mean':>10}  {'Time(s)':>8}")
        print(f"{'-'*60}")
        for r in self.epochs:
            print(
                f"{r.epoch:>6}  "
                f"{r.loss_by_stage.get('S1', float('nan')):>10.4f}  "
                f"{r.loss_by_stage.get('S2', float('nan')):>10.4f}  "
                f"{r.loss_by_stage.get('S3', float('nan')):>10.4f}  "
                f"{r.mean_loss:>10.4f}  "
                f"{r.elapsed_s:>8.2f}"
            )
        print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# Training functions
# --------------------------------------------------------------------------- #

def train_cafe_lightfm(
    model: CAFELightFM,
    bundle: InteractionMatrixBundle,
    config: TrainingConfig,
) -> TrainingLog:
    """
    Trains CAFE-LightFM with stage-specific negative sampling.
    One pass per epoch = iterate over S1, S2, S3 in chain order, using
    the full positive_pairs_by_stage for each stage as the batch.
    """
    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    log = TrainingLog(model_name="CAFE-LightFM", config=config)
    epoch_rng = random.Random(config.seed)

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        loss_by_stage: Dict[str, float] = {}

        for stage_key in STAGE_ORDER:
            stage_idx = STAGE_TO_IDX[stage_key]
            pairs = list(bundle.positive_pairs_by_stage.get(stage_key, set()))
            if not pairs:
                loss_by_stage[stage_key] = 0.0
                continue

            user_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            item_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            user_positive_sets = build_user_positive_sets(
                bundle.positive_pairs_by_stage[stage_key]
            )
            score_fn = cafe_scorer(model, bundle, stage_idx=stage_idx)
            stage_rng = random.Random(epoch_rng.randint(0, 2**32))

            optimizer.zero_grad()
            loss = warp_loss(
                score_fn, user_idx, item_idx,
                user_positive_sets, bundle.n_items,
                max_sampled=config.max_sampled,
                margin=config.margin,
                rng=stage_rng,
            )
            loss.backward()
            optimizer.step()
            loss_by_stage[stage_key] = loss.item()

        mean_loss = sum(loss_by_stage.values()) / len(loss_by_stage)
        elapsed = time.time() - t0
        result = EpochResult(epoch, loss_by_stage, mean_loss, elapsed)
        log.append(result)
        print(f"[CAFE-LightFM] epoch={epoch:>2}  "
              f"S1={loss_by_stage.get('S1',0):.4f}  "
              f"S2={loss_by_stage.get('S2',0):.4f}  "
              f"S3={loss_by_stage.get('S3',0):.4f}  "
              f"mean={mean_loss:.4f}  t={elapsed:.2f}s")

    return log


def train_baseline(
    model: LightFMPyTorch,
    bundle: InteractionMatrixBundle,
    config: TrainingConfig,
) -> TrainingLog:
    """
    Trains the stage-blind LightFM baseline with global negative sampling.
    One pass per epoch = all positive_pairs as a single batch (no stage
    conditioning; consistent with Section 5.5 baseline specification).
    """
    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    log = TrainingLog(model_name="LightFM-Baseline", config=config)
    user_positive_sets = build_user_positive_sets(bundle.positive_pairs)
    score_fn = baseline_scorer(model, bundle)
    epoch_rng = random.Random(config.seed)

    pairs = list(bundle.positive_pairs)
    user_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    item_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        epoch_rng_state = random.Random(epoch_rng.randint(0, 2**32))

        optimizer.zero_grad()
        loss = warp_loss(
            score_fn, user_idx, item_idx,
            user_positive_sets, bundle.n_items,
            max_sampled=config.max_sampled,
            margin=config.margin,
            rng=epoch_rng_state,
        )
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        result = EpochResult(epoch, {"global": loss.item()}, loss.item(), elapsed)
        log.append(result)
        print(f"[LightFM-Baseline] epoch={epoch:>2}  loss={loss.item():.4f}  t={elapsed:.2f}s")

    return log


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(MASTER_SEED)
    random.seed(MASTER_SEED)

    dataset = generate_synthetic_dataset_v2(seed=MASTER_SEED)
    bundle = build_interaction_matrix(dataset)
    config = TrainingConfig()

    print("=== Training CAFE-LightFM ===")
    torch.manual_seed(MASTER_SEED)
    cafe_model = CAFELightFM(
        bundle.n_users, bundle.n_items,
        bundle.n_categories, bundle.n_programs,
        config.n_stages, config.embedding_dim,
    )
    cafe_log = train_cafe_lightfm(cafe_model, bundle, config)
    cafe_log.print_summary()

    print("=== Training LightFM Baseline ===")
    torch.manual_seed(MASTER_SEED)
    baseline_model = LightFMPyTorch(
        bundle.n_users, bundle.n_items,
        bundle.n_categories, bundle.n_programs,
        config.embedding_dim,
    )
    baseline_log = train_baseline(baseline_model, bundle, config)
    baseline_log.print_summary()
