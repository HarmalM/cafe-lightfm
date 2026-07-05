"""
experiments/step5_train_and_compare.py

Phase 3, Step 5: deterministic train-then-compare driver for
CAFE-LightFM vs. the LightFM baseline on the unsaturated synthetic v3
dataset (data/synthetic_generator_v3.py, Phase 3 Step 1.5).

SCOPE (binding, restated per Steps 1-4 scope notes): this script is a
PIPELINE-VALIDATION run on synthetic data. No scientific claim of
CAFE-LightFM's architectural superiority is made or should be drawn from
its output. That claim is reserved exclusively for the pilot (N=50) and
full (N=300) Prolific Academic studies (proposal Section 5.3).

Zero re-implementation: this script performs NO re-implementation of
training, ranking, NDCG/Precision, or statistical-testing logic. It is a
thin orchestration layer over already-confirmed, already-tested
components:
    - data.synthetic_generator_v3.generate_synthetic_dataset_v3   [ASSUMED
      signature: generate_synthetic_dataset_v3(seed=MASTER_SEED),
      mirroring generate_synthetic_dataset_v2's single-argument call in
      experiments/training_loop.py's __main__ block -- confirmed by PI,
      2026-07-05]
    - data.interaction_matrix.build_interaction_matrix
    - models.cafe_lightfm.cafe_lightfm.CAFELightFM
    - models.baselines.lightfm_pytorch.LightFMPyTorch
    - experiments.training_loop.{TrainingConfig, train_cafe_lightfm,
      train_baseline}                                              [Phase 2
      Step 4, confirmed source pasted 2026-07-05]
    - experiments.step4_integration.run_ndcg_precision_paired_comparison,
      print_comparison_table                                       [Phase 3
      Step 4, 27/27 tests confirmed in Colab, 2026-07-05]

DECISIONS CONFIRMED (2026-07-05, PI):
    1. v3 dataset call: generate_synthetic_dataset_v3(seed=MASTER_SEED),
       single-argument, matching the v2 call pattern in
       experiments/training_loop.py.
    2. Batch size: experiments/training_loop.py is used AS-IS
       (full-batch WARP loss per stage/epoch). The locked project
       config value batch_size=256 is NOT enforced in this script --
       it is treated as informational/future-facing (binding for a
       later, larger-scale training stage), not applicable to this
       20-user / 30-item synthetic validation run. training_loop.py is
       NOT modified.
    3. Checkpoints saved to:
           outputs/checkpoints/cafe_lightfm_v3_seed42.pt
           outputs/checkpoints/lightfm_baseline_v3_seed42.pt
       via torch.save(model.state_dict(), path) -- state_dict only
       (not the full pickled module), the standard PyTorch-recommended
       serialization form [1].

Comparison family size (dynamic M, Decision C from Step 4): 3 stages x
3 K-values x 2 metrics (NDCG, Precision) = 18 comparisons on this
single-baseline v3 run. NOT the locked full-study M=63 (which spans the
full 7-baseline suite, proposal Section 5.4).

References
----------
[1] PyTorch Contributors. (2024). Saving and Loading Models. PyTorch
    documentation. https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    [Recommends state_dict serialization over pickling the full module
    for portability and forward-compatibility.]

Usage (Colab, from repo root):
    python -m experiments.step5_train_and_compare
"""

from __future__ import annotations

import os
from typing import Dict

import torch

from data.interaction_matrix import InteractionMatrixBundle, build_interaction_matrix
from data.synthetic_generator_v3 import generate_synthetic_dataset_v3
from models.baselines.lightfm_pytorch import LightFMPyTorch
from models.cafe_lightfm.cafe_lightfm import CAFELightFM
from experiments.training_loop import (
    MASTER_SEED,
    TrainingConfig,
    _seed_all,
    train_baseline,
    train_cafe_lightfm,
)
from experiments.step4_integration import (
    print_comparison_table,
    run_ndcg_precision_paired_comparison,
)

CHECKPOINT_DIR = "outputs/checkpoints"
CAFE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cafe_lightfm_v3_seed42.pt")
BASELINE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "lightfm_baseline_v3_seed42.pt")


def build_v3_bundle(seed: int = MASTER_SEED) -> InteractionMatrixBundle:
    """
    Generates the unsaturated synthetic v3 dataset and builds its
    InteractionMatrixBundle.

    ASSUMPTION (confirmed by PI, 2026-07-05): `generate_synthetic_dataset_v3`
    accepts a single `seed` keyword argument, matching the call pattern
    for `generate_synthetic_dataset_v2` in experiments/training_loop.py's
    `__main__` block. If this call fails with a TypeError, the v3
    generator's actual signature differs and must be re-confirmed before
    proceeding -- this script does not guess further parameters.

    Parameters
    ----------
    seed : int, default MASTER_SEED (42)

    Returns
    -------
    InteractionMatrixBundle
    """
    dataset = generate_synthetic_dataset_v3(seed=seed)
    return build_interaction_matrix(dataset)


def train_both_models(
    bundle: InteractionMatrixBundle, config: TrainingConfig
) -> Dict[str, torch.nn.Module]:
    """
    Deterministically trains CAFE-LightFM and the LightFM baseline on
    the given bundle, reusing experiments.training_loop unmodified
    (Decision 2, 2026-07-05: full-batch WARP loss per stage/epoch;
    batch_size=256 not enforced at this validation scale).

    Returns
    -------
    Dict with keys "cafe" -> trained CAFELightFM, "baseline" -> trained
    LightFMPyTorch.
    """
    print("=== Phase 3, Step 5: Training CAFE-LightFM (v3, seed=42) ===")
    _seed_all(config.seed)
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

    print("=== Phase 3, Step 5: Training LightFM Baseline (v3, seed=42) ===")
    _seed_all(config.seed)
    baseline_model = LightFMPyTorch(
        bundle.n_users,
        bundle.n_items,
        bundle.n_categories,
        bundle.n_programs,
        config.embedding_dim,
    )
    baseline_log = train_baseline(baseline_model, bundle, config)
    baseline_log.print_summary()

    # --- Sanity checks (substitute for a pre-run smoke test; this script
    # cannot be unit-tested offline since it depends on the real
    # CAFELightFM/LightFMPyTorch/warp_loss implementations) ---
    assert cafe_log.epochs[-1].mean_loss < cafe_log.epochs[0].mean_loss, (
        "CAFE-LightFM training loss did not decrease -- check optimizer "
        "or data before proceeding to evaluation."
    )
    assert baseline_log.epochs[-1].mean_loss < baseline_log.epochs[0].mean_loss, (
        "Baseline training loss did not decrease -- this is the exact "
        "saturation failure mode found in Phase 3 Step 1 on v2; verify "
        "the v3 dataset is actually being used, not v2."
    )

    return {"cafe": cafe_model, "baseline": baseline_model}


def save_checkpoints(models: Dict[str, torch.nn.Module]) -> None:
    """
    Saves both trained models' state_dicts to the paths confirmed by the
    PI (2026-07-05), creating outputs/checkpoints/ if needed.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(models["cafe"].state_dict(), CAFE_CHECKPOINT_PATH)
    torch.save(models["baseline"].state_dict(), BASELINE_CHECKPOINT_PATH)

    # Sanity check: files must actually exist and be non-empty after save.
    for path in (CAFE_CHECKPOINT_PATH, BASELINE_CHECKPOINT_PATH):
        assert os.path.isfile(path) and os.path.getsize(path) > 0, (
            f"Checkpoint save appears to have failed: {path}"
        )
    print(f"Checkpoints saved:\n  {CAFE_CHECKPOINT_PATH}\n  {BASELINE_CHECKPOINT_PATH}")


def main() -> None:
    print(
        "\n"
        "==================================================================\n"
        "Phase 3, Step 5 -- CAFE-LightFM vs. LightFM Baseline, synthetic v3\n"
        "SCOPE: pipeline validation ONLY. No scientific superiority claim.\n"
        "Formal evidence reserved for Prolific pilot (N=50) / full (N=300)\n"
        "studies (proposal Section 5.3).\n"
        "==================================================================\n"
    )

    bundle = build_v3_bundle(seed=MASTER_SEED)
    config = TrainingConfig()  # locked defaults: n_epochs=10, lr=0.05, embedding_dim=64

    models = train_both_models(bundle, config)
    save_checkpoints(models)

    print("\n=== Phase 3, Step 5: Paired NDCG@K / Precision@K Comparison ===")
    summary = run_ndcg_precision_paired_comparison(
        cafe_model=models["cafe"],
        baseline_model=models["baseline"],
        bundle=bundle,
    )

    # Sanity check: dynamic M for this single-baseline scaffold should be
    # <= 18 (3 stages x 3 K-values x 2 metrics); fewer only if some
    # (stage, K) slice was skipped (e.g. <2 users), which is a valid but
    # noteworthy outcome to flag rather than silently accept.
    assert len(summary) <= 18, (
        f"Expected at most 18 comparisons (3 stages x 3 K x 2 metrics), "
        f"got {len(summary)} -- investigate before reporting."
    )
    if len(summary) < 18:
        print(
            f"NOTE: only {len(summary)}/18 comparisons were produced -- "
            "one or more (stage, K) slices were skipped, likely due to "
            "<2 users with positives in that stage. Not an error, but "
            "should be reported alongside the results."
        )

    print_comparison_table(summary)

    print(
        "SCOPE REMINDER: the results above are a synthetic v3 "
        "pipeline-validation check only -- no scientific claim of "
        "CAFE-LightFM's superiority is made or implied.\n"
    )


if __name__ == "__main__":
    main()
