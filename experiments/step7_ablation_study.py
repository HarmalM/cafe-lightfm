"""
experiments/step7_ablation_study.py

Phase 3, Step 7: Ablation Study (proposal Section 5.6, secondary
objective 4) -- isolates the contribution of stage conditioning and the
attention mechanism to CAFE-LightFM's performance on the unsaturated
synthetic v3 dataset (data/synthetic_generator_v3.py, Phase 3 Step 1.5).

SCOPE (binding, restated per Steps 1-6 scope notes): this script is a
PIPELINE-VALIDATION run on synthetic data. No scientific claim of
CAFE-LightFM's or any ablation variant's architectural superiority is
made or should be drawn from its output. That claim is reserved
exclusively for the pilot (N=50) and full (N=300) Prolific Academic
studies (proposal Section 5.3).

Four variants (proposal Section 5.6, all decisions confirmed by PI,
2026-07-06/07):

    1. Full       -- the existing Step 5 checkpoint
                     (outputs/checkpoints/cafe_lightfm_v3_seed42.pt),
                     REUSED without retraining.
    2. noStage    -- CAFE-LightFM trained with
                     train_cafe_lightfm(..., stage_idx_override=0):
                     every stage's data is scored under a single fixed
                     stage_idx (0), so w_stage[1]/w_stage[2] never
                     receive gradient and remain zero-initialized.
                     Disables stage-conditional variation via a constant
                     stage index (NOT a bias-freeze -- this
                     implementation has no b_{s_j} term; see
                     models/cafe_lightfm/sca_layer.py).
    3. noAttention -- CAFE-LightFM trained with
                     train_cafe_lightfm(..., freeze_uniform=True): the
                     SCA layer's w_base/w_stage logit computation is
                     bypassed entirely and alpha is held at a constant
                     uniform 1/n_features, so w_base/w_stage never enter
                     the autograd graph and remain zero-initialized
                     throughout training.
    4. 2Stage     -- CAFE-LightFM trained on a bundle whose stage labels
                     are merged {S1: exploration} vs. {S2+S3: decision}
                     (PI-approved merge, 2026-07-07), with n_stages=2.
                     Evaluated DESCRIPTIVELY ONLY (see EVALUATION
                     PROTOCOL below) -- no paired significance test
                     against Full, because the 3-stage vs. 2-stage
                     cardinality mismatch would make such a comparison
                     structurally misleading. SW-NDCG is NOT computed
                     for this variant (its locked S1=0.20/S2=0.30/S3=0.50
                     weights assume exactly 3 stages).

EVALUATION PROTOCOL (PI-approved, 2026-07-07):
    - Full vs. noStage        : formal paired t-test + Bonferroni
                                 (both 3-stage, directly comparable).
    - Full vs. noAttention    : formal paired t-test + Bonferroni
                                 (both 3-stage, directly comparable).
    - 2Stage                  : descriptive-only NDCG@K / Precision@K
                                 table. Structural diagnostic, not a
                                 significance test.
    - All variants            : a descriptive NDCG@K / Precision@K table
                                 is printed first, before any formal
                                 testing, per PI's approved sequencing.

CRITICAL EVALUATION-TIME FIX (2026-07-07, confirmed necessary via direct
inspection of experiments/ndcg.py): noStage's evaluation MUST use the
same stage_idx_override=0 the model was TRAINED under. Evaluating
noStage via the ordinary per-stage STAGE_TO_IDX mapping (S1->0, S2->1,
S3->2) would score S2/S3 with the UNTRAINED w_stage[1]/w_stage[2],
producing a train/eval mismatch that contaminates the ablation with an
arbitrary untrained parameter rather than validly testing "no stage
conditioning." This script threads stage_idx_override=0 through every
noStage evaluation call (Phase 3, Step 7 additions to
experiments/ndcg.py and experiments/step4_integration.py's
per_user_metric). noAttention requires NO equivalent evaluation-time
fix: w_base/w_stage remain zero-initialized regardless of freeze_uniform
at eval time, and softmax(0) is uniform by construction.

ZERO RE-IMPLEMENTATION: this script performs NO re-implementation of
training, ranking, NDCG/Precision, or statistical-testing logic. It
orchestrates already-confirmed, already-tested components:
    - experiments.training_loop.{TrainingConfig, train_cafe_lightfm,
      _seed_all, MASTER_SEED, STAGE_ORDER, STAGE_TO_IDX}
    - experiments.step5_train_and_compare.build_v3_bundle
    - experiments.step4_integration.{per_user_metric, METRIC_SPECS,
      align_paired_arrays, print_comparison_table}
    - experiments.statistical_testing.{paired_t_test,
      summarize_paired_results}
    - experiments.ndcg.K_VALUES
    - models.cafe_lightfm.cafe_lightfm.CAFELightFM

NEW logic introduced by THIS script only (not duplicated elsewhere):
    - remap_bundle_to_2stage()             : bundle-level stage merge
    - compute_descriptive_table()          : mean-aggregation wrapper
      around per_user_metric (no ranking/metric re-implementation)
    - run_cafe_variant_paired_comparison() : CAFE-vs-CAFE paired testing
      driver (distinct from step4_integration's CAFE-vs-baseline
      driver -- see that module's Step 7 docstring note for why it
      cannot be reused directly for ablation-vs-ablation comparisons)

CONFIRMED (2026-07-08, via Colab traceback): InteractionMatrixBundle IS
a frozen dataclass (`dataclasses.FrozenInstanceError: cannot assign to
field 'positive_pairs_by_stage'`). The originally-flagged ASSUMPTION
(attribute reassignment on a `copy.copy()`) was therefore WRONG and has
been replaced below with `dataclasses.replace(bundle,
positive_pairs_by_stage=...)`, which is the correct construction path
for frozen dataclasses: it returns a NEW instance with the given
field(s) overridden and all other fields copied from the original --
no mutation of the original bundle occurs, and no `__init__` argument
order needs to be known or guessed.

Usage (Colab, from repo root):
    python -m experiments.step7_ablation_study
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import torch

from data.interaction_matrix import InteractionMatrixBundle
from models.cafe_lightfm.cafe_lightfm import CAFELightFM
from experiments.training_loop import (
    MASTER_SEED,
    STAGE_ORDER,
    STAGE_TO_IDX,
    TrainingConfig,
    _seed_all,
    train_cafe_lightfm,
)
from experiments.step5_train_and_compare import build_v3_bundle
from experiments.ndcg import K_VALUES
from experiments.step4_integration import (
    METRIC_SPECS,
    align_paired_arrays,
    per_user_metric,
    print_comparison_table,
)
from experiments.statistical_testing import PairedTestResult, paired_t_test, summarize_paired_results

CHECKPOINT_DIR = "outputs/checkpoints"
FULL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cafe_lightfm_v3_seed42.pt")  # Step 5, reused
NOSTAGE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cafe_lightfm_v3_seed42_nostage.pt")
NOATTENTION_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cafe_lightfm_v3_seed42_noattention.pt")
TWOSTAGE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cafe_lightfm_v3_seed42_2stage.pt")

TWO_STAGE_ORDER = ["S1", "decision"]
TWO_STAGE_TO_IDX = {"S1": 0, "decision": 1}


# --------------------------------------------------------------------------- #
# Bundle remapping for the 2Stage variant
# --------------------------------------------------------------------------- #

def remap_bundle_to_2stage(bundle: InteractionMatrixBundle) -> InteractionMatrixBundle:
    """
    Returns a bundle whose `positive_pairs_by_stage` is remapped from
    {S1, S2, S3} to {S1: exploration, decision: S2 union S3}, per the
    PI-approved 2Stage merge (2026-07-07): {S1} = exploration vs.
    {S2, S3} = decision.

    See module-level ASSUMPTION note regarding attribute-reassignment
    support on InteractionMatrixBundle -- requires Colab verification.

    Parameters
    ----------
    bundle : InteractionMatrixBundle (unmodified; a shallow copy is
        returned, so the original bundle's positive_pairs_by_stage is
        NOT mutated).

    Returns
    -------
    InteractionMatrixBundle with positive_pairs_by_stage =
        {"S1": <original S1 pairs>, "decision": <S2 union S3 pairs>}.
    All other fields (n_users, n_items, n_categories, n_programs,
    item_feature_idx_by_item, positive_pairs, ...) are copied unchanged
    from the original bundle via dataclasses.replace().
    """
    s1 = bundle.positive_pairs_by_stage.get("S1", set())
    s2 = bundle.positive_pairs_by_stage.get("S2", set())
    s3 = bundle.positive_pairs_by_stage.get("S3", set())

    return dataclasses.replace(
        bundle,
        positive_pairs_by_stage={
            "S1": set(s1),
            "decision": set(s2) | set(s3),
        },
    )


# --------------------------------------------------------------------------- #
# Training / loading each variant
# --------------------------------------------------------------------------- #

def load_full_model(bundle: InteractionMatrixBundle, config: TrainingConfig) -> CAFELightFM:
    """Loads the existing Step 5 Full checkpoint -- NO retraining."""
    model = CAFELightFM(
        bundle.n_users, bundle.n_items, bundle.n_categories, bundle.n_programs,
        config.n_stages, config.embedding_dim,
    )
    state_dict = torch.load(FULL_CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_nostage_variant(bundle: InteractionMatrixBundle, config: TrainingConfig) -> CAFELightFM:
    """Trains the noStage ablation: every stage scored under a fixed
    stage_idx=0 (see module docstring, variant 2)."""
    _seed_all(config.seed)
    model = CAFELightFM(
        bundle.n_users, bundle.n_items, bundle.n_categories, bundle.n_programs,
        config.n_stages, config.embedding_dim,
    )
    log = train_cafe_lightfm(model, bundle, config, stage_idx_override=0)
    log.print_summary()
    assert log.epochs[-1].mean_loss < log.epochs[0].mean_loss, (
        "noStage training loss did not decrease -- check optimizer/data "
        "before proceeding to evaluation."
    )
    return model


def train_noattention_variant(bundle: InteractionMatrixBundle, config: TrainingConfig) -> CAFELightFM:
    """Trains the noAttention ablation: alpha frozen uniform throughout
    (see module docstring, variant 3)."""
    _seed_all(config.seed)
    model = CAFELightFM(
        bundle.n_users, bundle.n_items, bundle.n_categories, bundle.n_programs,
        config.n_stages, config.embedding_dim,
    )
    log = train_cafe_lightfm(model, bundle, config, freeze_uniform=True)
    log.print_summary()
    assert log.epochs[-1].mean_loss < log.epochs[0].mean_loss, (
        "noAttention training loss did not decrease -- check optimizer/data "
        "before proceeding to evaluation."
    )
    return model


def train_2stage_variant(
    bundle_2stage: InteractionMatrixBundle, config: TrainingConfig
) -> CAFELightFM:
    """Trains the 2Stage ablation on the remapped bundle, n_stages=2
    (see module docstring, variant 4)."""
    _seed_all(config.seed)
    config_2stage = dataclasses.replace(config, n_stages=2)
    model = CAFELightFM(
        bundle_2stage.n_users, bundle_2stage.n_items,
        bundle_2stage.n_categories, bundle_2stage.n_programs,
        config_2stage.n_stages, config_2stage.embedding_dim,
    )
    log = train_cafe_lightfm(
        model, bundle_2stage, config_2stage,
        stage_order=TWO_STAGE_ORDER, stage_to_idx=TWO_STAGE_TO_IDX,
    )
    log.print_summary()
    assert log.epochs[-1].mean_loss < log.epochs[0].mean_loss, (
        "2Stage training loss did not decrease -- check optimizer/data "
        "before proceeding to evaluation."
    )
    return model


def save_all_ablation_checkpoints(
    nostage_model: CAFELightFM, noattention_model: CAFELightFM, twostage_model: CAFELightFM
) -> None:
    """Saves the three newly-trained variants' state_dicts. The Full
    checkpoint is the pre-existing Step 5 artifact and is not re-saved."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(nostage_model.state_dict(), NOSTAGE_CHECKPOINT_PATH)
    torch.save(noattention_model.state_dict(), NOATTENTION_CHECKPOINT_PATH)
    torch.save(twostage_model.state_dict(), TWOSTAGE_CHECKPOINT_PATH)
    for path in (NOSTAGE_CHECKPOINT_PATH, NOATTENTION_CHECKPOINT_PATH, TWOSTAGE_CHECKPOINT_PATH):
        assert os.path.isfile(path) and os.path.getsize(path) > 0, (
            f"Checkpoint save appears to have failed: {path}"
        )
    print(
        "Checkpoints saved:\n  "
        f"{NOSTAGE_CHECKPOINT_PATH}\n  {NOATTENTION_CHECKPOINT_PATH}\n  {TWOSTAGE_CHECKPOINT_PATH}"
    )


# --------------------------------------------------------------------------- #
# Evaluation: descriptive tables (all four variants)
# --------------------------------------------------------------------------- #

def compute_descriptive_table(
    model: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    stage_keys: Tuple[str, ...],
    k_values: Tuple[int, ...] = K_VALUES,
    stage_idx_override: Optional[int] = None,
    stage_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Mean NDCG@K / Precision@K per (stage, K), for a single CAFE-LightFM
    variant. Descriptive only -- no paired significance test. Reuses
    per_user_metric (Step 4) unchanged for all ranking/metric
    computation; this function only aggregates to a per-(stage,K,metric)
    mean.

    Parameters
    ----------
    stage_idx_override, stage_to_idx : see experiments/ndcg.py. Both
        default to None. MUST be set for noStage (stage_idx_override=0)
        and for 2Stage (stage_to_idx=TWO_STAGE_TO_IDX), per the
        module-level CRITICAL EVALUATION-TIME FIX note.

    Returns
    -------
    Dict[f"{metric}@{k}_{stage}" -> mean value (NaN if the stage/user
    set was empty)].
    """
    model.eval()
    table: Dict[str, float] = {}
    for stage_key in stage_keys:
        for k in k_values:
            for metric_name, metric_fn in METRIC_SPECS:
                scores = per_user_metric(
                    model, bundle, True, stage_key, k, metric_fn,
                    stage_idx_override=stage_idx_override, stage_to_idx=stage_to_idx,
                )
                label = f"{metric_name}@{k}_{stage_key}"
                table[label] = sum(scores.values()) / len(scores) if scores else float("nan")
    return table


def print_descriptive_table(name: str, table: Dict[str, float]) -> None:
    """Prints a formatted descriptive (non-paired) metric table."""
    print(f"\n--- {name} (descriptive mean NDCG@K / Precision@K) ---")
    for label, value in table.items():
        print(f"{label:<18}{value:>10.4f}")


# --------------------------------------------------------------------------- #
# Evaluation: formal paired comparisons (CAFE variant vs. CAFE variant)
# --------------------------------------------------------------------------- #

def run_cafe_variant_paired_comparison(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    bundle: InteractionMatrixBundle,
    k_values: Tuple[int, ...] = K_VALUES,
    stage_order: Tuple[str, ...] = STAGE_ORDER,
    stage_idx_override_a: Optional[int] = None,
    stage_to_idx_a: Optional[Dict[str, int]] = None,
    stage_idx_override_b: Optional[int] = None,
    stage_to_idx_b: Optional[Dict[str, int]] = None,
    n_comparisons_override: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Two-tailed paired t-test (Decision B) + Bonferroni correction
    (Decision C) between two CAFELightFM variants (e.g. Full vs.
    noStage, Full vs. noAttention), BOTH scored with is_cafe=True.

    Distinct from experiments.step4_integration's
    run_ndcg_precision_paired_comparison, which hardcodes is_cafe=False
    for its second model argument (correct for CAFE-vs-LightFM-baseline,
    not applicable to ablation-vs-ablation comparisons -- see that
    module's Step 7 docstring note). Reuses per_user_metric,
    align_paired_arrays, paired_t_test, summarize_paired_results
    UNCHANGED -- zero re-implementation of ranking, metric, or
    statistical-testing logic.

    Parameters
    ----------
    model_a, model_b : trained CAFELightFM instances.
    bundle : InteractionMatrixBundle shared by both models (for Full vs.
        noStage / Full vs. noAttention, this is the ordinary 3-stage
        v3 bundle -- NOT the 2Stage-remapped bundle).
    stage_idx_override_a/b, stage_to_idx_a/b : per-model Step 7
        evaluation-time overrides (see experiments/ndcg.py). Use
        stage_idx_override_b=0 when model_b is the noStage variant.
    n_comparisons_override : Optional[int]. None -> dynamic M (number of
        comparisons actually produced).

    Returns
    -------
    Dict keyed by f"{metric}@{k}_{stage}" -> per-comparison summary,
    identical schema to run_ndcg_precision_paired_comparison's output.

    Raises
    ------
    ValueError
        If no valid (stage, K, metric) comparison could be formed.
    """
    results: List[PairedTestResult] = []

    for stage_key in stage_order:
        for k in k_values:
            for metric_name, metric_fn in METRIC_SPECS:
                scores_a = per_user_metric(
                    model_a, bundle, True, stage_key, k, metric_fn,
                    stage_idx_override=stage_idx_override_a, stage_to_idx=stage_to_idx_a,
                )
                scores_b = per_user_metric(
                    model_b, bundle, True, stage_key, k, metric_fn,
                    stage_idx_override=stage_idx_override_b, stage_to_idx=stage_to_idx_b,
                )
                if not scores_a or not scores_b:
                    continue
                a_arr, b_arr, _ = align_paired_arrays(scores_a, scores_b)
                if a_arr.size < 2:
                    continue
                label = f"{metric_name}@{k}_{stage_key}"
                results.append(paired_t_test(a_arr, b_arr, metric_name=label))

    if not results:
        raise ValueError(
            "No valid paired comparisons were produced between the two "
            "CAFE-LightFM variants. Check that bundle.positive_pairs_by_stage "
            "contains at least one stage with >=2 users having positive "
            "interactions."
        )

    return summarize_paired_results(results, alpha=0.05, n_comparisons=n_comparisons_override)


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    print(
        "\n"
        "==================================================================\n"
        "Phase 3, Step 7 -- CAFE-LightFM Ablation Study, synthetic v3\n"
        "SCOPE: pipeline validation ONLY. No scientific superiority claim.\n"
        "Formal evidence reserved for Prolific pilot (N=50) / full (N=300)\n"
        "studies (proposal Section 5.3).\n"
        "==================================================================\n"
    )

    bundle = build_v3_bundle(seed=MASTER_SEED)
    config = TrainingConfig()  # locked defaults: n_epochs=10, lr=0.05, embedding_dim=64, n_stages=3

    print("=== Full: loading existing Step 5 checkpoint (no retraining) ===")
    full_model = load_full_model(bundle, config)

    print("=== Training noStage variant ===")
    nostage_model = train_nostage_variant(bundle, config)

    print("=== Training noAttention variant ===")
    noattention_model = train_noattention_variant(bundle, config)

    print("=== Training 2Stage variant ===")
    bundle_2stage = remap_bundle_to_2stage(bundle)
    twostage_model = train_2stage_variant(bundle_2stage, config)

    save_all_ablation_checkpoints(nostage_model, noattention_model, twostage_model)

    # --- Descriptive tables (all four variants), per approved sequencing ---
    print("\n=== Descriptive NDCG@K / Precision@K per variant ===")
    full_table = compute_descriptive_table(full_model, bundle, STAGE_ORDER)
    nostage_table = compute_descriptive_table(
        nostage_model, bundle, STAGE_ORDER, stage_idx_override=0
    )
    noattention_table = compute_descriptive_table(noattention_model, bundle, STAGE_ORDER)
    twostage_table = compute_descriptive_table(
        twostage_model, bundle_2stage, tuple(TWO_STAGE_ORDER), stage_to_idx=TWO_STAGE_TO_IDX
    )
    print_descriptive_table("Full", full_table)
    print_descriptive_table("noStage", nostage_table)
    print_descriptive_table("noAttention", noattention_table)
    print_descriptive_table("2Stage (structural diagnostic only)", twostage_table)

    # --- Formal paired comparisons: 3-stage variants only, per PI decision ---
    print("\n=== Full vs. noStage (paired t-test + Bonferroni) ===")
    summary_nostage = run_cafe_variant_paired_comparison(
        full_model, nostage_model, bundle, stage_idx_override_b=0,
    )
    print_comparison_table(summary_nostage)

    print("\n=== Full vs. noAttention (paired t-test + Bonferroni) ===")
    summary_noattention = run_cafe_variant_paired_comparison(
        full_model, noattention_model, bundle,
    )
    print_comparison_table(summary_noattention)

    print(
        "SCOPE REMINDER: all Step 7 results are a synthetic-v3 "
        "pipeline-validation check only -- no scientific claim of any "
        "variant's superiority is made or implied. The 2Stage variant is "
        "reported descriptively only; no paired significance test was run "
        "against Full (3-stage vs. 2-stage cardinality mismatch), per PI "
        "decision (2026-07-07).\n"
    )


if __name__ == "__main__":
    main()
