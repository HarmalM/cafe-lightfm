
# Phase 3, Step 1.5 — Unsaturated Synthetic Validation Dataset

## Status: COMPLETE

## Motivation

Phase 3 Step 1 (stage-stratified NDCG@K) was implemented and passed all
smoke tests on the existing `synthetic_generator_v2.py` (Step 4) dataset.
However, evaluation on that dataset revealed a validity problem, not an
implementation bug:

- Every user interacted with all 10 items in the Step 4 catalog
  (global saturation), because each event samples `item_id` uniformly
  from the *full* item pool with no per-user restriction.
- Under the frozen WARP-loss global exclusion scope (Phase 2, Step 3),
  this left the LightFM baseline with **zero valid negative items**.
- Consequence: baseline training loss was identically `0.0` across all
  10 training epochs. The baseline's parameters never moved from
  initialization.
- The resulting NDCG@K values for the baseline (0.82–0.96) reflect
  **random-initialization scores**, not learned preferences.

**Risk if uncorrected:** any Phase 3 Step 2–4 comparison (SW-NDCG,
Precision@K, paired t-test with Bonferroni correction) run on the Step 4
dataset would compare a trained CAFE-LightFM against an untrained
baseline. Statistical significance, if found, would be an artifact of
dataset saturation rather than evidence of the architectural
contribution.

## Fix

`synthetic_generator_v3.py` preserves the Step 4 generator's stage
semantics (S1/S2/S3), leakage model (80% nominal / 20% neighboring-stage,
chain topology), and continuous-signal parameters, changing only the
item-sampling universe:

| | v2 (Step 4) | v3 (Step 1.5) |
|---|---|---|
| Catalog size | 10 items | 30 items |
| Item sampling scope | full catalog, every event | fixed ~30% subset per (user, stage), drawn once per session |
| Max items touched by one user (global) | 10 (saturated) | ≤ 27 of 30 (deterministic upper bound: 3 × 9) |
| Min global negatives per user | 0 (empirically) | ≥ 3 (guaranteed by construction) |
| Min stage-specific negatives per (user, stage) | 0 (empirically) | ≥ 21 (guaranteed lower bound: 30 − 9) |

The negative-count guarantees are **deterministic lower bounds**, not
merely probabilistically likely: they follow directly from the fixed
subset size (round(30 × 0.30) = 9) and the fixed number of stages (3),
independent of the random seed. They are lower bounds rather than exact
counts because individual events sample WITH replacement from each
9-item subset, so a session may touch fewer than 9 distinct items —
meaning realized negative counts can exceed 3 (global) and 21
(stage-specific), never fewer.

## Validation

`tests/test_synthetic_generator_v3_smoke.py` — 10/10 checks:

1. Reproducibility under seed = 42
2. Records conform to the existing `InteractionRecord` schema
3. Stages present are exactly {S1, S2, S3}
4. Empirical leak rate ≈ 0.20
4b. (supplementary) Full 30-item catalog is actually represented in the
    generated dataset and in `bundle.n_items` — not merely assumed from
    `N_ITEMS_V3`, since `build_interaction_matrix()` indexes only
    observed items
5. No user is globally saturated
6. Every user has ≥ 1 valid global negative
7. Every (user, stage) pair has ≥ 1 valid stage-specific negative
8. `positive_pairs` does not cover the full user–item catalog
9. `positive_pairs_by_stage` is non-empty for all three stages
10. LightFM baseline training loss is **not** identically 0.0 across all
    10 epochs when trained via `experiments/training_loop.train_baseline`
    on this dataset (empirical confirmation the fix resolves the Step 1
    finding, not just the static negative-count guarantees)

## Scope boundary (binding)

This dataset is **synthetic and validation-only**:

- It is used exclusively to validate that the Phase 3 evaluation
  pipeline (NDCG@K, and subsequently SW-NDCG, Precision@K, paired
  t-testing) behaves correctly under a fair negative-sampling condition
  for *both* models.
- It does **not** replace real Prolific Academic participant data.
- It does **not** support any final scientific claim about CAFE-LightFM's
  performance relative to baselines. Such claims are reserved for the
  pilot (N=50) and full (N=300) Prolific studies per the proposal
  (Section 5.3).
- Any figure, table, or reported effect size computed on this dataset in
  Phase 3 Steps 2–4 must be labeled "unsaturated synthetic validation
  dataset — pipeline check only" and excluded from the dissertation's
  empirical-contribution chapters.

## Next step

Proceed to Phase 3 Step 2 (SW-NDCG, weights S1=0.20/S2=0.30/S3=0.50) using
`synthetic_generator_v3.py` as the validation dataset, per the confirmed
Step 1.5 → Step 2 sequencing.
