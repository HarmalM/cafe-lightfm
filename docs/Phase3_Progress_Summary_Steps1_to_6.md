# CAFE-LightFM — Phase 3 Progress Summary (Steps 1–6)

**DRAFT FOR REVIEW — NOT YET COMMITTED TO REPOSITORY**

**Project:** CAFE-LightFM (Paper I of III, PhD dissertation: *Stage-Aware
Recommendation Systems: Modeling, Detecting, and Generalizing Preference
Evolution Across Multi-Stage Decision Contexts*)
**Repository:** `HarmalM/cafe-lightfm`
**Status at time of writing:** Phase 3, Step 6 COMPLETE. Not proceeding
to Step 7 pending explicit confirmation.

---

## 1. Phase 3, Step 1 — Stage-Wise NDCG@K

**Files:** `experiments/ndcg.py`, `tests/test_ndcg_smoke.py`

Implements stage-stratified NDCG@K (Järvelin & Kekäläinen, 2002 [1]) as
the primary IR metric, K ∈ {5, 10, 20} (proposal Section 5.4). Relevance
was derived directly from `positive_pairs_by_stage[stage]` — in-training
relevance, validating metric correctness only, not generalization. All 7
smoke tests passed (perfect/worst ranking, zero-division guard, model
integration, boundedness, stage coverage).

**Status: ✅ COMPLETE.**

---

## 2. Key Finding — Baseline Saturation (motivated Step 1.5)

Evaluation on `synthetic_generator_v2.py` (10-item catalog) exposed full
global item saturation per user, leaving the LightFM baseline with zero
valid negatives under the WARP-loss global exclusion scope. Baseline
training loss was identically 0.0 across all 10 epochs — an artifact of
the dataset, not a fair architectural result. Documented transparently
per project policy; motivated Step 1.5.

---

## 3. Phase 3, Step 1.5 — Unsaturated Synthetic Validation Dataset (v3)

**Files:** `data/synthetic_generator_v3.py`,
`tests/test_synthetic_generator_v3_smoke.py`

20 users, 30-item catalog, ~30% per-(user, stage) item subset (seed=42),
guaranteeing ≥ 3 global negatives/user and ≥ 21 stage-specific
negatives/(user, stage) pair by construction. All 10 required checks + 1
supplementary check passed, including confirmation that baseline
training loss decreases meaningfully (2.5987 → 0.2814 over 10 epochs),
resolving the Step 1 finding.

**Binding scope statement:** this dataset is synthetic and
validation-only; it does not replace Prolific data and must not support
final scientific claims.

**Status: ✅ COMPLETE.**

---

## 4. Phase 3, Step 2 — Stage-Weighted NDCG (SW-NDCG)

**Files:** `experiments/sw_ndcg.py`, `tests/test_sw_ndcg_smoke.py`

SW-NDCG@K = 0.20·NDCG@K(S1) + 0.30·NDCG@K(S2) + 0.50·NDCG@K(S3) (locked
weights, proposal Section 5.4). Zero re-implementation: consumes
`stage_wise_ndcg()` from `ndcg.py` as sole per-stage NDCG source. 8/8
smoke tests passed, including weight-validation edge cases (non-unit
sum, mismatched keys, negative weights).

Observed validation figures (synthetic, pipeline-check only):

| Model | SW-NDCG@5 | SW-NDCG@10 | SW-NDCG@20 |
|---|---|---|---|
| CAFE-LightFM | 0.8419 | 0.7839 | 0.9049 |
| LightFM-Baseline | 0.5127 | 0.5381 | 0.7740 |

**Status: ✅ COMPLETE.**

---

## 5. Phase 3, Step 3 — Precision@K, Stage-Stratified

**Files:** `experiments/precision_at_k.py`

Standard Precision@K = (relevant items in top-K) / K, fixed denominator
per Järvelin & Kekäläinen (2002) [1] convention (never true-positive
count). Reuses `ranked_relevance_for_user()` from `ndcg.py` — zero
re-implementation of ranking logic. K ∈ {5, 10, 20}, evaluated on
synthetic-v3.

**Status: ✅ COMPLETE.**

---

## 6. Phase 3, Step 4 — Statistical Testing Scaffold

**Files:** `experiments/statistical_testing.py`,
`experiments/step4_integration.py`

- `cohens_dz`, `paired_t_test` (two-tailed, `scipy.stats.ttest_rel`,
  Decision B), `bonferroni_correction` (dynamic M by default, with
  `n_comparisons_override` to reproduce the locked full-study M=63,
  Decision C), `summarize_paired_results`.
- Zero-variance edge case fixed: identical paired differences no longer
  crash the Bonferroni [0,1] range check (`t=0.0, p=1.0` returned
  explicitly).
- SW-NDCG excluded from paired per-user testing in this scaffold
  (Decision A2) — deferred to Prolific data with guaranteed three-session
  coverage per participant.
- 28/28 tests passed (18 statistical-testing smoke tests + 10
  integration smoke tests), fully self-contained (no `conftest.py`
  dependency), per project test-pattern convention.

**Status: ✅ COMPLETE.**

---

## 7. Phase 3, Step 5 — Train-Then-Compare Pipeline

**Files:** `experiments/step5_train_and_compare.py`,
`experiments/training_loop.py`

Executed successfully end-to-end in Colab via
`python -m experiments.step5_train_and_compare`:

- CAFE-LightFM and the PyTorch-native LightFM baseline both trained
  successfully on synthetic-v3 (10 epochs each, full-batch — the locked
  `batch_size=256` is future-facing, not binding for this validation
  run).
- Checkpoints saved:
  - `outputs/checkpoints/cafe_lightfm_v3_seed42.pt`
  - `outputs/checkpoints/lightfm_baseline_v3_seed42.pt`
- Paired NDCG@K / Precision@K comparison table generated via
  `run_ndcg_precision_paired_comparison()`, with the validation-only
  scope reminder correctly included in the output.
- **Bonferroni correction used dynamic M=18** (3 stages × 3 K-values × 2
  metrics), confirmed via `n_comparisons_override=None` →
  `m = len(p_values) = 18`; adjusted α ≈ 0.05/18 ≈ 0.002778. This is
  **not** the locked full-study M=63 (α\* = 0.0008) — reproducing M=63
  requires explicitly passing `n_comparisons_override=63`.
- **Precision@K ranks over the full 30-item catalog** (confirmed via
  `generate_user_ranking()` using `bundle.n_items` /
  `torch.arange(n_items)`), not the per-(user, stage) ~9-item subset.
  Consequently, several Precision@20 comparisons were non-significant —
  attributed to real but non-degenerate sensitivity attenuation as K
  (20) approaches catalog size (30), **not** the previously-documented
  degenerate case (K = full catalog size, which would make Precision@K
  invariant by construction).
- Observed pattern: most NDCG@K and Precision@5/10 comparisons favored
  CAFE-LightFM and were significant under the scaffold.

**Status: ✅ COMPLETE.**

---

## 8. Phase 3, Step 6 — Stage-Differentiation Validation

**Files:** `experiments/stage_differentiation_validation.py`
**Outputs:** `outputs/step6/` —
`aggregated_feature_stage_matrix.csv`,
`item_level_attention_appendix.csv`,
`item_level_top_bottom.csv`,
`pairwise_jsd.csv`,
`step6_report.md`,
`attention_heatmap.png`

### 8.1 Reframing (confirmed decision)

Reframed from "interpretability analysis" to **Stage-Differentiation
Validation**, since synthetic-v3 features carry no confirmed real-world
semantic labels. This script therefore validates only that the SCA layer
produces valid, non-uniform attention allocation, while measuring
whether inter-stage differentiation appears in synthetic-v3 — a
necessary but not sufficient condition for eventual interpretability,
deferred to Prolific data with real and explicitly defined semantic
item metadata.

### 8.2 Corrected design vs. original handoff assumption

Direct source inspection of `models/cafe_lightfm/cafe_lightfm.py` and
`models/cafe_lightfm/sca_layer.py` showed the SCA layer attends over
exactly **two item-side metadata features** — `category` and `program` —
not the full 30-item catalog as originally assumed in the Step 5→6
handoff. Corrected design (confirmed):

1. Primary matrix: **2 features (category, program) × 3 stages**,
   averaged across all 30 catalog items per stage.
2. Appendix: 30 items × 2 features × 3 stages (item-level detail).
3. Item-level diagnostic: top-5 / bottom-5 items by `alpha_category` per
   stage — explicitly labeled as an item-level attention-allocation
   diagnostic, not feature-semantic interpretability.
4. Normalization: raw (item, stage) alpha rows sum to 1.0; aggregated
   stage columns (post-averaging) also sum to 1.0.
5. Pairwise JSD (Lin, 1991 [2]) between stage-level attention
   distributions, reported as a descriptive differentiation measure
   only — no p-value or significance claim attached.

Bundle reconstruction confirmed via source (`data/synthetic_generator_v3.py`
→ `generate_synthetic_dataset_v3()`, then `data/interaction_matrix.py`
→ `build_interaction_matrix()`); default parameters (seed=42, n_users=20,
n_items=30) reproduce the exact bundle used at training time, confirmed
by matching checkpoint-inferred dimensions.

### 8.3 Results

Checkpoint dimensions inferred (not hard-coded):
`n_users=20, n_items=30, n_categories=1, n_programs=3, n_stages=3,
embedding_dim=64`.

**Aggregated attention matrix (mean over 30 items):**

| feature | S1 | S2 | S3 |
|---|---|---|---|
| category | 0.6144 | 0.6035 | 0.6724 |
| program | 0.3856 | 0.3965 | 0.3276 |

**Normalization checks:** raw (item, stage) rows sum to 1.0: `True`.
Aggregated stage columns sum to 1.0: `True`.

**Pairwise JSD (descriptive only):**

| stage pair | JSD (bits) |
|---|---|
| JSD(S1,S2) | 0.0001 |
| JSD(S1,S3) | 0.0026 |
| JSD(S2,S3) | 0.0037 |

**Dataset limitation confirmed:** `n_categories=1` — `item_category` is
sampled independently per event in synthetic-v3, then resolved per
`item_id` via mode (`data/interaction_matrix.py`); the realized
vocabulary collapsed to a single value on this run. This is a documented
synthetic-v3 pipeline property, not a script defect, and limits the
diagnostic value of item-level `alpha_category` differentiation on this
dataset specifically.

**`‖w_stage‖` diagnostic (post-hoc, non-blocking):**

| Parameter | L2 norm |
|---|---|
| `‖w_stage[S1]‖` | 0.987506 |
| `‖w_stage[S2]‖` | 1.071722 |
| `‖w_stage[S3]‖` | 0.735336 |
| `‖w_base‖` | 0.612702 |

### 8.4 Key Finding / Limitation — Weak Inter-Stage Attention Differentiation on Synthetic-v3

The trained checkpoint shows non-zero, stage-differentiated SCA
parameters, confirming the model did **not** remain at its
zero-initialized, stage-agnostic state (baseline-equivalence property
documented in `sca_layer.py`). Aggregated attention over `category` and
`program` is non-uniform overall. However, pairwise JSD between
stage-level attention distributions remains very small relative to the
[0, 1]-bit range.

Given non-zero, differentiated `w_stage` learning, the weak JSD is more
plausibly attributable to a **structural limitation of the synthetic-v3
dataset** — its collapsed `n_categories=1` metadata vocabulary combined
with the 2-feature attention scope — **than to a failure of the SCA
mechanism** itself to learn stage-conditional attention. This is **not**
evidence for or against CAFE-LightFM's architectural capability.
Validation of genuine stage-aware attention differentiation is deferred
to the Prolific pilot (N=50) / full study (N=300), where items have
real and explicitly defined semantic metadata (proposal
Section 5.3).

- No semantic interpretability claim is made.
- No scientific superiority claim is made.

**Status: ✅ COMPLETE** (documented limitation; implementation and
diagnostics both technically successful; no retraining performed; no
Step 6 output files modified as a result of this finding).

---

## 9. Binding Scientific Scope (unchanged, restated)

- All Phase 3 results (Steps 1–6) are computed on **synthetic-v3** only.
- All results are **validation-only** — they confirm the pipeline
  computes correctly, nothing more.
- **No scientific superiority claims** about CAFE-LightFM may be drawn
  from any figure in this phase.
- Formal, publishable claims are reserved exclusively for the **Prolific
  pilot (N=50)** and **full study (N=300)** datasets.

---

## 10. Repository State After Phase 3, Step 6

```
experiments/
├── ndcg.py                              # Step 1
├── sw_ndcg.py                            # Step 2
├── precision_at_k.py                     # Step 3
├── statistical_testing.py                # Step 4
├── step4_integration.py                  # Step 4
├── step5_train_and_compare.py            # Step 5
├── training_loop.py                      # Step 5 (used as-is)
└── stage_differentiation_validation.py   # Step 6

data/
└── synthetic_generator_v3.py             # Step 1.5

models/cafe_lightfm/
├── cafe_lightfm.py
└── sca_layer.py

tests/
├── test_ndcg_smoke.py
├── test_synthetic_generator_v3_smoke.py
├── test_sw_ndcg_smoke.py
├── test_statistical_testing_smoke.py     # 18 tests
└── test_step4_integration_smoke.py       # 10 tests

outputs/
├── checkpoints/
│   ├── cafe_lightfm_v3_seed42.pt
│   └── lightfm_baseline_v3_seed42.pt
└── step6/
    ├── aggregated_feature_stage_matrix.csv
    ├── item_level_attention_appendix.csv
    ├── item_level_top_bottom.csv
    ├── pairwise_jsd.csv
    ├── step6_report.md
    └── attention_heatmap.png
```

---

## 11. Pending / Open Items (not yet actioned)

- Merge this draft into the repository's canonical Phase 3 summary file
  (pending your review and direct upload — not copy-paste, per the
  documented GitHub-web-interface corruption risk).
- Root cause of weak inter-stage JSD (training duration vs. structural
  synthetic-v3 limitation) noted as more likely structural, but not
  formally isolated beyond the `‖w_stage‖` norm diagnostic above.
- Phase 3, Step 7 — **not started**, pending your explicit confirmation
  to proceed.

---

## References

[1] Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-based
Evaluation of IR Techniques. *ACM Transactions on Information Systems*,
20(4), 422–446. DOI: 10.1145/582415.582418.

[2] Lin, J. (1991). Divergence Measures Based on the Shannon Entropy.
*IEEE Transactions on Information Theory*, 37(1), 145–151.
DOI: 10.1109/18.61115.

[3] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
Recommendations. *Proceedings of the 2nd Workshop on New Trends in
Content-Based Recommender Systems*, RecSys 2015. CEUR-WS, Vol. 1448,
14–21.
