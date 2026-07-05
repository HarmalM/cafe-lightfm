# CAFE-LightFM — Phase 3 Handoff Summary
**For pasting into a new conversation to continue at Phase 3, Step 7**

---

## 1. Project and Repository

- **Project:** CAFE-LightFM — Paper I of III, PhD dissertation *"Stage-Aware Recommendation Systems: Modeling, Detecting, and Generalizing Preference Evolution Across Multi-Stage Decision Contexts"*
- **Repository:** `HarmalM/cafe-lightfm`
- **Progress summary (finalized, uploaded):** `docs/Phase3_Progress_Summary_Steps1_to_6.md`
- **Current phase:** Phase 3 (Evaluation Pipeline), Steps 1–6 COMPLETE
- **Locked config (future-facing):** seed=42, embedding dim=64, batch size=256 — not binding for current synthetic-v3 runs

---

## 2. Binding Scientific Scope (do not relax)

- All current results are computed on **synthetic v3** (`data/synthetic_generator_v3.py`) only.
- All results are **validation-only** — they confirm the pipeline computes correctly, nothing more.
- **No scientific superiority claims** about CAFE-LightFM may be drawn from any figure in Phase 3.
- **No semantic interpretability claims** — synthetic-v3's `category`/`program` indices carry no confirmed real-world meaning.
- Formal, publishable claims are reserved exclusively for the **Prolific pilot (N=50)** and **full study (N=300)** datasets.

---

## 3. Completed Phase 3 Steps

| Step | Content | Status |
|---|---|---|
| 1 | Stage-wise NDCG@K (`experiments/ndcg.py`) — DCG/IDCG per Järvelin & Kekäläinen (2002) | ✅ Complete |
| 1.5 | Unsaturated synthetic v3 dataset (`data/synthetic_generator_v3.py`) — 20 users, 30 items, ~30% per-(user,stage) subsets; fixes v2 saturation | ✅ Complete |
| 2 | SW-NDCG@K (`experiments/sw_ndcg.py`) — weights S1=0.20, S2=0.30, S3=0.50; zero re-implementation of NDCG logic | ✅ Complete |
| 3 | Precision@K, stage-stratified (`experiments/precision_at_k.py`) — reuses `ranked_relevance_for_user()` from Step 1 | ✅ Complete |
| 4 | Statistical testing scaffold (`experiments/statistical_testing.py`) + integration layer (`experiments/step4_integration.py`) — paired t-test, Cohen's dz, Bonferroni, zero-variance edge case fixed | ✅ Complete |
| 5 | Train-then-compare pipeline (`experiments/step5_train_and_compare.py`) on synthetic v3 | ✅ Complete |
| 6 | Stage-Differentiation Validation (`experiments/stage_differentiation_validation.py`) on synthetic v3 | ✅ Complete |

---

## 4. Confirmed Step 5 State (verified against source, not assumed)

- CAFE-LightFM and the PyTorch-native LightFM baseline both trained successfully on synthetic-v3 (10 epochs each, full-batch).
- Checkpoints saved:
  - `outputs/checkpoints/cafe_lightfm_v3_seed42.pt`
  - `outputs/checkpoints/lightfm_baseline_v3_seed42.pt`
- Paired NDCG@K / Precision@K comparison table generated via `run_ndcg_precision_paired_comparison()`, including the validation-only scope reminder.
- Bonferroni used **dynamic M=18** (3 stages × 3 K-values × 2 metrics), confirmed via `n_comparisons_override=None` → `m = len(p_values) = 18`. Adjusted α ≈ 0.05/18 ≈ 0.002778. **Not** the locked full-study M=63 (α\*=0.0008) — reproducing M=63 requires `n_comparisons_override=63`.
- Precision@K ranks over the **full 30-item catalog** (confirmed via `generate_user_ranking()` using `bundle.n_items`), not the per-(user,stage) ~9-item subset — so Precision@20 non-significance is a real, non-degenerate sensitivity-attenuation effect (K approaching catalog size), not the previously documented degenerate case.
- Observed pattern: most NDCG@K and Precision@5/10 comparisons favored CAFE-LightFM and were significant under the scaffold.

---

## 5. Confirmed Step 6 State — Stage-Differentiation Validation

**Corrected design (vs. original handoff assumption):** direct source inspection of `models/cafe_lightfm/cafe_lightfm.py` and `models/cafe_lightfm/sca_layer.py` showed SCA attends over exactly **two item-side metadata features** (`category`, `program`) — not the full 30-item catalog. Alpha has shape `(batch, 2)`, not `(batch, 30)`.

**Implementation:** `experiments/stage_differentiation_validation.py` — loads the CAFE-LightFM checkpoint, infers all dimensions from `state_dict` (no hard-coding), reconstructs the synthetic-v3 bundle via the confirmed real functions `generate_synthetic_dataset_v3()` (`data/synthetic_generator_v3.py`) → `build_interaction_matrix()` (`data/interaction_matrix.py`), and extracts alpha via `CAFELightFM.item_representation()` directly (zero re-implementation of attention logic).

**Results:**

| feature | S1 | S2 | S3 |
|---|---|---|---|
| category | 0.6144 | 0.6035 | 0.6724 |
| program | 0.3856 | 0.3965 | 0.3276 |

Normalization checks passed: raw (item, stage) rows sum to 1.0 = `True`; aggregated stage columns sum to 1.0 = `True`.

**Pairwise JSD (descriptive only, no significance claim):**

| stage pair | JSD (bits) |
|---|---|
| JSD(S1,S2) | 0.0001 |
| JSD(S1,S3) | 0.0026 |
| JSD(S2,S3) | 0.0037 |

**`‖w_stage‖` diagnostic (post-hoc, confirms non-zero stage-conditional learning):**

| Parameter | L2 norm |
|---|---|
| `‖w_stage[S1]‖` | 0.987506 |
| `‖w_stage[S2]‖` | 1.071722 |
| `‖w_stage[S3]‖` | 0.735336 |
| `‖w_base‖` | 0.612702 |

**Dataset limitation confirmed:** checkpoint infers `n_categories=1` — `item_category` is sampled independently per event, then resolved per item via mode (`data/interaction_matrix.py`); the realized vocabulary collapsed to a single value on this run. Documented pipeline property, not a script defect.

**Finalized Key Finding / Limitation (recorded in `docs/Phase3_Progress_Summary_Steps1_to_6.md`, §8.4):** SCA parameters are non-zero and stage-differentiated (model did not remain stage-agnostic), and attention is non-uniform overall, but pairwise JSD between stage-level attention distributions is very small. This is more plausibly attributable to synthetic-v3's structural limitations (`n_categories=1`, 2-feature attention scope) than to a failure of the SCA mechanism. **Not** evidence for or against CAFE-LightFM's architectural capability — genuine stage-aware attention validation is deferred to Prolific data, where items have real and explicitly defined semantic metadata.

**Outputs (`outputs/step6/`):** `aggregated_feature_stage_matrix.csv`, `item_level_attention_appendix.csv`, `item_level_top_bottom.csv`, `pairwise_jsd.csv`, `step6_report.md`, `attention_heatmap.png`.

---

## 6. Repository State After Step 6

```
experiments/
├── ndcg.py                              # Step 1
├── sw_ndcg.py                            # Step 2
├── precision_at_k.py                     # Step 3
├── statistical_testing.py                # Step 4
├── step4_integration.py                  # Step 4
├── step5_train_and_compare.py            # Step 5
├── training_loop.py                       # Step 5 (used as-is)
└── stage_differentiation_validation.py   # Step 6

data/
├── synthetic_generator_v3.py             # Step 1.5
└── interaction_matrix.py                 # (InteractionMatrixBundle)

models/cafe_lightfm/
├── cafe_lightfm.py
├── sca_layer.py
└── warp_loss.py

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

docs/
└── Phase3_Progress_Summary_Steps1_to_6.md   # finalized, uploaded
```

---

## 7. Proposed Goal — Phase 3, Step 7 (PROPOSAL ONLY — not started, needs sign-off)

**Candidate: Ablation Study Design and Execution** (proposal Section 5.6, secondary objective 4 — "isolate the contribution of each architectural component").

Rationale for proposing this next: Steps 1–6 established the evaluation pipeline, trained the full model, and validated the SCA mechanism's basic behavior. The proposal's four ablation conditions have not yet been implemented or run on synthetic-v3:

1. `CAFE-LightFM-noStage` — stage conditioning removed (uniform attention weights)
2. `CAFE-LightFM-noAttention` — stage-specific embeddings without attention weighting
3. `CAFE-LightFM-2Stage` — two stages instead of three (exploration vs. decision)
4. `CAFE-LightFM-Full` — complete model (already trained, Step 5 checkpoint)

**Open design questions to resolve before coding (do not guess):**
- How is `noStage` implemented cleanly — forcing `stage_idx=0` for all inputs (reuses existing code, no `w_stage` learning signal per stage) vs. architecturally removing `w_stage` (requires a variant class)?
- How is `noAttention` implemented — bypass softmax entirely (raw mean of feature embeddings) vs. fixed uniform 0.5/0.5 weights (mathematically identical here, given only 2 features)?
- `2Stage` requires re-labeling S1/S2/S3 → 2 stages in the synthetic-v3 bundle. Is "exploration vs. decision" defined as {S1} vs. {S2,S3}, or some other merge? This needs an explicit, labeled decision before any dataset regeneration.
- Training protocol: same 10 epochs / full-batch / seed=42 as Step 5, for exact comparability?
- Evaluation: reuse Step 4's statistical testing scaffold (paired t-test + Bonferroni) across all four variants, or a simpler descriptive comparison first?

**This is a proposal, not a confirmed plan.** The next conversation should confirm the design before implementation, per project protocol (confirm all open decisions explicitly before proceeding).

---

## 8. Files to Inspect Before Coding Step 7

Not yet reviewed in this conversation thread and needed for a non-guessing implementation:

- `experiments/training_loop.py` — exact training loop / optimizer / epoch structure used by Step 5, to replicate identically for each ablation variant
- `experiments/step5_train_and_compare.py` — how Step 5 wires model construction, training, and checkpointing together (needed to create ablation variants without duplicating this logic)
- `models/cafe_lightfm/cafe_lightfm.py` and `models/cafe_lightfm/sca_layer.py` — already inspected (Step 6); re-confirm no changes since then
- If a `2Stage` variant is pursued: whichever module currently encodes `SESSION_INDEX_TO_STAGE` / stage labeling (referenced in `data/synthetic_generator_v3.py` as imported from `data/synthetic_generator.py`) — not yet inspected

---

## 9. Ready-to-Paste Prompt for the Next Conversation

```
MODE: EXPERIMENT — Phase 3, Step 7 planning.

Project: CAFE-LightFM (Paper I of III). Repository: HarmalM/cafe-lightfm.
Phase 3, Steps 1–6 are complete and documented in
docs/Phase3_Progress_Summary_Steps1_to_6.md. Summary: stage-wise NDCG@K,
unsaturated synthetic-v3 dataset (20 users, 30 items, seed=42), SW-NDCG@K,
Precision@K, statistical testing scaffold (paired t-test, Cohen's dz,
dynamic Bonferroni M=18), a train-then-compare pipeline (CAFE-LightFM and
a PyTorch-native LightFM baseline both trained successfully, checkpoints
saved), and Stage-Differentiation Validation (Step 6).

Step 6 key finding (binding, do not re-litigate without new evidence):
SCA learned non-zero, stage-differentiated parameters (‖w_stage[S1]‖=0.988,
‖w_stage[S2]‖=1.072, ‖w_stage[S3]‖=0.735 vs. ‖w_base‖=0.613), and mean
attention over {category, program} is non-uniform (~0.60-0.67 vs 0.5 at
init), but pairwise JSD between stage-level attention distributions is
very small (0.0001-0.0037 bits). This is documented as a synthetic-v3
structural limitation (n_categories=1, collapsed via per-event category
sampling + mode resolution, combined with the 2-feature attention scope)
rather than evidence against the SCA mechanism's capability. No semantic
interpretability or scientific superiority claim was made.

Binding scope: all results are on synthetic-v3 only, validation-only. No
scientific superiority claims. No semantic interpretability claims.
Formal claims reserved for Prolific pilot (N=50) / full study (N=300).

Proposed (NOT yet confirmed) goal for Step 7: Ablation Study Design and
Execution per proposal Section 5.6 — four variants (noStage,
noAttention, 2Stage, Full) to isolate the contribution of stage
conditioning, the attention mechanism, and hybrid feature integration.
Open design questions (see handoff Section 7) must be resolved explicitly
before any code is written — particularly how "noStage" and
"noAttention" are operationalized without duplicating existing model
code, and how "2Stage" merges S1/S2/S3 into two labels.

Files needed before coding (not yet inspected in this thread):
experiments/training_loop.py and experiments/step5_train_and_compare.py
(to replicate the Step 5 training protocol identically per variant), and
if 2Stage is pursued, the stage-labeling module referenced from
data/synthetic_generator.py.

Please: (1) confirm or challenge the proposed Step 7 goal, (2) ask only
the specific blocking questions needed (stage-merge definition for
2Stage, noStage/noAttention operationalization), (3) request the listed
files before writing any code, (4) do not proceed to implementation
without explicit confirmation of the design. All code in Python 3.10+,
with docstrings, type hints, reproducibility seed=42, and a smoke test
block, per established project convention. Zero re-implementation of
existing scoring/attention/training logic.
```
