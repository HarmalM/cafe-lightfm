# CAFE-LightFM — Phase 3 Handoff Summary
**For pasting into a new conversation to continue at Phase 3, Step 6**

---

## 1. Project and Repository

- **Project:** CAFE-LightFM — Paper I of III, PhD dissertation *"Stage-Aware Recommendation Systems: Modeling, Detecting, and Generalizing Preference Evolution Across Multi-Stage Decision Contexts"*
- **Repository:** `HarmalM/cafe-lightfm`
- **Current phase:** Phase 3 (Evaluation Pipeline)
- **Locked config:** seed=42, embedding dim=64, batch size=256 (future-facing; not binding for current synthetic v3 runs — see §6)

---

## 2. Binding Scientific Scope (do not relax)

- All current results are computed on **synthetic v3** (`data/synthetic_generator_v3.py`) only.
- All results are **validation-only** — they confirm the pipeline computes correctly, nothing more.
- **No scientific superiority claims** about CAFE-LightFM may be drawn from any figure in this phase.
- Formal, publishable claims are reserved exclusively for the **Prolific pilot (N=50)** and **full study (N=300)** datasets.

---

## 3. Completed Phase 3 Steps

| Step | Content | Status |
|---|---|---|
| 1 | Stage-wise NDCG@K (`experiments/ndcg.py`) — DCG/IDCG per Järvelin & Kekäläinen (2002) | ✅ Complete |
| 1.5 | Unsaturated synthetic v3 dataset (`data/synthetic_generator_v3.py`) — 20 users, 30 items, ~30% per-(user,stage) subsets; fixes v2 saturation (baseline loss was frozen at 0.0) | ✅ Complete |
| 2 | SW-NDCG@K (`experiments/sw_ndcg.py`) — weights S1=0.20, S2=0.30, S3=0.50; zero re-implementation of NDCG logic | ✅ Complete |
| 3 | Precision@K, stage-stratified (`experiments/precision_at_k.py`) — reuses `ranked_relevance_for_user()` from Step 1 | ✅ Complete |
| 4 | Statistical testing scaffold (`experiments/statistical_testing.py`) + integration layer (`experiments/step4_integration.py`) — paired t-test, Cohen's dz, Bonferroni, zero-variance edge case fixed | ✅ Complete |
| 5 | Train-then-compare pipeline (`experiments/step5_train_and_compare.py`) on synthetic v3 | ✅ Complete |

---

## 4. Step 5 — Latest Confirmed State

Executed successfully end-to-end in Colab via `python -m experiments.step5_train_and_compare`:
- CAFE-LightFM trained successfully on synthetic v3 (10 epochs).
- LightFM baseline (PyTorch-native) trained successfully on synthetic v3 (10 epochs).
- Checkpoints saved:
  - `outputs/checkpoints/cafe_lightfm_v3_seed42.pt`
  - `outputs/checkpoints/lightfm_baseline_v3_seed42.pt`
- Paired NDCG@K / Precision@K comparison table generated via `run_ndcg_precision_paired_comparison()`.
- Output correctly included the validation-only scope reminder.
- Observed pattern: most NDCG@K and Precision@5/10 comparisons favored CAFE-LightFM and were significant under the scaffold; several Precision@20 comparisons were not significant (see §5 for the verified, non-degenerate explanation).

---

## 5. Step 5 — Consistency Checks (verified against source, not assumed)

- Bonferroni used **dynamic M=18** (3 stages × 3 K-values × 2 metrics), confirmed via `bonferroni_correction(..., n_comparisons=n_comparisons_override)` with `n_comparisons_override=None` → `m = len(p_values) = 18`.
- **Not** the locked full-study M=63 (α\*=0.0008); reproducing M=63 requires explicitly passing `n_comparisons_override=63`.
- Adjusted α for this run = 0.05 / 18 ≈ **0.002778**.
- Precision@K ranking confirmed (via `generate_user_ranking()` in `ndcg.py`, using `bundle.n_items` / `torch.arange(n_items)`) to rank over the **full 30-item catalog**, not the per-(user,stage) ~9-item subset.
- Precision@20 non-significance is therefore attributable to reduced sensitivity as K (20) approaches catalog size (30) — a real but non-degenerate attenuation effect, **not** the previously documented degenerate case (K = full catalog size making Precision@K invariant by construction).

---

## 6. Important Design Decisions (binding for this scaffold)

- Precision@K uses a **fixed denominator K** (never true-positive count), per Järvelin & Kekäläinen (2002) convention.
- SW-NDCG is **excluded** from paired per-user testing in Step 4 (Decision A2) — deferred to Prolific data where three-session coverage per participant is guaranteed.
- Statistical test: **two-tailed paired t-test** (`scipy.stats.ttest_rel`), Decision B.
- Bonferroni correction uses **dynamic M by default** (Decision C), with an explicit override parameter to reproduce the locked M=63 later.
- Step 5 used `experiments/training_loop.py` **as-is**, full-batch training (no mini-batching applied yet).
- The locked `batch_size=256` is a **future-facing** project parameter — not binding for the current synthetic v3 validation run.
- Zero re-implementation rule maintained throughout: `ranked_relevance_for_user()` (in `ndcg.py`) is the single source of truth for ranking/relevance, reused unchanged by `precision_at_k.py` and `step4_integration.py`.

---

## 7. Key Files

```
experiments/
├── ndcg.py                    # Step 1 — NDCG@K, generate_user_ranking(), ranked_relevance_for_user()
├── sw_ndcg.py                 # Step 2 — SW-NDCG@K
├── precision_at_k.py          # Step 3 — Precision@K
├── statistical_testing.py     # Step 4 — paired t-test, Cohen's dz, Bonferroni
├── step4_integration.py       # Step 4 — per-user metric alignment, comparison runner
├── step5_train_and_compare.py # Step 5 — train-then-compare pipeline, checkpoint saving
└── training_loop.py           # Training loop used as-is by Step 5

data/
└── synthetic_generator_v3.py  # Step 1.5 — unsaturated synthetic dataset

tests/
├── test_ndcg_smoke.py
├── test_synthetic_generator_v3_smoke.py
├── test_sw_ndcg_smoke.py
├── test_statistical_testing_smoke.py     # 18 tests
└── test_step4_integration_smoke.py       # 10 tests

outputs/checkpoints/
├── cafe_lightfm_v3_seed42.pt
└── lightfm_baseline_v3_seed42.pt
```

---

## 8. Proposed Next Step — Phase 3, Step 6

**Step 6 — Stage-Differentiation Validation** (reframed from "interpretability analysis")

**Rationale for reframing (binding decision, made with full authority granted by Dali):** Synthetic v3 features have no confirmed semantic labels (e.g., no feature is tagged as "visa success rate" or "tuition"). A claim that learned attention weights match theoretically expected *semantic* patterns (per proposal §5.1, §6.3) cannot be validated on this dataset. Step 6 will therefore validate only that the SCA layer produces **statistically differentiated, non-uniform attention distributions across S1/S2/S3** — a necessary but not sufficient condition for true interpretability. Full semantic interpretability validation is deferred to Prolific data, where features carry real domain meaning.

**Design (agreed, pending items in §9):**
1. **Parameters inspected:** `W_base`, `W_s1`, `W_s2`, `W_s3`, `b_s1`, `b_s2`, `b_s3`, and the frozen feature embedding matrix `e_f` (dim=64), loaded from `outputs/checkpoints/cafe_lightfm_v3_seed42.pt`.
2. **Extraction method:** `model.eval()`, `torch.no_grad()`; for each stage sⱼ and each feature f in the 30-item catalog, compute α(f, sⱼ) = softmax(W_base·eᶠ + W_sⱼ·eᶠ + b_sⱼ) — the exact locked SCA equation, no re-derivation. Assemble into a Features × Stages (30×3) matrix.
3. **Tables/plots:**
   - Table 1: top-N features by α per stage (feature ID, α-value, rank)
   - Table 2: full 30×3 Feature×Stage α matrix (appendix)
   - Heatmap: Features × Stages, color = α
   - Pairwise Inter-Window JSD(S1,S2), JSD(S2,S3), JSD(S1,S3) — reusing the existing JSD definition from the stage-boundary framework, not a new implementation
4. **Smoke checks:** checkpoint loads without error and dimensions match (64, 3 stage matrices); each α(·,sⱼ) sums to 1.0 within tolerance; output shape = (30,3) with no NaN/Inf; JSD(Sᵢ,Sⱼ) > 0 for all stage pairs (descriptive differentiation criterion, not a significance test).

---

## 9. Step 6 — Pending Decisions (confirm before coding)

1. **Exact model path/class/state_dict attention parameter names** for CAFE-LightFM (e.g., which file defines the SCA layer, and the exact attribute names for `W_base`, `W_sⱼ`, `b_sⱼ` inside `state_dict()`) — not yet confirmed in this conversation; must be supplied before extraction code is written.
2. **Confirm the "Stage-Differentiation Validation" framing** (vs. full interpretability) — proposed and reasoned above; awaiting explicit sign-off.
3. **Top-N features per stage table** — proposed default N=5, adjustable.

---

## 10. Ready-to-Paste Prompt for the Next Conversation

```
MODE: EXPERIMENT — Phase 3, Step 6 continuation.

Project: CAFE-LightFM (Paper I of III). Repository: HarmalM/cafe-lightfm.
Current phase: Phase 3. Steps 1 through 5 are complete and verified
(stage-wise NDCG@K, unsaturated synthetic v3 dataset, SW-NDCG@K,
Precision@K, statistical testing scaffold + integration layer,
train-then-compare pipeline). Step 5 was executed successfully in
Colab: both CAFE-LightFM and LightFM-baseline models were trained on
synthetic v3 and checkpoints saved to
outputs/checkpoints/cafe_lightfm_v3_seed42.pt and
outputs/checkpoints/lightfm_baseline_v3_seed42.pt. The paired
NDCG@K/Precision@K comparison table was generated and verified
internally consistent: Bonferroni used dynamic M=18 (not the locked
full-study M=63), adjusted alpha = 0.05/18 ≈ 0.002778, and Precision@K
ranks over the full 30-item catalog (not the per-stage ~9-item
subset), so Precision@20 non-significance is a real but non-degenerate
sensitivity effect.

Binding scope: all results are on synthetic v3 only, validation-only,
no scientific superiority claims. Formal claims are reserved for the
Prolific pilot (N=50) / full study (N=300).

We are proceeding to Phase 3, Step 6 — Stage-Differentiation
Validation (reframed from "interpretability analysis" because
synthetic v3 features carry no confirmed semantic labels, so only
statistical differentiation of attention across S1/S2/S3 can be
validated, not semantic correctness). Design already agreed: inspect
W_base, W_s1/s2/s3, b_s1/s2/s3, and the frozen feature embedding
matrix from the CAFE-LightFM checkpoint; extract a 30(features)×3
(stages) alpha matrix using the exact locked SCA softmax equation;
produce a top-N(=5) table per stage, a full feature×stage table, a
heatmap, and pairwise Inter-Window JSD between stages (reusing the
existing JSD implementation); smoke checks: checkpoint loads
correctly, alpha rows sum to 1.0, output shape (30,3) with no NaN/Inf,
JSD(Si,Sj) > 0 for all pairs.

Pending before coding: (1) exact file/class/state_dict attribute names
for the SCA layer parameters in the CAFE-LightFM model — please
provide or confirm; (2) sign-off on the Stage-Differentiation framing;
(3) confirm top-N=5.

Please confirm the design, ask only the one blocking clarifying
question if still needed (model path/class), then proceed to Step 6
implementation once confirmed. Do not go to the next step without
explicit confirmation. All code in Python, with docstrings, type
hints, reproducibility seed=42, and a smoke test block.
```
