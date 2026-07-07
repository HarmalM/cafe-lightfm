# CAFE-LightFM — Phase 3 Progress Summary (Steps 1–7)

**SUPERSEDES:** `docs/Phase3_Progress_Summary_Steps1_to_6.md`

**Project:** CAFE-LightFM (Paper I of III, PhD dissertation: *Stage-Aware
Recommendation Systems: Modeling, Detecting, and Generalizing Preference
Evolution Across Multi-Stage Decision Contexts*)
**Repository:** `HarmalM/cafe-lightfm`
**Status at time of writing:** Phase 3, Step 7 COMPLETE. Repository
cleanup COMPLETE. Full test suite (125 tests) passing. Phase 3 is
CLOSED pending your review; next work moves to dissertation Sections
IV–VIII and/or the Prolific pilot (N=50).

---

## 1–7. Steps 1 through 6 (unchanged)

Steps 1 (Stage-wise NDCG@K), 1.5 (unsaturated synthetic-v3 dataset), 2
(SW-NDCG), 3 (Precision@K), 4 (statistical testing scaffold), 5
(train-then-compare pipeline), and 6 (Stage-Differentiation Validation)
are unchanged from `Phase3_Progress_Summary_Steps1_to_6.md` §1–8. That
document's content is incorporated here by reference rather than
duplicated; see that file for full detail. Key carried-forward facts
needed for Step 7 context:

- **Step 5 checkpoints** (reused, not retrained, in Step 7):
  `outputs/checkpoints/cafe_lightfm_v3_seed42.pt`,
  `outputs/checkpoints/lightfm_baseline_v3_seed42.pt`.
- **Step 6 key finding** (binding, re-stated because Step 7 directly
  builds on it): the trained CAFE-LightFM checkpoint shows non-zero,
  stage-differentiated SCA parameters (`w_stage[S1]=0.988,
  w_stage[S2]=1.072, w_stage[S3]=0.735` L2 norms vs. `w_base=0.613`),
  and mean attention over {category, program} is non-uniform overall
  (~0.60–0.67 vs. 0.5 at init) — yet pairwise JSD between stage-level
  attention distributions is very small (0.0001–0.0037 bits). This was
  attributed to a **structural limitation of synthetic-v3**
  (`n_categories=1`, collapsed via per-event sampling + mode
  resolution, combined with the 2-feature attention scope), not a
  failure of the SCA mechanism. No semantic interpretability or
  scientific superiority claim was made.

---

## 8. Phase 3, Step 7 — Ablation Study

**Files:**
- `models/cafe_lightfm/sca_layer.py` — added `freeze_uniform: bool = False`
- `models/cafe_lightfm/cafe_lightfm.py` — threaded `freeze_uniform` through
- `models/cafe_lightfm/warp_loss.py` — `cafe_scorer` gains `freeze_uniform` passthrough
- `experiments/training_loop.py` — `train_cafe_lightfm` gains
  `stage_idx_override`, `stage_order`, `stage_to_idx`, `freeze_uniform`
- `experiments/ndcg.py` — `generate_user_ranking` /
  `ranked_relevance_for_user` gain `stage_idx_override`, `stage_to_idx`
- `experiments/step4_integration.py` — `per_user_metric` gains the same
  two evaluation-time overrides (passthrough only)
- `experiments/step7_ablation_study.py` — new orchestration script
- `tests/test_training_loop_step7_smoke.py`,
  `tests/test_step7_ablation_smoke.py` — new smoke tests (13 checks)

**Design principle (maintained throughout):** every new parameter
across all six modified files defaults to `None`/`False`/the existing
module-level constant, so Steps 1–6 call sites and results are
byte-identical and unaffected. This was explicitly verified via a
Step 5 regression re-run (below) before any ablation result was
trusted.

### 8.1 Four variants (proposal Section 5.6)

1. **Full** — the existing Step 5 checkpoint, reused without retraining.
2. **noStage** — trained with `stage_idx_override=0`: every stage's
   data is scored under a single fixed stage index, so
   `w_stage[1]`/`w_stage[2]` never receive gradient and remain
   zero-initialized. Disables stage-conditional variation via a
   constant stage index — **not** a bias-freeze (this implementation
   has no `b_{s_j}` term; see `sca_layer.py`).
3. **noAttention** — trained with `freeze_uniform=True`: the SCA
   layer's `w_base`/`w_stage` logit computation is bypassed entirely,
   alpha held at a constant uniform 1/n_features, so `w_base`/`w_stage`
   never enter the autograd graph.
4. **2Stage** — stage labels merged {S1: exploration} vs. {S2 ∪ S3:
   decision}, `n_stages=2`. Evaluated **descriptively only** — no
   paired significance test against Full (3-stage vs. 2-stage
   cardinality mismatch would be structurally misleading). SW-NDCG not
   computed for this variant (its locked weights assume exactly 3
   stages).

### 8.2 Critical evaluation-time correctness fix

noStage's evaluation **must** use the same `stage_idx_override=0` it
trained under. Evaluating via the ordinary per-stage mapping (S1->0,
S2->1, S3->2) would score S2/S3 with the **untrained**
`w_stage[1]`/`w_stage[2]`, producing a train/eval mismatch. This was
identified during design review (2026-07-07) and implemented via the
Step 7 additions to `ndcg.py`/`step4_integration.py` before any ablation
result was computed. noAttention required no equivalent fix (zero-init
parameters remain zero regardless of eval-time flags; softmax(0) is
uniform by construction).

### 8.3 Verification sequence (all completed, Colab)

1. `pytest tests/test_training_loop_step7_smoke.py` — 5/5 passed
   (revised twice: once to remove an incorrect one-call-per-stage
   assumption that ignored WARP's negative-sampling call volume, and
   confirmed correct on the third pass).
2. **Step 5 regression re-run**: `experiments/step5_train_and_compare.py`
   re-executed unmodified end-to-end; per-epoch losses matched the
   original Step 5 log. This is the trust gate for having modified
   `training_loop.py`.
3. `python -m experiments.step7_ablation_study` — completed after one
   fix (§8.4) and one test-stub fix (§8.5).
4. `pytest tests/test_step7_ablation_smoke.py` — 8/8 passed.
5. Full repository test suite: **125/125 passed** (90.73s), post-cleanup.

### 8.4 Bug found and fixed: frozen-dataclass violation

`remap_bundle_to_2stage()` originally used `copy.copy()` + attribute
reassignment, which raised
`dataclasses.FrozenInstanceError: cannot assign to field
'positive_pairs_by_stage'` — `InteractionMatrixBundle` is a frozen
dataclass. Fixed by switching to
`dataclasses.replace(bundle, positive_pairs_by_stage={...})`, which
returns a new instance with only the specified field overridden and all
other fields (`n_users`, `n_items`, `item_feature_idx_by_item`, etc.)
copied unchanged. This had been flagged as an explicit, labeled
assumption requiring Colab verification before the bug surfaced —
confirmed as the correct call.

### 8.5 Test-stub fix (no production code change)

Once `remap_bundle_to_2stage()` required `dataclasses.replace()`, the
test suite's `_StubBundle` (a plain class) failed with
`TypeError: replace() should be called on dataclass instances`.
`_StubBundle` was converted to `@dataclass(frozen=True)` to match the
real bundle's construction contract. Test logic and assertions
unchanged.

### 8.6 Results

**Formal paired comparisons (Bonferroni-corrected, dynamic M=18 per
family, alpha* ~ 0.002778):**

| Comparison | Significant (corrected) | Significant (uncorrected alpha=0.05) | max abs(Cohen's dz) |
|---|---|---|---|
| Full vs. noStage | 0/18 | 0/18 | 0.325 |
| Full vs. noAttention | 0/18 | 1/18 (borderline, Precision@20_S1, p~0.031) | 0.522 |

**Descriptive-only (2Stage, structural diagnostic):** the merged
"decision" stage (S2 ∪ S3) shows higher NDCG@K/Precision@K than either
S2 or S3 alone under Full. This is very likely a **mechanical merge
artifact** — combining two stages' positive sets increases the number
of relevant items available per evaluated (user, stage) pair,
mechanically inflating NDCG@K/Precision@K at fixed K independent of any
real ranking-quality change. This is precisely why 2Stage was excluded
from formal significance testing.

### 8.7 Key Finding / Limitation — Null Ablation Result (binding, conservative)

- **0 of 18** Full-vs-noStage and **0 of 18** Full-vs-noAttention
  comparisons reached significance after Bonferroni correction. At
  uncorrected alpha=0.05, only 1 of 36 total comparisons reached
  nominal significance — a rate consistent with chance, not a detected
  effect.
- **This null result is not independent of the Step 6 finding.** It is
  read as the expected downstream consequence of Step 6's documented
  weak inter-stage attention differentiation on synthetic-v3
  (structural, not mechanism failure): if the SCA layer was not
  learning strongly differentiated stage-conditioned signal on this
  dataset to begin with, removing stage-conditioning or attention has
  correspondingly little measurable signal to remove.
- **No claim is made that stage-conditioning or attention weighting are
  architecturally unnecessary.** A null result under Bonferroni
  correction with n=20 users and small observed effect sizes
  (max |dz| ~ 0.32-0.52) indicates the effect (if any) was **not
  detectable at this sample size on this dataset** — it does not
  establish absence of an effect.
- **No claim is made for or against CAFE-LightFM's final architectural
  value.** This is a pipeline-validation exercise only.
- **2Stage's apparently favorable descriptive numbers are attributed to
  the merge artifact described in §8.6**, not to a genuine
  ranking-quality improvement from 2-stage decomposition.
- All figures are computed on **synthetic-v3 only** and are
  **validation-only**. Formal, publishable claims about
  CAFE-LightFM's architectural contribution — including the relative
  necessity of stage-conditioning and attention — remain reserved
  exclusively for the Prolific pilot (N=50) and full study (N=300)
  (proposal Section 5.3).

**Status: Phase 3, Step 7 — COMPLETE.**

---

## 9. Repository Cleanup (2026-07-08)

Following Step 7 completion, the repository was inspected for
duplicate/obsolete files. **Verified via `grep` that no remaining code
referenced the removed files before deletion.** Full test suite
(125 tests) re-run and confirmed passing after cleanup.

**Removed (10 files):**
```
experiments/ndcgOld.py
experiments/step4_integrationOld.py
experiments/training_loopOld.py
models/cafe_lightfm.py                    # flat duplicate, shadowed
models/sca_layer.py                       # models/cafe_lightfm/ package
models/warp_loss.py                       # (import-resolution hazard)
models/cafe_lightfm/cafe_lightfmOld.py
models/cafe_lightfm/sca_layerOld.py
models/cafe_lightfm/warp_lossOld.py
tests/test_training_loop_step7_smokeOld.py
```

**Renamed (1 file, malformed filename with a literal trailing space):**
```
models/__init__ .py  ->  models/__init__.py
```

**Not yet resolved (deferred, non-blocking for Phase 3 close-out):**
- `smoke_test.py` (top-level) — purpose/active-use status unconfirmed.
- `experiments/ablation/__init__.py`, `experiments/baselines/__init__.py`,
  `experiments/evaluation/__init__.py` — empty subpackages, alongside
  the flat `experiments/step7_ablation_study.py`. Intended future
  scaffolding vs. abandoned organizational plan is unconfirmed. Neither
  item blocks Phase 3 close-out; both are candidates for resolution
  before Paper I's code-release/reproducibility package is finalized.

---

## 10. Binding Scientific Scope (unchanged, restated)

- All Phase 3 results (Steps 1–7) are computed on **synthetic-v3**
  only.
- All results are **validation-only** — they confirm the pipeline
  computes correctly, nothing more.
- **No scientific superiority or necessity claims** about
  CAFE-LightFM or any of its components may be drawn from any figure in
  this phase.
- Formal, publishable claims are reserved exclusively for the
  **Prolific pilot (N=50)** and **full study (N=300)** datasets.

---

## 11. Repository State After Step 7 + Cleanup

```
experiments/
├── ndcg.py                              # Step 1 (+ Step 7 overrides)
├── sw_ndcg.py                            # Step 2
├── precision_at_k.py                     # Step 3
├── statistical_testing.py                # Step 4
├── step4_integration.py                  # Step 4 (+ Step 7 overrides)
├── step5_train_and_compare.py            # Step 5
├── training_loop.py                      # Step 5 (+ Step 7 params)
├── stage_differentiation_validation.py   # Step 6
├── step7_ablation_study.py               # Step 7 (new)
├── ablation/__init__.py                  # empty -- status unconfirmed
├── baselines/__init__.py                 # empty -- status unconfirmed
└── evaluation/__init__.py                # empty -- status unconfirmed

data/
├── synthetic_generator_v3.py             # Step 1.5
└── interaction_matrix.py                 # InteractionMatrixBundle (frozen dataclass)

models/
├── __init__.py                           # renamed from malformed "__init__ .py"
├── baselines/lightfm_pytorch.py
└── cafe_lightfm/
    ├── cafe_lightfm.py                   # + Step 7 freeze_uniform
    ├── sca_layer.py                      # + Step 7 freeze_uniform
    └── warp_loss.py                      # + Step 7 freeze_uniform passthrough

tests/                                    # 125 tests total, all passing
├── test_training_loop_step7_smoke.py     # 5 tests (Step 7)
├── test_step7_ablation_smoke.py          # 8 tests (Step 7)
└── ... (Steps 1-6 test files, unchanged)

outputs/
├── checkpoints/
│   ├── cafe_lightfm_v3_seed42.pt                 # Full (Step 5)
│   ├── lightfm_baseline_v3_seed42.pt             # Step 5
│   ├── cafe_lightfm_v3_seed42_nostage.pt         # Step 7
│   ├── cafe_lightfm_v3_seed42_noattention.pt     # Step 7
│   └── cafe_lightfm_v3_seed42_2stage.pt          # Step 7
└── step6/ ...                                     # Step 6 outputs, unchanged

docs/
└── Phase3_Progress_Summary_Steps1_to_7.md   # this file, supersedes Steps1_to_6
```

---

## 12. Open Items (not yet actioned)

- Resolve `smoke_test.py` and the three empty `experiments/` subpackages
  (§9) before the reproducibility/code-release package is finalized —
  non-blocking for Phase 3 close-out itself.
- Phase 3 is otherwise CLOSED. Next work: dissertation Sections IV–VIII
  (currently not started), Theoretical Framework section (Definition
  III.1, Proposition III.1, Theorem III.1 convergence sketch — pending),
  and the formal Prolific pilot (N=50) / full study (N=300), which are
  the sole source of any permitted scientific superiority/necessity
  claim about CAFE-LightFM.

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

[4] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
Sciences* (2nd ed.). Routledge. ISBN 978-0-8058-0283-2.

[5] Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo
delle probabilità. *Pubblicazioni del R. Istituto Superiore di Scienze
Economiche e Commerciali di Firenze*, 8, 3–62.
