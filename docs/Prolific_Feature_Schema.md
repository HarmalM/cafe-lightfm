# CAFE-LightFM — Prolific Pilot Feature Schema
## Protocol Component 2 of 6: Feature Schema

**Status:** DRAFT — pending sign-off before Components 3–6.
**Depends on:** Component 1 (Study Design), approved.
**Binding scope (restated):** This document defines the *feature schema*
only. No architectural change is made or proposed as final here; Section
6 is an audit of what the existing, already-reviewed source code
supports, not a code change.

---

## 1. Three-Way Attribute Taxonomy

| Category | Definition | Fed into $q_i^{(s)}$? |
|---|---|---|
| **Catalog display attributes** | Everything a participant sees on a profile card | No — display only |
| **Model metadata features** | The subset explicitly embedded and attention-weighted by CAFE-LightFM | **Yes** — this is $F_i^{\text{meta}}$ |
| **Behavioral signals** | Clicks, dwell time, comparisons, revisits, shortlist actions, ranking decisions | No — logged separately, used for manipulation-check corroboration and future descriptive analysis, never as an item feature |

Keeping these three streams structurally separate prevents a subtle
error: behavioral signals are *caused by* attention, so feeding them
back into the model as item features would contaminate the very
quantity CAFE-LightFM is trying to explain.

---

## 2. Final Model Metadata Feature List ($F_i^{\text{meta}}$, n=8)

All eight are both catalog-visible and model-encoded — i.e., every
model feature is also something the participant can actually see and
use to decide, which is required for the task to be behaviorally
meaningful. A small set of additional catalog-display-only attributes
(Section 3) is visible but *not* model-encoded.

| # | Feature | Type | Levels / Range | Theoretical rationale |
|---|---|---|---|---|
| 1 | `program_fit` | categorical, 3 levels | {mismatch, partial, match} | Direct proxy for the "program reputation/fit" driver named in proposal Section 1 as a first-order exploration-stage signal |
| 2 | `tuition_level` | ordinal, 5 bins | {very_low … very_high} | Named in proposal Section 1 as a comparison-stage driver ("shift attention to tuition fees") |
| 3 | `scholarship_availability` | categorical, 3 levels | {none, partial, full} | Paired with tuition as a comparison-stage cost signal; distinct construct (funding source vs. sticker price) |
| 4 | `reputation_band` | ordinal, 4 bins | {top-tier … lower-tier} | Exploration-stage broad-scanning signal; low cost to evaluate per item (fits "shallow feature depth" in $S_1$) |
| 5 | `location_region` | categorical, ~5–6 levels | {region groupings} | Exploration-stage filter signal, typically evaluated early and cheaply |
| 6 | `admission_difficulty` | ordinal, 4 bins | {low … very_high} | Risk-relevant, finalization-adjacent signal (proposal Section 1: "prioritize ... risk-weighted evaluation") |
| 7 | `language_requirement` | ordinal, 3 levels | {none, intermediate, advanced} | Practical feasibility constraint, plausible late-stage filter for international applicants specifically |
| 8 | `career_outlook` | ordinal, 3 bins | {low, medium, high} | Named in your list as "career/employability prospects" — long-horizon, outcome-relevant signal distinct from reputation |

**Design constraint honored:** 8 features, all semantically distinct
(no two features are alternate operationalizations of the same
underlying construct), all realistically variable across fictitious
profiles, and all *categorical/ordinal* by construction — no continuous
numeric feature is used directly. This is a deliberate architectural
consistency choice (Section 6): it matches the existing `category`/
`program` precedent exactly (both are already categorical embeddings in
`sca_layer.py`/`cafe_lightfm.py`), so every new feature slots into the
same embedding-table pattern without introducing a new feature-handling
mechanism (e.g., continuous-feature normalization) that would need
separate architectural support and separate validation.

---

## 3. Catalog Display Attributes (visible, not model-encoded)

- Fictitious institution name/ID (e.g., "University A-07") — identity
  only, never embedded as a feature (mirrors how `item_id` is embedded
  separately and left un-attended in the current architecture — Section
  6).
- One short neutral descriptive paragraph per institution (for
  immersion/realism; content generated to be attribute-consistent, not
  independently informative).
- A simple visual/layout element (e.g., a placeholder banner), for
  interface realism only.

These exist so the task does not feel like a bare feature table — but
they carry **no separate model signal**, avoiding an uncontrolled
"impression" channel that could bias choices independently of the eight
measured features above.

---

## 4. Fictitious Profile Generation Method

1. **Catalog size:** 15–20 institutions (unchanged from Component 1),
   generated once and held fixed across all pilot participants (a shared
   catalog, not regenerated per participant), so that population-level
   analyses of catalog properties are possible.
2. **Marginal distributions from cited real sources**, sampled
   independently per feature, then discretized into the bins in Section
   2:
   - `reputation_band`, `admission_difficulty`: derived from a public
     ranking/admissions-statistics source (e.g., QS World University
     Rankings percentile bands) — cite the specific edition used at data
     collection time.
   - `tuition_level`: derived from a public aggregated tuition database
     for international students (e.g., a national higher-education
     statistics agency), binned into quintiles.
   - `scholarship_availability`, `language_requirement`,
     `career_outlook`: derived from published institutional-aid and
     graduate-outcome survey aggregates where available; where no single
     public source covers all institutions consistently, use documented,
     labeled researcher judgment calibrated to plausible real-world
     proportions (e.g., "full scholarship" should not exceed the
     realistic population base rate).
   - `program_fit`, `location_region`: assigned by design to ensure
     balanced coverage (not sampled from an external source, since these
     are catalog-construction choices, not empirical institutional
     facts).
3. **Sampling procedure:** stratified/Latin-hypercube sampling across
   the 8-dimensional attribute space (standard practice in discrete
   choice experiment profile design — Louviere, Hensher, & Swait, 2000)
   rather than independent random draws per feature, specifically to
   support the correlation and dominance controls in Section 5.

---

## 5. Controlling Correlation, Dominance, and Implausible Combinations

| Control | Rule | Rationale |
|---|---|---|
| **Pairwise correlation cap** | No two of the 8 features may have $\|r\| > 0.3$ across the generated catalog | Prevents the model from being unable to distinguish two features' independent contributions to attention (a form of near-collinearity that would make $\alpha(f,s)$ hard to interpret) |
| **No dominant profile** | No institution may be in the top tier on more than 3 of the 8 features simultaneously | Prevents a trivial "always pick institution X" strategy that would collapse the Comparison/Finalization tasks into a non-decision, undermining the funnel design (Component 1) |
| **No dominated profile** | No institution may be in the bottom tier on more than 3 of the 8 features simultaneously (symmetric to above) | Prevents "obviously eliminate institution Y immediately," which would reduce effective catalog size below the intended 15–20 |
| **Plausibility constraints** | Reject combinations contradicting realistic co-occurrence (e.g., top-tier `reputation_band` with `admission_difficulty = low`) | Preserves face validity / reduces hypothetical bias (per the DCE literature on hypothetical bias cited in Component 1) |

Rejection sampling is applied during catalog generation: draw a
candidate profile, check all four constraints, discard and redraw if
violated, until 15–20 valid profiles are obtained.

---

## 6. Mapping to the Existing Item Representation — Architectural Audit (no code changes made)

Based on direct source inspection already performed on the verified
`sca_layer.py` and `cafe_lightfm.py` (Phase 3, Step 7b review), the
audit finding is:

**`StageConditionedAttention.forward()` is already feature-count-agnostic.**
The logit computation —
```
logit_base  = einsum("bfd,d->bf", feature_embeddings, self.w_base)
logit_stage = einsum("bfd,bd->bf", feature_embeddings, w_s)
```
— infers `n_features` directly from `feature_embeddings.shape[1]` and
applies `w_base`/`w_stage` identically regardless of that value. **No
change to `sca_layer.py` is required to support 8 features instead of
2.** The $|F_i^{\text{meta}}|$ scale-correction factor (proposal Section
5.2) also generalizes automatically, since it is just `n_features` read
from the same tensor.

**`CAFELightFM.item_representation()` is the one place that is
currently hard-coded to exactly 2 features**, specifically:
```
cat_emb  = self.category_embedding(category_idx)
prog_emb = self.program_embedding(program_idx)
feature_embeddings = torch.stack([cat_emb, prog_emb], dim=1)
```
This requires: (a) one `nn.Embedding` table per new categorical/ordinal
feature (8 tables total, replacing the current 2), and (b) generalizing
the `torch.stack([...], dim=1)` call to stack all 8 embeddings in a
fixed, documented order — a mechanical, additive change (new embedding
tables + a longer stack call), not a redesign of the SCA mechanism
itself.

**This audit is based on the last-reviewed version of these files
(Step 7b) and should be re-confirmed against the current repository
state in Colab before any implementation work begins** — consistent
with the project's standing practice of verifying source directly
rather than assuming continuity across sessions.

**Recommended item-representation formula under the expanded schema**
(same structural form as the current proposal Section 5.2 equation,
only $F_i^{\text{meta}}$ changes size):

$$q_i^{(s)} = e_{\text{item\_id}} + |F_i^{\text{meta}}| \cdot \sum_{f \in F_i^{\text{meta}}} \alpha(f,s) \cdot e_f, \qquad |F_i^{\text{meta}}| = 8$$

No change to the SCA equations, the shift-invariance argument for
bias removal, or the WARP loss is implied by this expansion.

---

## 7. Open Items (feed into Components 3–6)

1. Exact final source citations for `reputation_band`, `tuition_level`,
   `admission_difficulty` (specific ranking/statistics edition, access
   date) — needed before profile generation, not before this schema is
   approved.
2. Whether `program_fit` is computed relative to a participant's
   self-declared field of interest (collected at intake) or is a fixed
   catalog property independent of the participant — affects whether
   the catalog is shared or partially personalized (Component 1,
   Section 4.2 shared-catalog assumption may need revisiting).
3. This document does not yet specify the manipulation-check instrument
   wording (Component 3) or the code changes audited in Section 6 above
   (deferred pending your confirmation to proceed).

---

## References

Louviere, J. J., Hensher, D. A., & Swait, J. D. (2000). *Stated Choice
Methods: Analysis and Applications*. Cambridge University Press. ISBN
978-0521788304. [Profile-design methodology: stratified sampling,
dominance/correlation control in discrete choice experiments]
