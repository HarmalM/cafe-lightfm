# Theoretical Framework — Planning / Audit Note
**Purpose:** map Proposal Section 5.1 against the implemented CAFE-LightFM
architecture (Phase 2–3) before drafting Definition III.1 / Proposition
III.1 / Theorem III.1. No code. No scientific superiority claim. No
overclaimed mathematical guarantees.

---

## 1. What Section 5.1 Currently Claims

Section 5.1 ("Theoretical Framework — Formal Stage Definition") makes
three distinct claims, which are worth separating because they carry
different evidentiary weight:

**(a) An ontological/statistical definition of a decision stage:**
stage sⱼ is defined as an interval of a user's decision trajectory
during which the distribution of feature-attention weights P(f | sⱼ) is
*significantly different* from adjacent stages, as measured by
Jensen–Shannon divergence (JSD). This is framed as grounding stage
boundaries in *observable behavioral data*, and as enabling *automatic
stage inference from interaction logs*.

**(b) A three-stage taxonomy** (Exploration / Comparison /
Finalization), described qualitatively (breadth of scanning,
revisitation rate, elimination behavior) and attributed to "established
behavioral decision theory literature" — though no specific citation is
given in this section for that attribution (a gap noted here, not
resolved).

**(c) An implicit claim of operational unity:** the section reads as if
one framework both *defines* stages statistically (via JSD) and *models*
preferences conditioned on those stages (via CAFE-LightFM), as a single
integrated theoretical apparatus.

Claim (c) is the one that needs the most scrutiny below — it is not
false, but it currently conflates two logically separable
constructs that the codebase itself keeps apart.

---

## 2. What Is Supported by the Implemented CAFE-LightFM Architecture

The implemented `StageConditionedAttention` layer and `CAFELightFM`
model (Phase 2, Step 2) formalize and verify a **narrower, conditional**
claim than (a) above:

- **Given** a stage label sⱼ (as an index into a finite, ordered set),
  the model defines a stage-conditioned attention distribution
  α(f, sⱼ) = softmax_f(w_base·e_f + w_{sⱼ}·e_f) over item metadata
  features, and a corresponding stage-conditioned item representation
  q_i(sⱼ). This is fully implemented, gradient-verified, and covered by
  smoke tests (`test_gradient_flow`, `test_attention_is_valid_distribution`).
- The model is architecturally proven (not just empirically observed) to
  reduce exactly to the stage-blind LightFM baseline (Kula, 2015) at
  zero initialization — `test_equivalence_to_step1_baseline_at_init`
  confirms this numerically, and the reduction is analytically derivable
  from softmax shift-invariance plus the |F_i| scale correction (see
  `sca_layer.py` docstring). This is a genuine, provable mathematical
  property of the architecture, independent of any dataset.

**What is NOT implemented:** the JSD-based automatic stage *detection*
criterion described in claim (a) — the mechanism that would take raw
interaction logs and output stage boundaries — is a **separate**
component (`data/jsd_detector.py`, `data/mdl_threshold.py`, and the
HMM-distilled-to-MLP stage inferrer referenced in project memory). It is
not exercised anywhere in the Phase 3 ablation/evaluation pipeline.
Every stage label used in Steps 1–7 (S1/S2/S3) is a **ground-truth
label supplied by the synthetic generator**, not a label produced by
JSD-based boundary detection.

**Audit conclusion:** CAFE-LightFM (Paper I) formalizes and validates
stage-*conditioned* preference modeling given stage labels. It does
**not** itself formalize or validate stage *detection*. This is not a
weakness to hide — it is a legitimate scope boundary that should be
stated explicitly, and it clarifies the cross-paper dependency already
noted in project context: Paper II is the natural home for formalizing
and validating the JSD/MDL/HMM-based detection mechanism that claim (a)
describes, while Paper I formalizes what happens *once* a stage
assignment (however obtained) is available.

---

## 3. What Is Supported Only as Synthetic-v3 Validation

- The Step 6 finding that trained CAFE-LightFM parameters are non-zero
  and stage-differentiated (`‖w_stage[S1]‖=0.988`, etc.), while pairwise
  JSD between the *model's learned attention distributions* across
  given stages is small (0.0001–0.0037 bits), is a validation of
  whether the SCA mechanism learns differentiated behavior when trained
  on ground-truth-labeled stages — a narrower question than whether
  P(f|sⱼ) computed directly from raw behavioral signal satisfies claim
  (a)'s JSD-based stage-boundary criterion. These must not be conflated
  in the write-up: Step 6 evaluated the trained model's internal
  attention, not the raw-signal stage-boundary detectability that
  Section 5.1 (a) describes.
- The Step 7 null ablation result (no significant Full-vs-noStage or
  Full-vs-noAttention difference after Bonferroni correction) is, per
  the already-documented conservative reading, attributable to
  synthetic-v3's structural limitations (collapsed `n_categories=1`,
  2-feature attention scope) combined with low statistical power
  (n=20). It says nothing about whether the architecture's design
  motivation (Section 2.1–2.3 of the proposal) is theoretically sound —
  only that this specific synthetic dataset could not detect an effect.
- Any numeric figure from Steps 1–7 is validation-only and must not be
  cited as evidence for Definition III.1, Proposition III.1, or Theorem
  III.1 themselves. Those should stand or fall on mathematical argument
  and architectural verification (Section 2 above), not on synthetic
  empirical results.

---

## 4. What Must Be Deferred to the Prolific Pilot / Full Study

- Whether real behavioral interaction data actually exhibits
  statistically distinguishable P(f|sⱼ) distributions per stage
  (i.e., whether claim (a)'s JSD-based stage-boundary criterion holds
  empirically at all, outside a synthetic dataset engineered to make it
  trivially true or, as here, structurally muted).
- Whether stage-conditioned attention (as opposed to a stage-blind
  baseline) yields a measurable, generalizable improvement in
  recommendation quality — the central empirical claim of Paper I,
  explicitly reserved for N=50 pilot / N=300 full study data (proposal
  Section 5.3, restated throughout Phase 3 documentation).
- Semantic interpretability of learned attention weights (i.e., whether
  α(category, sⱼ) vs. α(program, sⱼ) tracks any theoretically meaningful
  pattern) — undecidable on synthetic-v3 since its features carry no
  confirmed real-world semantics (Step 6 finding, restated).
- The practical performance of any JSD/MDL/HMM-based stage *detection*
  mechanism (claim (a)'s operationalization) — out of scope for Paper I
  regardless of data source, and belongs to Paper II's validation
  program.

---

## 5. What Should Become Definition III.1

**Recommendation:** Scope Definition III.1 to what Paper I actually
formalizes and has verified architecturally — the stage-conditioned
representation, not the stage-detection criterion.

Proposed shape (to be drafted formally in the next step, not here):

> **Definition III.1 (Stage-Conditioned Item Representation).** Given a
> finite, ordered set of decision stages S = {s₁, ..., s_K} (K=3 in the
> present instantiation), whose members are assumed given — whether
> from ground-truth labels, self-report, or an external stage-detection
> mechanism (cf. Section 5.1's JSD-based criterion, formalized
> separately) — define the stage-conditioned attention weight α(f, sⱼ)
> and item representation q_i(sⱼ) as [the SCA equations, cited from
> `sca_layer.py`, exactly as implemented, including the scale-correction
> and bias-omission already justified in that module].

This keeps Definition III.1 **honest and narrow**: it defines a
*modeling* construct (stage-conditioned representation), explicitly
treats stage membership as an input rather than something Paper I
itself derives, and cross-references (without absorbing) the
statistical stage-boundary definition from Section 5.1 as the
motivating — but separately-scoped — construct that a detection
mechanism (Paper II) would need to satisfy.

**Open question for you to confirm before drafting:** should Definition
III.1 also restate the JSD-based statistical stage definition from
Section 5.1 as a *named*, separate sub-definition (e.g., Definition
III.1 for the statistical stage boundary, Definition III.2 for the
stage-conditioned representation)? I'd lean toward keeping the
statistical definition as an informal motivating paragraph rather than
a numbered Definition in Paper I, since Paper I never operationalizes or
tests it — numbering it as a formal Definition here would invite a
reviewer to ask why Paper I doesn't validate it. That validation
belongs in Paper II.

---

## 6. What Should Become Proposition III.1

**Recommendation:** the one clean, provable, already-verified claim in
the codebase is the **baseline-equivalence property**:

> **Proposition III.1 (Baseline Equivalence).** At initialization (w_base
> = 0, w_stage = 0 for all stages), or under the `freeze_uniform`
> condition, CAFE-LightFM's scoring function r̂(u, i, sⱼ) is identical,
> for every user u, item i, and stage sⱼ, to the stage-blind hybrid
> matrix factorization baseline (Kula, 2015).

This is attractive as Proposition III.1 because:
- It is **analytically provable** (softmax shift-invariance under a
  stage-only bias, plus the |F_i| scale-correction canceling exactly
  against uniform softmax weights of 1/|F_i|), not merely empirically
  observed — the proof sketch already exists in `sca_layer.py`'s
  docstring and needs only formal restatement.
- It is independently **verified by an executable test**
  (`test_equivalence_to_step1_baseline_at_init`), which is a rare and
  valuable property for a dissertation proposition to have — the proof
  and the implementation cannot silently drift apart undetected.
- It is modest: it does not claim CAFE-LightFM is *better* than the
  baseline — only that it strictly *generalizes* it (the baseline is a
  special case), which is a standard and defensible architectural
  contribution claim distinct from any empirical superiority claim.

A secondary candidate — the bias-cancellation argument (why the
proposal's original b_{sⱼ} term is mathematically inert under softmax)
— is better framed as a **Lemma** supporting Proposition III.1's proof,
not as its own numbered Proposition; it is a single-line shift-invariance
argument, not a standalone contribution.

---

## 7. Theorem III.1 — Recommendation: Downgrade From "Theorem"

**My recommendation is a conservative Proposition or an explicitly
labeled Convergence Remark/Sketch — not a Theorem — and I hold this
view with reasonable confidence.**

Reasoning:
- A true convergence *theorem* requires stating and discharging formal
  assumptions (e.g., convexity or specific smoothness/Lipschitz
  conditions, bounded gradients, decreasing step sizes) and proving that
  the training procedure converges to a well-defined target (a
  stationary point, a global optimum, or a bounded regret) under exactly
  those assumptions.
- CAFE-LightFM's training objective (WARP loss, rejection-sampled
  negatives, Adagrad optimizer) is **non-convex** end-to-end: the
  softmax-gated attention couples with the bilinear user–item scoring
  and the max-margin hinge, and WARP's negative sampling introduces a
  *non-smooth, data-dependent* rank-approximation term (the
  `rank_estimate` weighting) that standard SGD/Adagrad convergence
  theorems do not directly cover. Proving a novel convergence theorem
  for this exact combination from first principles would be a
  significant undertaking in its own right, disproportionate to Paper
  I's scope, and risks an unsound or silently-flawed proof if attempted
  without specialist optimization-theory review.
- The academically honest and defensible move is to **cite existing,
  established convergence theory for adaptive subgradient methods**
  (Duchi, Hazan, & Singer, 2011, *JMLR* — the original Adagrad
  regret-bound analysis) as the *grounding* theoretical result, then
  state explicitly, as a **Remark or Proposition (not Theorem III.1)**,
  that: (i) the WARP+SCA objective is non-convex, so only
  stationary-point / bounded-regret-style guarantees inherited from the
  general adaptive-gradient literature apply — not global optimality;
  (ii) no claim is made of a novel convergence result specific to this
  architecture; (iii) empirical training-loss decrease (Steps 5 and 7's
  monotonically decreasing per-epoch losses) is offered as **empirical
  corroboration**, not as evidence discharging the theoretical
  assumptions.

This keeps the dissertation's mathematical claims strictly
falsifiable and avoids the single most common failure mode in
ML-adjacent theory sections: presenting a "Theorem" whose assumptions
are never explicitly checked against the actual non-convex,
non-smooth objective being optimized.

**Suggested labeling:** "Remark III.1 (Convergence Considerations)" or
"Proposition III.2 (Convergence Under Standard Adaptive-Gradient
Assumptions)" rather than "Theorem III.1" — final numbering to be
decided once Definition III.1 / Proposition III.1 are drafted and the
section's overall structure is fixed.

---

## Summary Table

| Item | Recommended Status |
|---|---|
| JSD-based statistical stage-boundary definition (Section 5.1, claim a) | Motivating paragraph, cross-referenced to Paper II — not a numbered Definition in Paper I |
| Stage-conditioned representation (SCA equations) | **Definition III.1** |
| Baseline-equivalence property | **Proposition III.1** (provable + test-verified) |
| Bias-cancellation (shift-invariance) argument | Lemma supporting Proposition III.1's proof |
| Convergence claim | **Remark/conservative Proposition**, NOT "Theorem III.1" — cites Duchi et al. (2011) Adagrad theory, explicitly scoped to non-convexity |

---

## References Used in This Audit

[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
Recommendations. *Proceedings of the 2nd Workshop on New Trends in
Content-Based Recommender Systems*, RecSys 2015. CEUR-WS, Vol. 1448,
14–21.

[2] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient
Methods for Online Learning and Stochastic Optimization. *Journal of
Machine Learning Research*, 12, 2121–2159.
