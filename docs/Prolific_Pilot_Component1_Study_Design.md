# CAFE-LightFM — Prolific Pilot Study Design
## Protocol Component 1 of 6: Study Design & Stage Operationalization

**Project:** CAFE-LightFM (Paper I of III, PhD dissertation: *Stage-Aware
Recommendation Systems: Modeling, Detecting, and Generalizing Preference
Evolution Across Multi-Stage Decision Contexts*)
**Status:** DRAFT — pending sign-off before Components 2–6 (feature schema,
stage-label collection procedure detail, inclusion/exclusion criteria,
evaluation plan, preregistered statistical analysis) are drafted.
**Binding scope (restated):** Stage membership $s_j$ is a given,
experimentally assigned input to CAFE-LightFM (proposal Section 3, Section
7 item 1). Automatic stage detection from unlabeled behavior is explicitly
outside Paper I and reserved for Paper II. The N=50 pilot specified here is
a **feasibility, data-quality, and variance-estimation study** — it does
not license any scientific superiority, necessity, or equivalence claim
about CAFE-LightFM (proposal Section 7, item 4).

---

## 1. Overall Design

**Within-subject, three-session, task-structured design.** Each participant
completes all three decision stages (Exploration, Comparison, Finalization)
in a fixed order, across three separate sessions spread over a 7-day
window (Section 4 below). Stage membership $s_j$ is **experimentally
assigned by session** — session 1 = $S_1$, session 2 = $S_2$, session 3 =
$S_3$ — not inferred from behavior or self-report. This makes $s_j$ a
manipulated independent variable with an unambiguous ground-truth value,
consistent with the "given or externally supplied" stage-input framing in
Sections 3 and 5.3 of the proposal.

**Why within-subject, not between-subject.** A between-subject design
(different participants per stage) would confound stage effects with
between-person preference heterogeneity — exactly the confound
stage-conditioned attention is designed to model *within* a person's
decision journey (proposal Section 2.1). Within-subject tracking is
required to compute the paired statistics (paired t-test, Cohen's $d_z$)
already locked into the evaluation protocol (proposal Section 5.4) and
used throughout Phase 3.

---

## 2. Session-by-Session Task Specification

Each session presents a fictitious university-admission catalog (item
profiles per Component 2, pending) and a task instruction designed to
*induce* — not merely label — the behavioral signature already defined
theoretically in proposal Section 5.1. Sessions form a funnel: each
session's output set is the next session's input set, so narrowing is a
structural property of the task, not a self-reported impression.

### Session 1 — Exploration ($S_1$)
- **Catalog shown:** full profile set (proposed N=15–20 fictitious
  institutions, Component 2).
- **Task instruction:** "Browse the full list of institutions freely.
  Build a shortlist of at least 5 institutions you would seriously
  consider applying to." No comparison tool is shown; only individual
  profile views, to encourage breadth over depth.
- **Induced signature (per Section 5.1 definition of $S_1$):** broad
  feature scanning, high item variety touched, low revisitation, shallow
  per-item depth.
- **Session output:** participant's shortlist (≥5 items), carried forward
  as the Session 2 catalog.

### Session 2 — Comparison ($S_2$)
- **Catalog shown:** the participant's own Session 1 shortlist only
  (re-presented, not the full catalog).
- **Task instruction:** "Using the side-by-side comparison tool, compare
  your shortlisted institutions directly on specific features. Narrow
  your shortlist down to your top 3."
- **Induced signature:** focused feature analysis on a fixed, small set;
  high revisitation (comparison tool encourages repeated feature
  lookups); explicit comparative behavior (the tool logs feature-pair
  comparisons directly).
- **Session output:** participant's top-3 set, carried forward as the
  Session 3 catalog.

### Session 3 — Finalization ($S_3$)
- **Catalog shown:** the participant's own top-3 set from Session 2 only.
- **Task instruction:** "From your top 3, make your final decision.
  Rank all three, then select the one institution whose offer you would
  accept — considering funding certainty and visa outcome as final
  deciding factors."
- **Induced signature:** concentrated attention on a small set;
  asymmetric weighting toward risk-relevant attributes (explicitly
  prompted); elimination/ranking behavior.
- **Session output:** final rank + single accept decision.

**Labeled assumption.** The funnel structure (shortlist → top-3 → final
choice) assumes participants can meaningfully narrow at each step without
the task feeling artificially constrained by session boundaries alone.
This should be checked against the manipulation-check data (Section 3
below) in the pilot before the full study; if agreement rates are poor,
the task structure — not just the labeling procedure — may need revision.

---

## 3. Hybrid Stage-Labeling Protocol

- **Primary label (ground truth):** the session number itself
  ($s_j \in \{S_1, S_2, S_3\}$ = session $j$), assigned by experimental
  design, used directly as CAFE-LightFM's stage input. This is the label
  used in all model training and evaluation.
- **Self-report (manipulation check only, not a label source).** At the
  end of each session, participants complete a short 3-item instrument
  (5-point Likert) asking the extent to which they felt they were (a)
  broadly exploring many options, (b) closely comparing a shortlisted
  set, (c) making a final, consequential decision. This is a
  **manipulation check**, following standard experimental-psychology
  practice for verifying that a task instruction produced its intended
  psychological state — it is never used to relabel or override the
  session-assigned $s_j$.
- **Use of the manipulation check:** (i) reported descriptively in the
  pilot as a feasibility outcome (Section 5); (ii) used to flag, not
  relabel, sessions where the self-reported state clearly contradicts the
  assigned stage (e.g., a Session 1 respondent scoring maximally on the
  "final decision" item) as a candidate data-quality exclusion, per the
  exclusion criteria to be specified in Component 4.
- **Explicit scope boundary.** This procedure keeps automatic stage
  inference fully outside Paper I: $s_j$ is never computed *from*
  behavior or self-report in this design. It is assigned before the
  participant sees the task.

---

## 4. Timing and Session Duration

| Session | Stage | Target day (of 7-day window) | Estimated duration |
|---|---|---|---|
| 1 | Exploration ($S_1$) | Day 1 | 12–15 min |
| 2 | Comparison ($S_2$) | Day 3–4 | 8–10 min |
| 3 | Finalization ($S_3$) | Day 6–7 | 6–8 min |

**Rationale for multi-day spacing (not same-day/back-to-back).** A
same-day administration of all three sessions would not exercise the
multi-session, multi-day decision horizon that motivates this research in
the first place (proposal Section 2.2 — high-stakes decisions unfold over
days/weeks, not minutes). Spacing sessions across a week also more
faithfully induces the "returning to a narrowed set after a gap"
experience characteristic of real admissions decisions, and creates a
natural test of whether attention patterns are stable or drift across
the *session* dimension independent of the *stage* dimension (a useful
diagnostic, though not the primary research question of Paper I).

**Total participant time budget:** ≈ 30 minutes across the full week,
relevant for Prolific compensation-rate calculation (Component 4/6, not
finalized here).

---

## 5. Unit of Analysis and Pilot Feasibility Outcomes

**Primary unit of analysis:** participant × session × item interaction
record. This is deliberately schema-compatible with the existing
`InteractionRecord` structure used throughout Phase 3's synthetic-v3
pipeline (zero re-implementation carries forward: the same NDCG/SW-NDCG/
Precision computation code should consume real Prolific records without
modification, once the feature schema — Component 2 — is finalized).

**The N=50 pilot is scoped as a feasibility study.** Its outcomes are
data-quality and design-validation metrics, **not** a significance test
of RQ1. Recommended pilot feasibility outcomes:

| Outcome | What it checks | Illustrative threshold (to confirm) |
|---|---|---|
| 3-session completion rate | Attrition across the 7-day window | ≥ 80% complete all 3 sessions |
| Manipulation-check agreement rate | Task-induced stage matches self-reported experience (Section 3) | ≥ 70% per session |
| Session duration distribution (median, IQR) | Feasibility of time estimates above; detects rushed/speeding responses | Flag sessions < 2 min (implausibly fast) |
| Attention-check failure rate | Data-quality screening (detail in Component 4) | To be set alongside exclusion criteria |
| Variance of SW-NDCG@K / NDCG@K per stage | **Primary use of the pilot**: empirical variance estimate to power the N=300 full-study sample-size justification (proposal Section 5.3) | N/A — descriptive input to power analysis |
| Platform/technical error rate | Data platform robustness before full-scale deployment | To confirm |

**Explicit non-goal.** The pilot is not designed or powered to detect the
Full-vs-baseline or Full-vs-ablation effects targeted by RQ1; those
significance tests are reserved for the N=300 full study per the binding
scope statement (proposal Section 7, item 4). Any effect-size or p-value
computed on N=50 pilot data should be reported, if at all, as a variance
estimate for power-analysis purposes, explicitly labeled as
non-confirmatory.

---

## 6. Open Items (feed into Components 2–6)

1. Exact catalog size and feature list per profile → **Component 2
   (Feature Schema)**.
2. Full wording and validation of the 3-item manipulation-check
   instrument → **Component 3 (Stage-Label Collection Procedure)**.
3. Concrete attention-check items, speeding thresholds, and
   manipulation-check-disagreement exclusion rule → **Component 4
   (Inclusion/Exclusion Criteria)**.
4. Mapping this design onto the existing NDCG@K / SW-NDCG@K / Precision@K
   protocol (proposal Section 5.4) for real data → **Component 5
   (Evaluation Plan)**.
5. Formal power analysis using pilot variance estimates, and the
   preregistration document itself → **Component 6 (Preregistered
   Statistical Analysis)** — to be written last, after Components 1–5 are
   fixed, per standard preregistration practice.

---

*No new code is required to act on this document. This is a design
artifact; implementation (data collection platform, task UI) begins only
after Components 1–6 are collectively confirmed.*