# Paper I — Theoretical Framework (Draft)
**CAFE-LightFM: A Stage-Conditioned Attention Framework for Hybrid Matrix Factorization**

**Status:** Standalone draft, approved in substance (2026-07-08).
Consolidates Definition III.1, Proposition III.1, and Remark III.1, as
reviewed and approved section-by-section. Section III's scope is
closed at these three items — WARP is deliberately excluded as a
separate theoretical Definition and instead belongs to the Methodology
section (citation + procedural description only). Pending final
integration into the full dissertation manuscript.

**Scope note:** This document formalizes stage-*conditioned* preference
modeling only. It does not formalize or validate stage *detection*
(the JSD/MDL/HMM-based mechanism for inferring stage boundaries from
raw interaction data); that is addressed separately in Paper II of this
dissertation. See `Paper1_Theoretical_Framework_Audit_Note.md` for the
full scoping rationale.

---

## III. Theoretical Framework

Multi-stage decision processes are characterized, at a descriptive
level, by systematic shifts in which item features a user attends to as
the decision progresses. One way to make this notion precise is
statistically: a decision stage can be characterized as an interval of
a user's interaction trajectory over which the empirical distribution
of feature-attention weights is distinguishable from adjacent intervals
— for instance, via Jensen–Shannon divergence between per-interval
attention distributions. This statistical characterization is a
natural and useful *motivating* picture of what a "stage" is, and it
foreshadows a distinct research question — *how should stage
boundaries be detected from raw behavioral signal?* — that this
dissertation addresses separately (stage-transition detection is the
subject of Paper II of this dissertation). **The present paper does
not formalize** a stage-detection procedure, and Definition III.1
below does not depend on one: it assumes stage membership is *given*,
however it may have been obtained (ground-truth simulation label,
participant self-report, or an external detector), and formalizes only
how a recommendation model should represent an item once a stage
assignment is available.

### Notation

- $U$, $I$: the sets of users and items, respectively.
- $S = \{s_1, \dots, s_K\}$: a finite, ordered set of decision stages
  ($K = 3$ in the present instantiation: exploration, comparison,
  finalization).
- $F_i$: the set of metadata features associated with item $i \in I$.
  In the present instantiation, $F_i = \{\text{category}(i),
  \text{program}(i)\}$ for every item (i.e., $|F_i| = 2$ uniformly);
  item identity itself is deliberately excluded from $F_i$, since
  attention over a single item-unique embedding has no meaningful
  interpretation as feature salience.
- $e_f \in \mathbb{R}^d$: the learned embedding of feature $f$, for
  embedding dimension $d$.
- $e_i \in \mathbb{R}^d$: the learned embedding of item $i$
  (item-identity embedding, distinct from any feature embedding).
- $w_{\text{base}} \in \mathbb{R}^d$: a single learned vector, shared
  across all stages.
- $w_{s_j} \in \mathbb{R}^d$: a learned vector specific to stage
  $s_j$, for each $s_j \in S$.

---

### Definition III.1 (Stage-Conditioned Item Representation)

Let $s_j \in S$ be a decision stage, assumed given for the purpose of
this definition (see the motivating discussion above; membership
itself is not derived here). For an item $i \in I$ with metadata
feature set $F_i$, define the **stage-conditioned attention weight**
of feature $f \in F_i$ under stage $s_j$ as

$$
\alpha(f, s_j) \;=\; \frac{\exp\big(w_{\text{base}} \cdot e_f + w_{s_j} \cdot e_f\big)}{\displaystyle\sum_{f' \in F_i} \exp\big(w_{\text{base}} \cdot e_{f'} + w_{s_j} \cdot e_{f'}\big)},
$$

i.e., a softmax over $F_i$ of a logit formed additively from a
stage-invariant term ($w_{\text{base}}$) and a stage-specific term
($w_{s_j}$), so that $\alpha(\cdot, s_j)$ is, for every $s_j$, a valid
probability distribution over $F_i$ ($\sum_{f \in F_i} \alpha(f, s_j) =
1$, $\alpha(f, s_j) \ge 0$).

Define the **stage-conditioned item representation** as

$$
q_i(s_j) \;=\; e_i \;+\; |F_i| \sum_{f \in F_i} \alpha(f, s_j)\, e_f.
$$

The scale factor $|F_i|$ is a deliberate correction, not an arbitrary
choice: since $\alpha(\cdot, s_j)$ sums to $1$ rather than $|F_i|$, an
uncorrected weighted sum would shrink the metadata contribution by a
factor of $|F_i|$ relative to a stage-blind hybrid matrix-factorization
representation of the same item (cf. Proposition III.1 below), even
under a uniform (maximally uninformative) attention distribution. The
correction restores exact correspondence with the stage-blind case at
that boundary condition.

The full stage-conditioned recommendation score for user $u \in U$,
item $i \in I$, under stage $s_j$, is then

$$
\hat r(u, i, s_j) \;=\; e_u \cdot q_i(s_j) \;+\; b_u \;+\; b_i,
$$

where $e_u \in \mathbb{R}^d$ is the (stage-invariant) user embedding
and $b_u, b_i \in \mathbb{R}$ are scalar user/item bias terms.

**Remark (scope of this definition).** Definition III.1 formalizes
*how a fixed model represents an item conditional on a stage label*;
**the present paper does not formalize** a criterion for determining
$s_j$ from raw interaction data. The latter question — whether, and
how, stage membership can be *inferred* from behavioral signal (e.g.,
via a distributional-divergence criterion) — is addressed separately,
**in Paper II of this dissertation, which formalizes and validates the
stage-transition-detection component.**

**Remark (no bias term).** An earlier formulation of this framework
additionally included a stage-only additive bias term $b_{s_j}$ in the
logit. That term is omitted here: because $b_{s_j}$ does not depend on
$f$, it is constant across the softmax's normalization axis for fixed
$s_j$, and softmax is invariant to the addition of any
feature-independent constant to all of its input logits. The term
therefore has no effect on $\alpha(\cdot, s_j)$ as defined, and its
omission changes nothing about the model's expressive capacity.

---

### Proposition III.1 (Baseline Equivalence)

**Recalled definition (stage-blind hybrid matrix factorization).**
Following Kula (2015) [1], define the stage-blind hybrid MF
representation of item $i$ as

$$
q_i^{\text{base}} \;=\; e_i \;+\; \sum_{f \in F_i} e_f,
$$

with corresponding score $\hat r^{\text{base}}(u,i) = e_u \cdot
q_i^{\text{base}} + b_u + b_i$. This is stage-invariant by construction
— it does not depend on $s_j$ at all.

**Statement.** Let $q_i(s_j)$ be the stage-conditioned item
representation of Definition III.1. If $w_{\text{base}} = \mathbf{0}$
and $w_{s_j} = \mathbf{0}$ for every $s_j \in S$, then for every item
$i \in I$ and every stage $s_j \in S$,

$$
q_i(s_j) \;=\; q_i^{\text{base}}.
$$

Consequently, $\hat r(u, i, s_j) = \hat r^{\text{base}}(u, i)$ for every
user $u$, item $i$, and stage $s_j$ — i.e., CAFE-LightFM's
stage-conditioned scoring function is, at this parameter setting,
identical to the stage-blind hybrid MF baseline for every input.

**Proof sketch.** Fix any item $i$ and stage $s_j$, and let $n =
|F_i|$.

*Step 1 — the logits are identically zero.* Under $w_{\text{base}} =
\mathbf 0$ and $w_{s_j} = \mathbf 0$, the logit for every $f \in F_i$
is
$$
w_{\text{base}} \cdot e_f + w_{s_j} \cdot e_f = 0 \cdot e_f + 0 \cdot e_f = 0,
$$
regardless of $e_f$ (i.e., regardless of the embedding values
themselves — this holds for *any* learned feature embeddings, not
only at their own initialization).

*Step 2 — softmax of identical logits is uniform.* Applying the
softmax definition of $\alpha(\cdot, s_j)$ to a constant vector of
logits (all equal to $0$) yields
$$
\alpha(f, s_j) = \frac{\exp(0)}{\sum_{f' \in F_i} \exp(0)} = \frac{1}{n} \quad \text{for every } f \in F_i.
$$

*Step 3 — the scale correction exactly cancels the uniform weight.*
Substituting into the definition of $q_i(s_j)$:
$$
q_i(s_j) = e_i + n \sum_{f \in F_i} \frac{1}{n}\, e_f = e_i + \sum_{f \in F_i} e_f = q_i^{\text{base}}.
$$

*Step 4 — the score follows immediately.* Since $q_i(s_j) =
q_i^{\text{base}}$ for every $s_j$, and the user embedding $e_u$ and
bias terms $b_u, b_i$ are themselves stage-invariant by construction
(Definition III.1), $\hat r(u,i,s_j) = e_u \cdot q_i(s_j) + b_u + b_i =
e_u \cdot q_i^{\text{base}} + b_u + b_i = \hat r^{\text{base}}(u,i)$
for every $u$, $i$, $s_j$. $\blacksquare$

**Remark (nature of the claim).** This is a statement about the model
*class* — it establishes that stage-blind hybrid MF is a special case
of CAFE-LightFM, reachable at a specific, easily-identified parameter
setting, and that this equivalence holds exactly (not approximately)
for arbitrary feature embeddings, not merely at random initialization.
No claim of empirical superiority is made or implied; the proposition
establishes architectural generality, not performance. This proof is
exact and requires no distributional or dataset assumptions, in
contrast to the convergence considerations discussed next in Remark
III.1, which does require standard optimization-theoretic assumptions.

**Remark (independent verification).** This proposition's claim is
additionally confirmed by an executable regression test
(`test_equivalence_to_step1_baseline_at_init`), so the proof and the
implementation cannot silently diverge without a test failure — a form
of **implementation-level regression verification**, though the proof
itself stands independently of that test.

---

### Remark III.1 (Convergence Considerations)

CAFE-LightFM is trained by minimizing the Weighted Approximate-Rank
Pairwise (WARP) loss (Weston, Bengio, & Usunier, 2011 [2]) via the
Adagrad optimizer (Duchi, Hazan, & Singer, 2011 [3]), following the
training procedure described in the Methodology section. It is natural
to ask what, if anything, can be said theoretically about this
procedure's convergence behavior. This remark states explicitly and
conservatively what is — and is not — claimed.

**What grounds this discussion.** Duchi et al. (2011) [3] establish
regret and convergence guarantees for adaptive subgradient methods
(including Adagrad) in online convex optimization, under standard
assumptions: convexity of the objective in the optimization variables,
and bounded (sub)gradients. These results are well-established and are
the appropriate theoretical reference point for any optimizer-level
discussion of this training procedure.

**Why those assumptions do not transfer directly here.** The
WARP+SCA training objective, as actually optimized in this work, does
not satisfy the convexity assumption underlying [3]. Three sources of
non-convexity/non-smoothness are identifiable in the objective as
implemented:

1. The bilinear scoring function $\hat r(u,i,s_j) = e_u \cdot
   q_i(s_j) + b_u + b_i$ is a product of jointly-learned embeddings
   ($e_u$, item and feature embeddings feeding into $q_i(s_j)$), which
   is non-convex in those parameters jointly, as is standard for
   essentially all matrix-factorization-style objectives.
2. The stage-conditioned attention weight $\alpha(f,s_j)$ (Definition
   III.1) is itself a softmax of a bilinear form in $w_{\text{base}}$,
   $w_{s_j}$, and the feature embeddings — introducing further
   non-convex coupling between the attention parameters and the
   embeddings they attend over.
3. The WARP loss's rank-approximation weighting (`rank_estimate`,
   derived from the number of rejection-sampling trials until a
   margin-violating negative is found) is a **data-dependent,
   non-differentiable, non-smooth** term with respect to the model
   parameters — it depends on a discrete sampling procedure, not a
   closed-form differentiable function of the scores.

**What is therefore claimed, and what is not:**

- **No novel convergence theorem is claimed** for the WARP+SCA
  training procedure. No proof is offered, attempted, or implied that
  this specific non-convex, non-smooth objective converges to any
  particular target under Adagrad.
- **No global optimality claim is made.** Nothing here asserts, or
  should be read as asserting, that training reaches a global minimum
  of the WARP+SCA objective, or any specific stationary point.
- **The objective is acknowledged as non-convex and non-smooth**, for
  the three reasons given above, and the established adaptive-gradient
  convergence theory cited (Duchi et al., 2011 [3]) is offered only as
  the relevant background theory for the optimizer used — not as a
  guarantee applicable to this specific, more complex objective.
- **The observed monotonic decrease in per-epoch training loss**
  (documented in the Methodology/Experiments sections, e.g., the Step
  5 and Step 7 training logs) is offered strictly as **empirical
  corroboration** that the optimization procedure behaves reasonably
  in practice on the datasets examined so far — it is not, and is not
  presented as, a substitute for or evidence toward a formal
  convergence proof.

This remark is deliberately conservative: establishing a rigorous
convergence theory for attention-gated matrix factorization trained
under a rank-based, sample-dependent ranking loss is a substantial
optimization-theory undertaking in its own right, outside the scope of
this paper's contribution, and is not attempted here.

---

## References

[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
Recommendations. *Proceedings of the 2nd Workshop on New Trends in
Content-Based Recommender Systems*, RecSys 2015. CEUR-WS, Vol. 1448,
14–21.

[2] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to
Large Vocabulary Image Annotation. *Proceedings of the 22nd
International Joint Conference on Artificial Intelligence (IJCAI)*,
2764–2770.

[3] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient
Methods for Online Learning and Stochastic Optimization. *Journal of
Machine Learning Research*, 12, 2121–2159.

---

## Open Items for Next Session

- **Resolved (2026-07-08):** Section III does not include a separate
  formal Definition of the WARP training objective. WARP is cited to
  Weston, Bengio, & Usunier (2011) [2] and is described/equationed in
  the Methodology section as a training procedure, not restated here as
  a numbered theoretical construct — Section III is scoped to the three
  claims that are this paper's own architectural/theoretical
  contribution: Stage-Conditioned Item Representation (Definition
  III.1), Baseline Equivalence (Proposition III.1), and Convergence
  Considerations (Remark III.1).
- **Section III is considered complete** on this basis, pending final
  integration into the full dissertation manuscript (below).
- This draft has not yet been checked against the dissertation's
  overall notation conventions used elsewhere (if any exist outside
  this section) for consistency.
- Not yet integrated into the full manuscript document; remains a
  standalone draft pending your review.
