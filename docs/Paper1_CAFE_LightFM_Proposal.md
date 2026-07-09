الورقة األولى مشروع بحثي— _Research Project Proposal — Paper I_

## **CAFE-LightFM: A Stage-Conditioned Attention Framework for Hybrid Matrix Factorization**

_**Modeling Preference Evolution in Multi-Stage Recommendation Systems**_

**Target Journal: Research Domain: Role in Dissertation:**

IEEE Transactions on Knowledge and Data Engineering (IEEE TKDE) Recommender Systems — Machine Learning — Information Retrieval Paper I of III — Core Technical Contribution

> **Revision note (this version):** Sections 3, 4.1, and 5.2 have been revised to formally narrow the scope of Paper I, to remove a stage-bias term not present in the implemented model, and to align the mathematical formulation with the verified `sca_layer.py` implementation (Phase 3, Steps 6–7). Sections 5.5 and 6 have been updated for consistency with this narrower scope, and a new Section 7 ("Limitations and Scope Boundaries") has been added. The former Section 7 ("Key References") is now Section 8. No experimental results are affected by this revision; it is a scope and notation correction only.

## **1. Scientific Background and Literature Context**

Recommender systems have become indispensable infrastructure across domains ranging from e-commerce and digital media to education and professional development. The predominant paradigm in collaborative filtering — anchored by foundational techniques such as Matrix Factorization (MF) — rests on a critical assumption: that user preferences can be adequately represented as static latent vectors learned from historical interaction data. This assumption, while computationally convenient, systematically fails to capture a fundamental characteristic of human decision-making: preferences are not static; they evolve dynamically across the stages of a decision process.

The evolution of recommendation methodology has proceeded through several phases. Early collaborative filtering approaches relied on explicit ratings and nearest-neighbor heuristics. Matrix factorization methods, introduced and popularized through the Netflix Prize competition, decomposed the user-item interaction matrix into low-dimensional latent spaces, achieving significant accuracy improvements. Hybrid matrix factorization — exemplified by the LightFM framework — extended these methods by incorporating user and item metadata as feature embeddings, enabling generalization to cold-start scenarios. However, throughout this evolution, the assumption of static user preference representations has remained largely unchallenged.

The introduction of attention mechanisms from the natural language processing literature, particularly the Transformer architecture, catalyzed a new generation of sequential recommendation models. Systems such as SASRec and BERT4Rec demonstrated that dynamically weighting historical interactions according to their relevance to a current context improves recommendation quality. Nevertheless, these models conceptualize attention within the temporal dimension — weighting items by their recency or relevance — rather than within the decision-stage dimension, which is the dimension most relevant to complex, multi-stage decision processes.

Multi-stage decision contexts — university selection, career transitions, real estate, hotel booking — exhibit a characteristic pattern of preference evolution that fundamentally differs from the sequential consumption behaviors modeled by existing systems. In these contexts, users progress through identifiable decision stages: an exploratory phase characterized by broad feature scanning, a comparative phase characterized by focused feature analysis across shortlisted options, and a finalization phase characterized by risk-weighted evaluation of a small candidate set. Critically, the salience of individual item features shifts substantially across these stages. A user evaluating international universities will weight geographic region and program reputation heavily during exploration, shift attention to tuition fees and scholarship availability during comparison, and prioritize visa success rates and supervisor compatibility during finalization. No existing recommendation model captures this stage-conditioned feature sensitivity shift.

This research addresses this gap through the development of CAFE-LightFM — a Stage-Conditioned Attention Framework for hybrid matrix factorization — which extends the LightFM model with a formal mechanism for conditioning feature attention weights on the user's current decision stage. The contribution is simultaneously theoretical, in providing a formal definition of decision stages within the recommendation context, and empirical, in testing whether stage-conditioned feature weighting yields measurable improvements in recommendation quality, given decision-stage membership as an input (Section 3).

## **2. Research Significance and Motivation**

The significance of this research derives from three converging gaps in the current literature, each of which represents both a theoretical limitation and a practical deficiency.

## **2.1 Theoretical Gap: The Static Preference Assumption**

The static preference assumption — that a user's latent factor vector adequately represents their preferences regardless of where they are in a decision journey — is not merely a simplification but an error. Empirical studies of user behavior in high-stakes decision contexts consistently demonstrate that the features users attend to change substantially across decision stages. Matrix factorization models that learn a single preference vector per user conflate preferences that are stage-specific, producing representations that are accurate averages but poor predictors at any specific decision moment. This research provides a formal framework for decomposing user preference representations along the decision-stage dimension within a hybrid matrix factorization architecture.

## **2.2 Practical Gap: High-Stakes Domain Neglect**

The recommendation systems literature has historically concentrated on entertainment and consumer goods — domains where decision stakes are low and preferences are relatively stable. High-stakes domains — international university selection, career transitions, major travel decisions — involve longer decision horizons, higher feature sensitivity, greater preference evolution, and more severe consequences of misalignment between recommendations and user needs. These domains are both scientifically underexplored and practically important. The present research provides a framework specifically designed for multi-stage high-stakes recommendation, with primary validation in the international university admission domain.

## **2.3 Methodological Gap: Feature Weighting Without Stage Conditioning**

Existing attention-based recommendation models assign dynamic weights to items or interactions, but do not condition these weights on the user's current decision stage. The distinction is fundamental: attending to the most recently interacted item is categorically different from attending to the features most relevant to a user who is currently engaged in comparative evaluation. CAFE-LightFM introduces stage-conditioned feature attention as a formally defined mechanism, grounded in a probabilistic definition of decision stages derived from behavioral signals, providing a methodological contribution applicable beyond the specific domains validated in this research.

## **3. Problem Statement**

Current hybrid matrix factorization models assign static attention weights to user and item features across all phases of a decision process, failing to capture the systematic variation in feature salience that characterizes multi-stage decision-making. This limitation manifests in two concrete deficiencies:

First, models trained on interaction data aggregated across decision stages learn preference representations that average across stage-specific signals, diluting the predictive power of features that are highly discriminative at specific stages. A university recommendation model that pools feature interactions from exploration, comparison, and finalization stages will underweight features that are critical at finalization but infrequently examined during exploration.

Second, the absence of stage conditioning means that recommendation quality degrades precisely when it is most consequential — at the finalization stage, when users are making actionable choices. A model that cannot distinguish a user in exploratory mode from a user in decision mode will generate recommendations appropriate for neither.

**Scope boundary (binding).** Paper I does not propose or evaluate a mechanism for automatically detecting decision-stage boundaries from raw, unlabeled behavioral signals; that contribution is reserved for Paper II of this dissertation, which formalizes stage-boundary detection via distributional-divergence thresholds and probabilistic sequence inference over interaction logs. Paper I's empirical claim concerns exclusively the value of conditioning feature attention on decision stage *once stage membership is available* — whether through explicit self-report (as collected in the Prolific protocol, Section 5.3) or externally supplied inference. The formal theoretical definition of what constitutes a decision stage (Section 5.1) remains a Paper I contribution; only the automatic algorithmic detection of stage boundaries from unlabeled logs is deferred to Paper II.

The formal research problem is accordingly stated as follows:

_Given a user u whose current decision stage sⱼ ∈ D = {s₁, s₂, s₃} is observed, explicitly labeled, or externally inferred, and an item set I characterized by a feature space F, how can a hybrid matrix factorization model be extended to condition feature attention weights on sⱼ, such that the resulting stage-specific recommendation scores are more accurately aligned with the user's stage-conditioned preferences than those produced by a stage-agnostic hybrid MF baseline sharing an otherwise identical factorization backbone?_

**Central Research Question (RQ1):** *Does conditioning feature-attention weights on decision stage improve hybrid matrix factorization recommendation quality relative to a stage-agnostic hybrid MF baseline, under a matched factorization backbone — i.e., identical user/item latent-factor architecture, bias terms, and controlled training protocol across the compared models?*

This problem statement encapsulates two research sub-problems, narrowed from the dissertation's broader three-part formulation (modeling, detecting, generalizing) to reflect the scope boundary above:

1. The architectural design of a stage-conditioned feature-attention mechanism compatible with hybrid matrix factorization, taking stage membership sⱼ as a given input rather than inferring it.
2. The empirical validation of stage-conditioned recommendation quality against a stage-agnostic hybrid MF baseline with an identical factorization backbone, isolating the contribution of stage-conditioned attention through controlled ablation (Section 5.6) and paired statistical testing (Section 5.4).

## **4. Research Objectives**

This research pursues the following primary and secondary objectives:

## **4.1 Primary Objectives**

1. To formally define decision stages within the recommendation context as statistically distinguishable intervals of feature-attention distribution, grounded in Jensen-Shannon divergence between stage-conditional attention distributions (Section 5.1). This is a definitional and theoretical contribution only; the design of an algorithm to automatically detect stage boundaries from unlabeled behavioral logs is outside the scope of Paper I and is reserved for Paper II.

2. To design and implement CAFE-LightFM: a Stage-Conditioned Attention Framework that extends the LightFM hybrid matrix factorization architecture with a feature-attention mechanism conditioned on a given or externally supplied decision-stage label sⱼ, dynamically reweighting item-side feature embeddings via α(f, sⱼ) rather than inferring sⱼ itself from raw behavior.

3. To empirically test, via controlled ablation and paired significance testing, whether CAFE-LightFM's stage-conditioned attention improves recommendation quality relative to a stage-agnostic hybrid MF baseline with an identical factorization backbone — first as a pipeline-validation exercise on synthetic data, and subsequently, as the sole basis for any scientific superiority claim, on the primary Prolific dataset from the international university admission domain (Section 5.3).

## **4.2 Secondary Objectives**

4. To conduct ablation studies isolating the contribution of each architectural component — stage conditioning, attention mechanism, and hybrid feature integration — to overall model performance.

5. To analyze the interpretability of learned stage-conditioned attention weights, examining whether the model captures theoretically expected patterns of feature salience across decision stages, subject to the constraints of available item metadata (see Section 7).

6. To establish a reproducible experimental protocol for evaluating recommendation quality in multi-stage decision contexts, including stage-stratified performance metrics suitable for adoption by subsequent research.

## **5. Proposed Methodology**

## **5.1 Theoretical Framework — Formal Stage Definition**

Decision stages are formally defined as intervals within a user's decision trajectory characterized by statistically distinguishable patterns of feature interaction. Specifically, stage sⱼ is defined as a period during which the distribution of feature attention weights P(f | sⱼ) is significantly different from the distributions characterizing adjacent stages, as measured by Jensen-Shannon divergence. **Within Paper I, Jensen-Shannon divergence is used exclusively as a theoretical, definitional criterion for what constitutes a decision stage; no JSD-based (or other) automatic stage-detection algorithm is implemented, trained, or evaluated in this paper.** This definition grounds stage boundaries in observable behavioral data rather than subjective characterization. **Operationalizing this definition into an automatic detection algorithm that infers stage boundaries directly from unlabeled interaction logs is the contribution of Paper II; within Paper I, stage membership is treated as a given or externally supplied input to the model described in Section 5.2.**

Three primary decision stages are defined for high-stakes multi-stage decisions, consistent with established behavioral decision theory literature:

- Exploration Stage (s₁): Characterized by broad feature scanning, high item variety, low revisitation rate, and shallow feature depth. Users are constructing their consideration set.

- Comparison Stage (s₂): Characterized by focused feature analysis on a shortlisted item set, high revisitation rate, deep feature engagement, and explicit comparative behavior. Users are evaluating relative item quality.

- Finalization Stage (s₃): Characterized by concentrated attention on a small item set, asymmetric feature weighting toward high-risk attributes, and elimination behavior. Users are making terminal decisions.

## **5.2 Model Architecture — CAFE-LightFM**

CAFE-LightFM extends the LightFM hybrid matrix factorization framework through the introduction of a Stage-Conditioned Attention (SCA) layer. The baseline LightFM model represents each user u and item i as the sum of latent representations of their associated features:

$$p_u = \sum_{f \in F_u} e_f, \qquad q_i = \sum_{f \in F_i} e_f$$

where $e_f$ denotes the embedding of feature $f$, and $F_u$, $F_i$ denote the feature sets of user $u$ and item $i$ respectively. The baseline prediction score is:

$$\hat{y}(u,i) = b_u + b_i + \langle p_u, q_i \rangle$$

where $b_u$ and $b_i$ are user and item bias terms.

CAFE-LightFM modifies this architecture by replacing the uniform item-side feature summation with a stage-conditioned weighted attention mechanism:

$$q_i^{(s)} = \sum_{f \in F_i} \alpha(f, s) \cdot e_f$$

where $\alpha(f, s)$ is the stage-conditioned attention weight of feature $f$ given decision stage $s$, computed as:

$$\alpha(f, s) = \mathrm{softmax}_f\big( W_{\text{base}} \cdot e_f + W_{s} \cdot e_f \big)$$

Here, $W_{\text{base}}$ is a stage-invariant weight matrix and $W_{s}$ is a stage-specific learned weight matrix. **This formulation contains no additive bias term inside the softmax argument, and no separate stage-level bias term in the final prediction score.** An earlier version of this proposal included stage-specific bias vectors $b_{s}$ both inside the attention computation and as an additive term in the prediction score; both have been removed in this revision to match the verified `sca_layer.py` implementation (Phase 3, Steps 6–7), which contains no such parameters. A published equation must correspond to an implemented and evaluated module (see Section 7); the model therefore conditions stage exclusively through the feature-attention weights $\alpha(f,s)$, not through any bias pathway.

The final prediction score is:

$$\hat{y}(u,i,s) = b_u + b_i + \langle p_u, q_i^{(s)} \rangle$$

Expanding $q_i^{(s)}$ by linearity of the inner product, and defining the per-feature interaction term as $g_f(u,i) := \langle p_u, e_f \rangle$, this is exactly equal to:

$$\hat{y}(u,i,s) = b_u + b_i + \left\langle p_u, \sum_{f \in F_i} \alpha(f,s)\,e_f \right\rangle = b_u + b_i + \sum_{f \in F_i} \alpha(f,s) \cdot g_f(u,i)$$

This is an exact algebraic identity — not an approximation — given the stated definitions of $q_i^{(s)}$ and $g_f(u,i)$; it follows directly from linearity of the dot product and requires no additional assumption about the model. The second form is retained in preference over the first because it isolates the stage-conditioned attention term $\sum_f \alpha(f,s)\cdot g_f(u,i)$ as the model's explicit point of departure from stage-agnostic hybrid MF, directly supporting RQ1 (Section 3).

**Verification note (pending, labeled assumption).** The identity above holds for the mathematical model as specified. Confirming that it also holds for the literal forward-pass computation in `sca_layer.py` / `warp_loss.py` — i.e., that no additional normalization, clipping, or auxiliary term is applied between the weighted-sum step ($q_i^{(s)}$) and the scoring step ($\langle p_u, q_i^{(s)}\rangle$) — is a source-code verification step to be completed before this equation is presented as implementation-exact in the manuscript. Until that check is performed, the equivalence above should be described as a property of the mathematical formulation, not as an already-confirmed property of the running code.

**Implementation scope note (labeled assumption).** The formulation above is general over the item-side feature set $F_i$. The CAFE-LightFM implementation verified in Phase 3 (Step 6 source inspection of `models/cafe_lightfm/sca_layer.py`) restricts $F_i$ to exactly two item-side metadata features, `category` and `program`; attention is not computed over user-side features or the full item catalog. This is a current implementation scope, not a limit of the general formulation above — richer feature sets (e.g., from Prolific data with real item metadata) are expected to populate $F_i$ more fully without requiring a change to the SCA equations themselves. Rationale: the two-feature scope was the feature set available in the synthetic-v3 validation dataset; it has not yet been tested against a larger feature set, so this is flagged as an assumption carrying forward into Prolific data collection rather than a settled design choice.

Model training employs the Weighted Approximate-Rank Pairwise (WARP) loss, inherited from the LightFM framework, which optimizes for ranking quality rather than rating prediction accuracy — appropriate for implicit feedback settings characteristic of behavioral data.

## **5.3 Data Collection Platform**

A dedicated behavioral research platform will be developed to collect multi-stage interaction data from participants engaging in a simulated international university selection task. The platform will record timestamped behavioral events at fine granularity, including feature hover events with duration, item comparison activations, shortlist modifications, and explicit stage self-reports collected at session initiation. Behavioral signals will be used to construct both the training dataset for CAFE-LightFM and the ground-truth stage labels used to condition and evaluate the model — these explicit self-reported labels are the "given or externally supplied" stage input referenced in Sections 3 and 4.1; they are not produced by an automatic detection algorithm within Paper I.

Target sample: 300–400 participants, recruited from international student communities at the host institution and through academic networks, consistent with the Prolific pilot (N=50) and full-study (N=300) design referenced elsewhere in this dissertation. Each participant completes three sessions across a seven-day window, corresponding to the three defined decision stages.

## **5.4 Evaluation Protocol**

Model performance will be evaluated using standard information retrieval metrics stratified by decision stage:

- NDCG@K (Normalized Discounted Cumulative Gain) computed separately for each stage, with K ∈ {5, 10, 20}

- Precision@K stratified by decision stage

- Stage-Weighted NDCG: a composite metric weighting stage-specific performance by the decision salience of each stage, with higher weight assigned to finalization-stage performance

- Coverage and Diversity metrics to assess recommendation quality beyond accuracy

Statistical significance of performance differences will be assessed using paired t-tests with Bonferroni correction for multiple comparisons. Effect sizes will be reported using Cohen's d.

## **5.5 Baseline Models**

Consistent with the scope boundary in Section 3, baseline models for Paper I are separated into two categories: models that are (or are planned to be) actually implemented, trained, and evaluated under a fair protocol, and models that are discussed conceptually in Related Work to position CAFE-LightFM but are **not** placed in the experimental comparison table unless they meet the implementation bar below.

**5.5.1 Implemented / experimental baselines (minimum set).**

- LightFM-style hybrid MF baseline (PyTorch-native, stage-agnostic) — direct predecessor comparison, isolates the effect of stage conditioning together with the ablation variants below.
- CAFE-LightFM (Full) — the complete proposed model.
- CAFE-LightFM (noStage) — ablation isolating the contribution of stage conditioning (Section 5.6).
- CAFE-LightFM (noAttention) — ablation isolating the contribution of the attention mechanism (Section 5.6).
- Context-aware Factorization Machines (Rendle et al., 2011) — included if implementation time permits before the Prolific pilot; if not feasible, it is discussed in Related Work only and explicitly marked as not evaluated.
- SASRec (Kang & McAuley, 2018) / BERT4Rec (Sun et al., 2019) — included in the experimental table **only if actually implemented and evaluated under the same protocol**; otherwise they are discussed in Related Work as strong sequential-attention baselines without an empirical comparison row, to avoid presenting an unsupported claim.

**5.5.2 Related Work only (explicitly excluded from the experimental baseline table).**

Graph-based and session-based sequential models — including SR-GNN (Wu et al., 2019), TIGSA (Chen & Wang, 2022), and DSIN (Feng et al., 2019) — are discussed in Related Work to delineate the boundary of this work (Section 1), but are not included as empirical baselines in Paper I unless a future revision implements, tunes, and fairly evaluates them. The rationale is architectural rather than a claim of inferiority: these models presuppose denser, higher-frequency interaction sequences suited to browsing/session behavior, whereas the present work targets sparse, feature-rich, cold-start-prone, multi-session, high-stakes decision settings (Section 2.2) in which item-transition graph structure is comparatively less observable. STAN (Li et al., 2023), TimeSVD++ (Koren, 2009), and A-DNR (Wei et al., 2022) remain in Related Work pending an explicit implementation decision; they are not currently in the implemented set above and should not be presented as evaluated baselines until they are.

## **5.6 Ablation Study Design**

To isolate the contribution of each architectural component, the following ablation conditions are evaluated. Status reflects the current, verified state of implementation (Phase 3, Step 7 — synthetic-v3, validation-only; not yet run on Prolific data):

- **CAFE-LightFM-noStage** — CAFE-LightFM with stage-conditioned variation disabled by using a constant stage index across all samples during training and evaluation. *Implemented and evaluated on synthetic-v3.*
- **CAFE-LightFM-noAttention** — CAFE-LightFM with the attention computation bypassed and $\alpha$ held at a constant uniform value. *Implemented and evaluated on synthetic-v3.*
- **CAFE-LightFM-2Stage** — CAFE-LightFM with two stages instead of three (exploration vs. decision, i.e., S2∪S3 merged). Evaluated descriptively only, not through paired significance testing, due to a structural cardinality mismatch with the 3-stage Full model. *Implemented and evaluated on synthetic-v3.*
- **CAFE-LightFM-Full** — Complete model as described in Section 5.2. *Implemented and evaluated on synthetic-v3.*
- **CAFE-LightFM-fixed-stage-weights** *(planned, not yet implemented)* — CAFE-LightFM with the learned attention weight matrices $W_{\text{base}}, W_s$ replaced by manually fixed, non-learned per-stage weights. This variant directly tests whether the empirical benefit (if any) of stage conditioning derives from *learned* stage-conditioned attention specifically, as opposed to any stage-stratified reweighting of features. This ablation is recommended to be run first on synthetic-v3 as a pipeline check, prior to inclusion in the Prolific evaluation protocol.

All results from the four already-implemented variants are, per the binding scope statement (Section 7), validation-only and computed on synthetic-v3; no necessity or superiority claim about any architectural component is made pending the Prolific pilot and full study.

## **6. Expected Scientific Contributions**

This research is expected to produce the following original scientific contributions, understood throughout as *targets to be empirically established*, not results already demonstrated outside the synthetic validation pipeline (Section 7).

## **6.1 Theoretical Contributions**

- A formal mathematical definition of decision stages within the recommendation context, grounded in measurable behavioral signals (Section 5.1). This definition establishes a foundation for a nascent sub-field of stage-aware recommendation research; the automatic operationalization of this definition into a detection algorithm is the contribution of Paper II.

- A stage-conditioned attention mechanism formalized as a modification to the hybrid matrix factorization framework (Section 5.2), whose architectural compatibility with hybrid MF is established by construction; whether it is empirically *beneficial* relative to a stage-agnostic baseline is precisely RQ1 (Section 3) and is not presupposed by the theoretical contribution.

## **6.2 Methodological Contributions**

- The CAFE-LightFM model: an open-source, reproducible implementation of stage-conditioned hybrid matrix factorization, providing a reference implementation for subsequent research in stage-aware recommendation.

- A behavioral research platform for multi-stage decision data collection, enabling the generation of datasets with high-quality, explicitly self-reported stage labels — a persistent bottleneck in this research area.

- A stage-stratified evaluation protocol providing metrics and statistical procedures for assessing recommendation quality across decision stages, with particular emphasis on finalization-stage performance.

## **6.3 Empirical Contributions (gated on Prolific data — see Section 7)**

- An empirical test of whether stage-conditioned feature attention yields statistically significant improvements in recommendation quality relative to a stage-agnostic hybrid MF baseline, reported with effect sizes for reproducibility regardless of outcome, including a null result if the ablation evidence supports one.

- An interpretability analysis of learned attention weights, examining whether CAFE-LightFM captures theoretically expected patterns of feature salience — reported descriptively, with any semantic interpretability claim explicitly conditioned on the availability of item metadata carrying real-world semantic content (not guaranteed on synthetic validation data; see Section 7).

No claim of demonstrated superiority, necessity, or significant improvement is made in this document independent of the Prolific pilot (N=50) and full study (N=300) results (Section 5.3); synthetic-data results referenced elsewhere in project documentation are validation-only and confirm pipeline correctness, not scientific findings.

## **7. Limitations and Scope Boundaries**

This section consolidates, in one place, the scope limitations that govern every claim made in this document and in all downstream experimental reporting for Paper I.

1. **No automatic stage detection in Paper I.** All experiments to date, and all experiments planned under this proposal, take decision-stage membership as a given, explicitly labeled, or externally supplied input. The design and evaluation of an automatic stage-boundary detection algorithm is exclusively the contribution of Paper II of this dissertation.

2. **No stage-bias term.** The SCA attention computation and the final prediction score contain no additive stage-bias parameter ($b_s$); stage enters the model exclusively through the feature-attention weights $\alpha(f,s)$ (Section 5.2). Any earlier draft of this proposal, or any dissertation-level summary document, referencing a stage-bias term should be read as superseded by this revision.

3. **Synthetic-data results are validation-only.** All results computed to date on the synthetic-v3 dataset confirm that the modeling and evaluation pipeline executes and computes metrics correctly; they do not constitute evidence for or against CAFE-LightFM's architectural value, including the relative necessity of stage-conditioning or attention weighting. This includes a documented null ablation result on synthetic-v3 (0 of 18 Full-vs-noStage and 0 of 18 Full-vs-noAttention comparisons significant after Bonferroni correction), attributed to a structural limitation of the synthetic dataset's collapsed item-category vocabulary rather than to a failure of the SCA mechanism — but this attribution is itself unverified pending Prolific data and is not presented as an established finding.

4. **Formal, publishable performance or necessity claims are reserved exclusively for the Prolific pilot (N=50) and full study (N=300).** No superiority, benefit, or necessity claim about CAFE-LightFM or any of its components is made independent of that data.

5. **Restricted attention scope in the current implementation.** The verified SCA implementation attends over two item-side metadata features (`category`, `program`) only; this is a scope of the current implementation, not an inherent limit of the general SCA formulation (Section 5.2), and is expected to be revisited once richer, semantically meaningful item metadata is available from Prolific data collection.

6. **Baseline coverage is intentionally limited.** Graph-based and session-based sequential models (SR-GNN, TIGSA, DSIN) are discussed only in Related Work and are not empirical baselines in this paper (Section 5.5.2); their absence reflects a scope decision grounded in differing modeling assumptions (dense sequential/session data vs. sparse, high-stakes, multi-session decision data), not a claim that hybrid MF outperforms them. Sequential Transformer baselines (SASRec, BERT4Rec) are included in the experimental comparison only if actually implemented and evaluated under a fair, identical protocol; otherwise they remain conceptual points of comparison in Related Work.

7. **No claim of semantic interpretability on synthetic data.** Because the synthetic-v3 item-category vocabulary is collapsed (a single realized category value; Phase 3, Step 6), any attention-weight interpretability analysis performed on synthetic data is a diagnostic of allocation behavior only, not a validation that learned weights track real-world feature semantics. That validation is deferred to Prolific data, where item metadata carries genuine domain meaning.

## **8. Key References**

The following references constitute the primary scholarly foundation for this research. References are organized by thematic cluster.

## **8.1 Hybrid Matrix Factorization and LightFM**

- **[1]** Kula, M. (2015). Metadata Embeddings for User and Item Cold-start Recommendations. Proceedings of the 2nd Workshop on New Trends in Content-Based Recommender Systems, RecSys 2015. CEUR-WS, Vol. 1448, pp. 14–21. [Primary framework reference — direct predecessor of CAFE-LightFM]

- **[2]** Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30–37. IEEE. [Foundational matrix factorization reference]

- **[3]** He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural Collaborative Filtering. Proceedings of the 26th International Conference on World Wide Web (WWW), pp. 173–182. [Neural extension of matrix factorization]

## **8.2 Attention Mechanisms in Recommendation**

- **[4]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS), 30. [Foundational attention mechanism reference]

- **[5]** Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. Proceedings of the 2018 IEEE International Conference on Data Mining (ICDM), pp. 197–206. arXiv:1808.09781. [Primary attention-based recommendation baseline]

- **[6]** Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM), pp. 1441–1450. [BERT-based sequential recommendation baseline]

- **[7]** Wei, C., Qin, J., & Ren, Q. (2022). A Ranking Recommendation Algorithm Based on Dynamic User Preference. Sensors, 22(22), 8683. MDPI. DOI: 10.3390/s22228683. [Attention + Matrix Factorization + Dynamic Preference — closest architectural precedent]

- **[8]** Li, Z., Jin, D., & Yuan, K. (2023). Attentional Factorization Machine with Review-based User-Item Interaction for Recommendation. Scientific Reports, 13, 13451. Nature. DOI: 10.1038/s41598023-40633-4. [Attention-based factorization machine — 2023]

- **[9]** Zhang, C.K., & Wang, C. (2020). Probabilistic Matrix Factorization Recommendation of Self-Attention Mechanism Convolutional Neural Networks with Item Auxiliary Information. IEEE Access, 8, 208311–208321. DOI: 10.1109/ACCESS.2020.3037955. [Self-attention + probabilistic matrix factorization]

## **8.3 Stage-Aware and Multi-Stage Recommendation**

- **[10]** Li, W., Zheng, W., Xiao, X., & Wang, S. (2023). STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation. Proceedings of the 17th ACM Conference on Recommender Systems (RecSys 2023), pp. 602–612. [Most directly related stage-aware model — key baseline]

- **[11]** Cheng, Z., et al. (2026). Modeling Stage-wise Evolution of User Interests for News Recommendation. arXiv:2603.10471. [Stage-wise evolution — most recent related work]

- **[12]** Zhao, H., Zhang, Z., et al. (2024). Full Stage Learning to Rank: A Unified Framework for Multi-Stage Systems. Proceedings of the ACM Web Conference 2024, pp. 3621–3631. [Multi-stage ranking framework]

## **8.4 Preference Evolution and Dynamic User Modeling**

- **[13]** Hu, D., et al. (2021). PEN4Rec: Preference Evolution Networks for Session-based Recommendation. arXiv:2106.09306. [Preference evolution networks — direct thematic precedent]

- **[14]** Ju, B., Qian, Y., Ye, M., Ni, R., & Zhu, C. (2015). Using Dynamic Multi-Task Non-Negative Matrix Factorization to Detect the Evolution of User Preferences in Collaborative Filtering. PLOS ONE, 10(8), e0135090. DOI: 10.1371/journal.pone.0135090. [Dynamic preference evolution in matrix factorization]

- **[15]** Koren, Y. (2009). Collaborative Filtering with Temporal Dynamics. Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 447–456. [Temporal dynamics in collaborative filtering — foundational]

- **[16]** Sun, Z., et al. (2019). Research on Dynamic Recommendation Based on User Preferences and Behaviors. Proceedings of the International Conference on Knowledge Science, Engineering and Management. [Dynamic user preference modeling]

## **8.5 Cold-Start and Hybrid Recommendation**

- **[17]** Rendle, S. (2010). Factorization Machines. Proceedings of the 2010 IEEE International Conference on Data Mining (ICDM), pp. 995–1000. [Factorization machines — methodological foundation for feature-based factorization]

- **[17b]** Rendle, S., Gantner, Z., Freudenthaler, C., & Schmidt-Thieme, L. (2011). Fast Context-aware Recommendations with Factorization Machines. Proceedings of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2011), pp. 635–644. DOI: 10.1145/2009916.2010002. [Context-aware factorization machines — candidate implemented baseline, Section 5.5.1]

- **[18]** Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). Neural Graph Collaborative Filtering. Proceedings of the 42nd International ACM SIGIR Conference, pp. 165–174. [Graph-based collaborative filtering — comparison context]

- **[19]** Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep Learning Based Recommender System: A Survey and New Perspectives. ACM Computing Surveys, 52(1), 1–38. DOI: 10.1145/3285029. [Comprehensive survey — provides contextual positioning]

## **8.6 Evaluation Methodology**

- **[20]** Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-based Evaluation of IR Techniques. ACM Transactions on Information Systems, 20(4), 422–446. [NDCG evaluation metric — foundational reference]

- **[21]** Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI), pp. 452–461. [WARP/BPR loss functions — training objective reference]

- **[22]** Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based Recommendations with Recurrent Neural Networks. Proceedings of the 4th International Conference on Learning Representations (ICLR). [Session-based recommendation — methodological reference]

## **8.7 Application Domain — University and High-Stakes Recommendation**

- **[23]** Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments. User Modeling and User-Adapted Interaction, 12(4), 331–370. Springer. [Hybrid recommendation survey — domain context]

- **[24]** Forsati, R., Mahdavi, M., Shamsfard, M., & Meybodi, M. R. (2014). Matrix Factorization with Explicit Trust and Distrust Side Information for Improved Social Recommendation. ACM Transactions on Information Systems, 32(4), 1–38. [Trust-aware recommendation — applicable to high-stakes decisions]

- **[25]** Zheng, Y., et al. (2025). A Survey of Real-World Recommender Systems: Challenges, Constraints, and Industrial Perspectives. arXiv:2509.06002. [Most comprehensive recent survey — multi-stage industrial systems]

## **8.8 Session-Based and Graph-Based Models (Related Work only — Section 5.5.2)**

- **[26]** Feng, Y., Lv, F., Shen, W., Wang, M., Sun, F., Zhu, Y., & Yang, K. (2019). Deep Session Interest Network for Click-Through Rate Prediction. Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019). arXiv:1905.06482. [Session-heterogeneity motivation — Related Work, not an empirical baseline]

- **[27]** Wu, S., Tang, Y., Zhu, Y., Wang, L., Xie, X., & Tan, T. (2019). Session-Based Recommendation with Graph Neural Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 33(1), 346–353. arXiv:1811.00855. [Graph-based session model — Related Work, not an empirical baseline]

- **[28]** Chen, Z., & Wang, W. (2022). Time Interval-Aware Graph with Self-Attention for Sequential Recommendation. Proceedings of the 2022 5th International Conference on Algorithms, Computing and Artificial Intelligence (ACAI 2022). ACM. DOI: 10.1145/3579654.3579729. [Graph + time-interval attention — Related Work, not an empirical baseline]

- **[29]** Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T., & Ma, J. (2017). Neural Attentive Session-based Recommendation. Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM 2017), pp. 1419–1428. DOI: 10.1145/3132847.3132926. [Global/local session-interest separation — conceptual analogy to $W_{\text{base}}$/$W_s$ decomposition, Section 5.2; not an architectural equivalence]

- **[30]** Li, J., Wang, Y., & McAuley, J. (2020). Time Interval Aware Self-Attention for Sequential Recommendation. Proceedings of the 13th ACM International Conference on Web Search and Data Mining (WSDM 2020), pp. 322–330. DOI: 10.1145/3336191.3371786. [Time-interval-aware attention — motivates optional temporal control features, Section 5.2, not a core Paper I component]

_This proposal constitutes Paper I of a three-paper doctoral dissertation entitled:_

_**"Stage-Aware Recommendation Systems: Modeling, Detecting, and Generalizing Preference Evolution Across Multi-Stage Decision Contexts"**_
