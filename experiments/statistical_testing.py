"""
experiments/statistical_testing.py

Phase 3, Step 4 — Statistical significance testing for CAFE-LightFM vs.
baseline model comparisons.

SCOPE (binding, per project convention established in Phase 3 Step 1.5 /
Step 2): this module is a **scaffold / pipeline-validation component**.
Any p-value, effect size, or significance flag it produces on synthetic
data (synthetic_generator_v3.py) is a pipeline-correctness check only and
carries **no scientific claim** about CAFE-LightFM's architectural
superiority. That claim is reserved for the pilot (N=50) and full
(N=300) Prolific Academic studies (proposal Section 5.3).

Design decisions confirmed 2026-07-05 (see project session log):
  A2 — SW-NDCG is EXCLUDED from paired per-user testing in this scaffold.
       SW-NDCG (experiments/sw_ndcg.py) has no natural per-user value
       unless a user has positive interactions in all three stages
       (S1, S2, S3), which is not guaranteed in synthetic_generator_v3.
       Formal SW-NDCG paired testing is deferred to the Prolific dataset,
       where three-session coverage per participant is guaranteed by the
       data-collection protocol (proposal Section 5.3).
  B  — Two-tailed paired t-test (conservative default; no a priori
       direction of superiority is assumed in this validation scaffold).
  C  — Bonferroni M is DYNAMIC by default (computed from the number of
       comparisons actually supplied to `bonferroni_correction`), with
       an optional `n_comparisons` override so the locked full-study
       value M=63 (proposal Section 5.4) can be substituted later
       without any change to this module's logic.

Zero re-implementation rule: this module does not re-derive ranking,
DCG/IDCG, or NDCG/Precision computation. It consumes per-user metric
arrays as plain numpy arrays; callers are responsible for producing
those arrays from experiments/ndcg.py and experiments/precision_at_k.py.

References
----------
[1] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T.,
    Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J.,
    van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov,
    N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., ... SciPy 1.0
    Contributors. (2020). SciPy 1.0: Fundamental Algorithms for
    Scientific Computing in Python. Nature Methods, 17, 261-272.
    DOI: 10.1038/s41592-019-0686-2
    [Source of scipy.stats.ttest_rel used below.]

[2] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.). Lawrence Erlbaum Associates.
    [PRE-2020 — flagged per user preference for 2020+ sources. Retained
    as irreplaceable: this is the foundational definition of standardized
    effect size (d) and its paired-sample variant (d_z) used throughout
    the behavioral and IR literature; no 2020+ source redefines the
    quantity itself. Confirmed acceptable by user, 2026-07-05.]

[3] Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo
    delle probabilità. Pubblicazioni del R. Istituto Superiore di Scienze
    Economiche e Commerciali di Firenze, 8, 3-62.
    [PRE-2020 — flagged. Accepted as an irreplaceable classical
    foundational reference per explicit user decision, 2026-07-05
    (Decision: Option 1 — no 2020+ replacement search performed).]

[4] Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-based
    Evaluation of IR Techniques. ACM Transactions on Information
    Systems, 20(4), 422-446. DOI: 10.1145/582415.582418
    [Context reference: this module tests significance of differences
    in the NDCG@K / Precision@K metrics defined per this source and
    implemented in experiments/ndcg.py and experiments/precision_at_k.py.]

Author: CAFE-LightFM project (Paper I of III)
Master seed (project-locked, config.py): 42
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

# Locked project master seed (config.py). Not used for randomness in this
# module (no stochastic operations occur here), retained for consistency
# with project-wide reproducibility documentation conventions.
SEED = 42


@dataclass
class PairedTestResult:
    """
    Container for a single paired two-tailed t-test result.

    Attributes
    ----------
    metric_name : str
        Identifier for the metric being compared (e.g. "NDCG@10_S3").
    n_pairs : int
        Number of paired observations (users) contributing to the test.
    mean_a, mean_b : float
        Sample means of model A and model B respectively.
    mean_diff : float
        mean_a - mean_b.
    t_statistic : float
        Paired t-statistic from scipy.stats.ttest_rel [1].
    p_value : float
        Two-tailed p-value, uncorrected for multiple comparisons.
    cohens_dz : float
        Standardized paired effect size (Cohen, 1988) [2].
    degrees_of_freedom : int
        n_pairs - 1.
    """
    metric_name: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float
    t_statistic: float
    p_value: float
    cohens_dz: float
    degrees_of_freedom: int


def cohens_dz(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Compute Cohen's d_z, the standardized effect size for a paired
    (within-subjects) design (Cohen, 1988, Ch. 2) [2]:

        d_z = mean(D) / std(D, ddof=1),  where D = scores_a - scores_b

    Parameters
    ----------
    scores_a, scores_b : np.ndarray, shape (n,)
        Paired per-user metric values for model A and model B. Must be
        the same length and represent the same users in the same order.

    Returns
    -------
    float
        Cohen's d_z. Returns 0.0 if the paired differences have zero
        sample variance (all differences identical, including the
        degenerate case scores_a == scores_b everywhere), to avoid a
        division-by-zero NaN in an otherwise well-defined "no effect"
        case.

    Raises
    ------
    ValueError
        If scores_a and scores_b have different lengths, or fewer than
        2 paired observations are supplied (sample std is undefined
        for n < 2).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must have matching shapes; "
            f"got {a.shape} vs {b.shape}."
        )
    if a.size < 2:
        raise ValueError(
            "cohens_dz requires at least 2 paired observations "
            f"(got {a.size})."
        )
    diff = a - b
    diff_std = np.std(diff, ddof=1)
    if diff_std == 0.0:
        return 0.0
    return float(np.mean(diff) / diff_std)


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "unnamed_metric",
) -> PairedTestResult:
    """
    Two-tailed paired t-test comparing model A and model B on a single
    metric, evaluated per-user.

    Uses scipy.stats.ttest_rel (Virtanen et al., 2020) [1] with
    alternative="two-sided", per project Decision B (confirmed
    2026-07-05): this scaffold does not assume a direction of
    superiority a priori.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray, shape (n,)
        Paired per-user metric values (e.g. NDCG@10 for stage S3) for
        model A (typically CAFE-LightFM) and model B (typically the
        LightFM baseline). Index i in both arrays must refer to the
        same user.
    metric_name : str, default "unnamed_metric"
        Human-readable label carried into the result for reporting.

    Returns
    -------
    PairedTestResult

    Raises
    ------
    ValueError
        If array lengths mismatch or fewer than 2 paired observations
        are supplied.

    Notes
    -----
    EDGE CASE (discovered during Step 4 integration testing,
    2026-07-05): if every paired difference (a_i - b_i) is identical
    (most commonly all zero, e.g. both models score identically for
    every user), scipy.stats.ttest_rel [1] computes t = 0/0 = NaN and
    correspondingly p = NaN, since the standard error of the mean
    difference is zero. A NaN p-value is not a valid statistical
    result and would crash any downstream Bonferroni correction
    (nan fails the [0, 1] range check by design). This function
    detects zero-variance differences directly (the same check used in
    `cohens_dz`) and reports t_statistic = 0.0, p_value = 1.0 in that
    case -- the correct two-tailed conclusion when there is genuinely
    no evidence of a difference between models on this metric, rather
    than propagating an undefined NaN.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must have matching shapes; "
            f"got {a.shape} vs {b.shape}."
        )
    if a.size < 2:
        raise ValueError(
            f"paired_t_test requires at least 2 paired observations "
            f"(got {a.size})."
        )

    diff = a - b
    diff_std = np.std(diff, ddof=1)
    if diff_std == 0.0:
        # See "EDGE CASE" note above: avoid scipy's 0/0 -> NaN result.
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = stats.ttest_rel(a, b, alternative="two-sided")
    dz = cohens_dz(a, b)

    return PairedTestResult(
        metric_name=metric_name,
        n_pairs=int(a.size),
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        mean_diff=float(np.mean(a) - np.mean(b)),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        cohens_dz=dz,
        degrees_of_freedom=int(a.size - 1),
    )


def bonferroni_correction(
    p_values: Dict[str, float],
    alpha: float = 0.05,
    n_comparisons: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Apply Bonferroni correction (Bonferroni, 1936) [3] to a family of
    p-values from independent paired tests.

    alpha* = alpha / M

    Per project Decision C (confirmed 2026-07-05): M defaults to
    len(p_values) — i.e. the number of comparisons actually supplied in
    this call ("dynamic"). This is intentional for the current
    synthetic-validation scaffold, where the full M=63 comparison count
    (proposal Section 5.4: 7 baseline models x 3 stages x 3 K-values
    x ... ) does not yet apply. Pass `n_comparisons=63` explicitly to
    reproduce the locked full-study correction once the complete
    baseline suite and Prolific dataset are in place.

    Parameters
    ----------
    p_values : Dict[str, float]
        Mapping of comparison identifier (e.g. "NDCG@10_S3") to its
        uncorrected two-tailed p-value.
    alpha : float, default 0.05
        Family-wise significance level before correction.
    n_comparisons : Optional[int], default None
        Override for M. If None, M = len(p_values) (dynamic, Decision C).
        If provided, must be a positive integer >= len(p_values) is NOT
        enforced here (a smaller override is permitted for sub-family
        analyses), but M must be > 0.

    Returns
    -------
    Dict[str, Dict[str, float]]
        For each comparison identifier: {"p_value", "alpha_corrected",
        "significant"} where "significant" = (p_value < alpha_corrected).

    Raises
    ------
    ValueError
        If p_values is empty, if n_comparisons is provided and <= 0, or
        if any p-value lies outside [0, 1].
    """
    if len(p_values) == 0:
        raise ValueError("p_values must contain at least one comparison.")
    for name, p in p_values.items():
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"p-value for '{name}' is out of range [0, 1]: {p}"
            )

    m = n_comparisons if n_comparisons is not None else len(p_values)
    if m <= 0:
        raise ValueError(f"n_comparisons must be a positive integer, got {m}.")

    alpha_corrected = alpha / m

    return {
        name: {
            "p_value": float(p),
            "alpha_corrected": float(alpha_corrected),
            "significant": bool(p < alpha_corrected),
        }
        for name, p in p_values.items()
    }


def summarize_paired_results(
    results: Sequence[PairedTestResult],
    alpha: float = 0.05,
    n_comparisons: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience wrapper: run Bonferroni correction (Decision C) across a
    batch of PairedTestResult objects produced by `paired_t_test`, keyed
    by each result's `metric_name`.

    NOTE — zero re-implementation: this function does not compute any
    NDCG, Precision, or t-test values itself; it only aggregates already-
    computed PairedTestResult objects and applies `bonferroni_correction`.

    Parameters
    ----------
    results : Sequence[PairedTestResult]
    alpha : float, default 0.05
    n_comparisons : Optional[int], default None
        See `bonferroni_correction`. Left as None (dynamic, M=len(results))
        unless the caller explicitly locks it (e.g. to 63).

    Returns
    -------
    Dict[str, Dict[str, float]]
        Keyed by metric_name; each value merges the original
        PairedTestResult fields with the Bonferroni-corrected fields.

    Raises
    ------
    ValueError
        If `results` is empty, or if metric_name values are not unique
        (a duplicate would silently overwrite a prior result).
    """
    if len(results) == 0:
        raise ValueError("results must contain at least one PairedTestResult.")

    names = [r.metric_name for r in results]
    if len(names) != len(set(names)):
        raise ValueError(
            "Duplicate metric_name values found in results; "
            "each comparison must have a unique identifier."
        )

    p_values = {r.metric_name: r.p_value for r in results}
    corrected = bonferroni_correction(
        p_values, alpha=alpha, n_comparisons=n_comparisons
    )

    summary: Dict[str, Dict[str, float]] = {}
    for r in results:
        summary[r.metric_name] = {
            "n_pairs": r.n_pairs,
            "mean_a": r.mean_a,
            "mean_b": r.mean_b,
            "mean_diff": r.mean_diff,
            "t_statistic": r.t_statistic,
            "cohens_dz": r.cohens_dz,
            "degrees_of_freedom": r.degrees_of_freedom,
            **corrected[r.metric_name],
        }
    return summary
