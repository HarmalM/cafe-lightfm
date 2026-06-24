"""
mdl_threshold.py

Step 5: MDL-estimated global threshold tau for JSD-based stage-boundary
detection (Definition 5.1, CAFE-LightFM proposal).

Design (confirmed 2026-06-24):
    - Simple grid search over candidate tau values, NOT a full dynamic-
      programming segmentation (e.g. PELT). Boundary CANDIDATE LOCATIONS
      are already fixed by the schema (one window = one full session);
      only the DECISION THRESHOLD needs calibrating at this step.
      Location-free change-point segmentation belongs to Paper II
      (HMM-based stage inference), not here.
    - Histogram-based two-part MDL cost, consistent with the histogram
      representation already used for JSD in Step 3 (data/jsd_detector.py)
      -- avoids introducing an unverified Gaussian assumption.
    - ONE GLOBAL tau across all users, since per-user data (2 candidate
      pairs/user) is too sparse to estimate a reliable per-user
      threshold, and Definition 5.1 defines boundaries as a property of
      the general framework, not an individually-calibrated cutoff.

MDL formulation (two-part code; Rissanen, 1978):
    For between-session candidate i, with n_i events per window and
    averaged per-signal JSD JSD_i (bits, base-2, Step 3's convention):

        Cost(NOT a boundary) = n_i * 2 * JSD_i_nats
            (data-coding savings forfeited by describing both windows
            with one pooled histogram instead of two separate ones)
        Cost(IS a boundary)  = param_cost_nats
            (fixed extra cost of a second histogram model's
            (n_bins - 1) free probability parameters, at the BIC rate
            of 0.5*log2(n_i) bits each)

    Summed over all candidates, the GLOBAL tau (bits) minimizing total
    cost under the decision rule "boundary iff JSD > tau" is the
    MDL-estimated threshold.

Explicit simplifying assumption: the four continuous signals are
already compressed into one averaged JSD score (Step 3's convention).
This formulation treats that average as a single effective divergence
rather than summing four separate per-signal MDL costs. Revisiting this
(per-signal weighted MDL) is a candidate refinement for Paper II.

Decoupling of bin counts (confirmed 2026-06-24): a diagnostic run showed
that using the SAME bin count for (a) the JSD histogram representation
and (b) the BIC model-complexity penalty creates an unintended conflict
at this window size (20 events): more bins improve JSD's ability to
resolve fine distributional differences, but simultaneously inflate the
BIC penalty `(n_bins - 1) * 0.5 * log2(n_events)` faster than the
achievable data-coding savings can grow, so MDL never favors declaring
a boundary once bins exceed ~3. These two bin counts measure different
things -- representation precision vs. parametric complexity -- and
have no mathematical requirement to share a value. This module therefore
exposes them as independent parameters:
    - `jsd_n_bins` (passed into build_candidates_from_dataset): controls
      JSD histogram precision, kept at 10 for consistency with Step 3.
    - `mdl_n_bins` (passed into the cost functions and grid search):
      controls ONLY the BIC penalty's free-parameter count.

References
----------
[1] Rissanen, J. (1978). Modeling by shortest data description.
    Automatica, 14(5), 465-471. DOI: 10.1016/0005-1098(78)90005-5
[2] Davis, R. A., Lee, T. C. M., & Rodriguez-Yam, G. A. (2006). Structural
    Break Estimation for Nonstationary Time Series Models. Journal of the
    American Statistical Association, 101(473), 223-239.
    DOI: 10.1198/016214505000000745
[3] Lin, J. (1991). Divergence measures based on the Shannon entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
    DOI: 10.1109/18.61115
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from data.jsd_detector import (
    CONTINUOUS_SIGNALS,
    DEFAULT_EPSILON,
    average_signal_jsd,
    compute_global_bin_edges,
    group_by_user_session,
)
from data.interaction_schema import InteractionRecord

LN2 = math.log(2.0)


@dataclass(frozen=True)
class MDLCandidate:
    """One between-session boundary-decision candidate."""
    user_id: str
    session_idx_a: int
    session_idx_b: int
    jsd_bits: float  # averaged per-signal JSD, base-2 (Step 3 convention)
    n_events: int    # events per window (assumed equal on both sides)


def bits_to_nats(value_bits: float) -> float:
    """Converts a quantity in bits (log base 2) to nats (log base e)."""
    return value_bits * LN2


def param_cost_bits(mdl_n_bins: int, n_events: int) -> float:
    """BIC-style cost of one EXTRA histogram model's (mdl_n_bins - 1) free
    probability parameters, each at 0.5*log2(n_events) bits (Rissanen, 1978).
    `mdl_n_bins` is independent of the bin count used for JSD itself
    (see module docstring, 'Decoupling of bin counts')."""
    free_params = max(mdl_n_bins - 1, 1)
    return free_params * 0.5 * math.log2(max(n_events, 2))


def candidate_cost_no_boundary_nats(candidate: MDLCandidate) -> float:
    """Data-coding cost forfeited by NOT splitting: n * 2 * JSD_nats."""
    return candidate.n_events * 2.0 * bits_to_nats(candidate.jsd_bits)


def candidate_cost_boundary_nats(candidate: MDLCandidate, mdl_n_bins: int) -> float:
    """Fixed extra-model-parameter cost of declaring a boundary."""
    return bits_to_nats(param_cost_bits(mdl_n_bins, candidate.n_events))


def total_mdl_cost(tau_bits: float, candidates: Sequence[MDLCandidate], mdl_n_bins: int) -> float:
    """Total relative description length (nats) for global threshold tau_bits,
    under the rule: boundary iff jsd_bits > tau_bits. `mdl_n_bins` controls
    ONLY the BIC penalty -- independent of the JSD histogram's bin count."""
    total = 0.0
    for c in candidates:
        if c.jsd_bits > tau_bits:
            total += candidate_cost_boundary_nats(c, mdl_n_bins)
        else:
            total += candidate_cost_no_boundary_nats(c)
    return total


def estimate_global_tau_via_grid_search(
    candidates: Sequence[MDLCandidate],
    mdl_n_bins: int,
    n_grid_points: int = 200,
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Grid search over tau in [0, max(jsd_bits)] minimizing total_mdl_cost.
    `mdl_n_bins` is the BIC-penalty bin count, independent of the JSD
    histogram's bin count used when the candidates were built.

    Returns (best_tau_bits, grid) where grid is a list of (tau, cost)
    pairs for diagnostics/plotting.
    """
    if not candidates:
        raise ValueError("Need at least one candidate to estimate tau.")

    max_jsd = max(c.jsd_bits for c in candidates)
    grid_taus = np.linspace(0.0, max_jsd, n_grid_points)

    grid: List[Tuple[float, float]] = []
    best_tau, best_cost = 0.0, math.inf
    for tau in grid_taus:
        cost = total_mdl_cost(float(tau), candidates, mdl_n_bins)
        grid.append((float(tau), cost))
        if cost < best_cost:
            best_cost, best_tau = cost, float(tau)

    return best_tau, grid


def build_candidates_from_dataset(
    records: Sequence[InteractionRecord],
    jsd_n_bins: int = 10,
    epsilon: float = DEFAULT_EPSILON,
) -> List[MDLCandidate]:
    """Builds the MDLCandidate list directly from a dataset, reusing
    Step 3's grouping/JSD machinery for consistency. `jsd_n_bins` controls
    JSD histogram precision ONLY -- independent of the MDL cost model's
    `mdl_n_bins` used downstream in the cost functions."""
    bin_edges_by_signal = {
        signal: compute_global_bin_edges(records, signal, jsd_n_bins) for signal in CONTINUOUS_SIGNALS
    }
    grouped = group_by_user_session(records)

    candidates: List[MDLCandidate] = []
    for user_id, sessions in grouped.items():
        session_ids_sorted = sorted(sessions.keys(), key=lambda sid: int(sid.split("_s")[-1]))
        for i in range(len(session_ids_sorted) - 1):
            sid_a, sid_b = session_ids_sorted[i], session_ids_sorted[i + 1]
            session_a, session_b = sessions[sid_a], sessions[sid_b]
            jsd_bits = average_signal_jsd(session_a, session_b, bin_edges_by_signal, epsilon)
            idx_a = int(sid_a.split("_s")[-1])
            idx_b = int(sid_b.split("_s")[-1])
            candidates.append(
                MDLCandidate(
                    user_id=user_id,
                    session_idx_a=idx_a,
                    session_idx_b=idx_b,
                    jsd_bits=jsd_bits,
                    n_events=len(session_a),
                )
            )
    return candidates


if __name__ == "__main__":
    from data.synthetic_generator_v2 import generate_synthetic_dataset_v2
    from data.jsd_detector import cross_user_same_stage_baseline

    JSD_N_BINS = 10   # representation precision -- matches Step 3
    MDL_N_BINS = 4    # BIC penalty bins -- decoupled; best TPR-FPR in scan

    dataset = generate_synthetic_dataset_v2()
    bin_edges = {s: compute_global_bin_edges(dataset, s) for s in CONTINUOUS_SIGNALS}

    candidates = build_candidates_from_dataset(dataset, jsd_n_bins=JSD_N_BINS)
    tau, _grid = estimate_global_tau_via_grid_search(candidates, mdl_n_bins=MDL_N_BINS)

    negative_pairs = cross_user_same_stage_baseline(dataset, bin_edges, seed=42)
    negative_jsd = [jsd for (_, _, _, jsd) in negative_pairs]
    positive_jsd = [c.jsd_bits for c in candidates]

    tpr = sum(1 for v in positive_jsd if v > tau) / len(positive_jsd)
    fpr = sum(1 for v in negative_jsd if v > tau) / len(negative_jsd)

    print(f"JSD representation bins: {JSD_N_BINS} | MDL cost-model bins: {MDL_N_BINS}")
    print(f"MDL-estimated global tau: {tau:.4f} bits")
    print(f"TPR (same-user, adjacent-stage pairs): {tpr:.2f}")
    print(f"FPR (cross-user, same-stage pairs):    {fpr:.2f}")
    print(f"TPR - FPR: {tpr - fpr:.2f}")
