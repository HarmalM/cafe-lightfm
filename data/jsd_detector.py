"""
jsd_detector.py

Minimal JSD-based stage-boundary detector for Phase 1, Step 3.

Implements a first, deliberately simplified test of Definition 5.1 from the
CAFE-LightFM proposal: stage s_j is a period whose feature-attention
distribution P(f | s_j) differs significantly (by Jensen-Shannon divergence)
from adjacent stages.

Scope of THIS step (confirmed 2026-06-24):
    - Distribution representation: each continuous signal is discretized
      into a fixed-bin histogram (consistent bin edges across all sessions
      for that signal, computed globally), with additive (Laplace-style)
      smoothing to avoid zero-bin instability. JSD is computed per signal,
      then averaged across the four signals -- NOT a joint multivariate
      histogram (curse of dimensionality with only 10 events/session) and
      NOT KDE (unnecessary complexity for this sanity check).
    - Window granularity: one window = one full session (10 events),
      matching the deterministic session->stage design of Step 2.
    - Threshold: a SIMPLE PLACEHOLDER (multiple of the mean intra-session
      split-half JSD, i.e. the JSD "noise floor" when no real stage
      boundary exists). The full MDL-estimated tau described in the
      proposal is explicitly DEFERRED to Step 4, where realistic
      (weakly-separable) data makes a real threshold-estimation test
      meaningful. Using MDL on this artificially strongly-separable
      dataset would not actually test MDL's correctness -- any reasonable
      threshold succeeds here by construction.

References
----------
[1] Lin, J. (1991). Divergence measures based on the Shannon entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
    DOI: 10.1109/18.61115
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from data.interaction_schema import InteractionRecord

CONTINUOUS_SIGNALS: Tuple[str, ...] = (
    "hover_duration_ms",
    "dwell_time_ms",
    "revisit_count",
    "scroll_depth_pct",
)

DEFAULT_N_BINS = 10
DEFAULT_EPSILON = 1e-3  # additive smoothing constant
DEFAULT_THRESHOLD_MULTIPLIER = 3.0  # placeholder; MDL deferred to Step 4


# --------------------------------------------------------------------------- #
# Histogram construction (fixed bin edges, additive smoothing)
# --------------------------------------------------------------------------- #

def compute_global_bin_edges(
    records: Sequence[InteractionRecord],
    signal_name: str,
    n_bins: int = DEFAULT_N_BINS,
) -> np.ndarray:
    """
    Computes fixed bin edges for `signal_name` from the GLOBAL min/max
    across all records, so every session's histogram for this signal uses
    identical edges (required for a well-defined JSD comparison).
    """
    values = np.array([getattr(r.continuous, signal_name) for r in records], dtype=float)
    lo, hi = values.min(), values.max()
    if lo == hi:
        # Degenerate case safeguard: pad to avoid a zero-width bin range.
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, n_bins + 1)


def histogram_probabilities(
    values: np.ndarray,
    bin_edges: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Builds a smoothed empirical probability vector over `bin_edges`.
    Additive smoothing (+epsilon per bin before normalizing) prevents
    zero-probability bins from destabilizing the JSD computation when a
    session's values happen to miss some bins entirely.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    smoothed = counts.astype(float) + epsilon
    return smoothed / smoothed.sum()


# --------------------------------------------------------------------------- #
# Jensen-Shannon divergence
# --------------------------------------------------------------------------- #

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = (P + Q) / 2
    (Lin, 1991). With base=2, JSD is bounded in [0, 1].
    Assumes p and q are already normalized probability vectors with no
    exact zeros (guaranteed by histogram_probabilities' smoothing).
    """
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)) / np.log(base))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# --------------------------------------------------------------------------- #
# Session-level grouping and per-signal / averaged JSD
# --------------------------------------------------------------------------- #

def group_by_user_session(
    records: Sequence[InteractionRecord],
) -> Dict[str, Dict[str, List[InteractionRecord]]]:
    """Groups records as {user_id: {session_id: [records, ordered by event_index]}}."""
    grouped: Dict[str, Dict[str, List[InteractionRecord]]] = {}
    for r in records:
        user_bucket = grouped.setdefault(r.identifiers.user_id, {})
        session_bucket = user_bucket.setdefault(r.identifiers.session_id, [])
        session_bucket.append(r)
    for user_bucket in grouped.values():
        for session_id, session_records in user_bucket.items():
            user_bucket[session_id] = sorted(session_records, key=lambda r: r.time.event_index)
    return grouped


def average_signal_jsd(
    session_a: Sequence[InteractionRecord],
    session_b: Sequence[InteractionRecord],
    bin_edges_by_signal: Dict[str, np.ndarray],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Computes per-signal JSD between two record windows, averaged over
    the four continuous signals (per-signal histogram JSD, averaged --
    confirmed design, 2026-06-24)."""
    per_signal_scores = []
    for signal in CONTINUOUS_SIGNALS:
        values_a = np.array([getattr(r.continuous, signal) for r in session_a], dtype=float)
        values_b = np.array([getattr(r.continuous, signal) for r in session_b], dtype=float)
        p = histogram_probabilities(values_a, bin_edges_by_signal[signal], epsilon)
        q = histogram_probabilities(values_b, bin_edges_by_signal[signal], epsilon)
        per_signal_scores.append(jensen_shannon_divergence(p, q))
    return float(np.mean(per_signal_scores))


def split_half_internal_jsd(
    session_records: Sequence[InteractionRecord],
    bin_edges_by_signal: Dict[str, np.ndarray],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """
    Internal "noise floor" estimate: splits ONE session's events in half
    and computes the (averaged, per-signal) JSD between the two halves.
    A real stage boundary should produce a between-session JSD well above
    this floor; this is the placeholder substitute for MDL-based tau.
    """
    midpoint = len(session_records) // 2
    first_half = session_records[:midpoint]
    second_half = session_records[midpoint:]
    return average_signal_jsd(first_half, second_half, bin_edges_by_signal, epsilon)


# --------------------------------------------------------------------------- #
# End-to-end detection
# --------------------------------------------------------------------------- #

def detect_stage_boundaries(
    records: Sequence[InteractionRecord],
    n_bins: int = DEFAULT_N_BINS,
    epsilon: float = DEFAULT_EPSILON,
    threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
) -> Dict[str, object]:
    """
    Runs the full sanity-check pipeline:
        1. Fixed global bin edges per signal.
        2. Internal (split-half) JSD per session -> noise-floor estimate.
        3. Between-session JSD per consecutive session pair, per user.
        4. Placeholder threshold = threshold_multiplier * mean(internal JSD).
        5. Boundary detected where between-session JSD > threshold.

    Returns
    -------
    dict with keys:
        'bin_edges'        : Dict[str, np.ndarray]
        'internal_jsd'     : List[float]   (noise-floor samples)
        'between_jsd'      : List[Tuple[str, int, int, float]]
                              (user_id, session_idx_a, session_idx_b, jsd)
        'threshold'        : float
        'detections'       : List[Tuple[str, int, int, float, bool]]
                              (..., jsd, boundary_detected)
    """
    bin_edges_by_signal = {
        signal: compute_global_bin_edges(records, signal, n_bins)
        for signal in CONTINUOUS_SIGNALS
    }

    grouped = group_by_user_session(records)

    internal_jsd: List[float] = []
    between_jsd: List[Tuple[str, int, int, float]] = []

    for user_id, sessions in grouped.items():
        session_ids_sorted = sorted(sessions.keys(), key=lambda sid: sessions[sid][0].time.event_index * 0 + int(sid.split("_s")[-1]))
        # internal noise floor, one estimate per session
        for sid in session_ids_sorted:
            internal_jsd.append(split_half_internal_jsd(sessions[sid], bin_edges_by_signal, epsilon))

        # between consecutive sessions
        for i in range(len(session_ids_sorted) - 1):
            sid_a, sid_b = session_ids_sorted[i], session_ids_sorted[i + 1]
            jsd = average_signal_jsd(sessions[sid_a], sessions[sid_b], bin_edges_by_signal, epsilon)
            idx_a = int(sid_a.split("_s")[-1])
            idx_b = int(sid_b.split("_s")[-1])
            between_jsd.append((user_id, idx_a, idx_b, jsd))

    threshold = threshold_multiplier * float(np.mean(internal_jsd))

    detections = [
        (user_id, idx_a, idx_b, jsd, jsd > threshold)
        for (user_id, idx_a, idx_b, jsd) in between_jsd
    ]

    return {
        "bin_edges": bin_edges_by_signal,
        "internal_jsd": internal_jsd,
        "between_jsd": between_jsd,
        "threshold": threshold,
        "detections": detections,
    }


if __name__ == "__main__":
    from data.synthetic_generator import generate_synthetic_dataset

    dataset = generate_synthetic_dataset()
    result = detect_stage_boundaries(dataset)
    print(f"Threshold (placeholder, {DEFAULT_THRESHOLD_MULTIPLIER}x internal mean): {result['threshold']:.4f}")
    for user_id, idx_a, idx_b, jsd, detected in result["detections"]:
        print(f"  {user_id}: S{idx_a}->S{idx_b}  JSD={jsd:.4f}  boundary_detected={detected}")
