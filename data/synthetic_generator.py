"""
synthetic_generator.py

Minimal, crude synthetic data generator for Phase 1, Step 2.

Purpose (per agreed sequencing, 2026-06-24): sanity-check the full
interaction_schema -> JSD pipeline under the SIMPLEST possible conditions
before introducing realism. This generator deliberately uses:

    - Tiny scale (5 users x 3 sessions x ~10 events/session = 150 events)
    - Deterministic stage-per-session mapping (session 1 -> S1,
      session 2 -> S2, session 3 -> S3), so any JSD detection failure is
      attributable to the JSD logic itself, not to label noise.
    - Strongly separable per-stage continuous-signal distributions
      (target Cohen's d >= 1.5 between adjacent stages), so the sanity
      check isolates correctness of the JSD computation from its
      statistical sensitivity.

Realistic, probabilistic, weakly-separable variants are deferred to
Step 4 ("refine the generator for realism") per the agreed sequencing.

Reproducibility: a single numpy Generator seeded at 42 (project-wide
convention) drives all randomness.

References
----------
[1] Lin, J. (1991). Divergence measures based on the Shannon entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
    DOI: 10.1109/18.61115
[2] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.). Routledge. ISBN 978-0-8058-0283-2.
    [Effect-size convention used to set distribution separability.]
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from data.interaction_schema import (
    CategoricalFeatures,
    ContinuousSignals,
    DecisionStage,
    Identifiers,
    InteractionRecord,
    InteractionType,
    StageLabels,
    StageLabelSource,
    TimeFields,
)

MASTER_SEED = 42

# --------------------------------------------------------------------------- #
# Deterministic session -> stage mapping
# --------------------------------------------------------------------------- #

SESSION_INDEX_TO_STAGE: Dict[int, DecisionStage] = {
    0: DecisionStage.EXPLORATION,
    1: DecisionStage.COMPARISON,
    2: DecisionStage.FINALIZATION,
}

# Sessions spread across a 7-day window: day 0, day 3, day 6
SESSION_INDEX_TO_DAY_OFFSET: Dict[int, int] = {0: 0, 1: 3, 2: 6}

# --------------------------------------------------------------------------- #
# Strongly separable continuous-signal parameters (mean, std) per stage
# --------------------------------------------------------------------------- #

CONTINUOUS_PARAMS: Dict[DecisionStage, Dict[str, Tuple[float, float]]] = {
    DecisionStage.EXPLORATION: {
        "hover_duration_ms": (800.0, 150.0),
        "dwell_time_ms": (2000.0, 300.0),
        "revisit_count": (1.0, 1.0),
        "scroll_depth_pct": (30.0, 8.0),
    },
    DecisionStage.COMPARISON: {
        "hover_duration_ms": (1400.0, 150.0),
        "dwell_time_ms": (3500.0, 300.0),
        "revisit_count": (4.0, 1.0),
        "scroll_depth_pct": (60.0, 8.0),
    },
    DecisionStage.FINALIZATION: {
        "hover_duration_ms": (2200.0, 150.0),
        "dwell_time_ms": (5500.0, 300.0),
        "revisit_count": (7.0, 1.0),
        "scroll_depth_pct": (85.0, 8.0),
    },
}

# Verify d = (mean_diff) / pooled_std (equal-std case, Cohen, 1988) >= 1.5
# for every adjacent stage pair, at module load time, so a parameter typo
# fails immediately rather than silently producing a weak dataset.
def _verify_separability(min_d: float = 1.5) -> None:
    stages = [
        DecisionStage.EXPLORATION,
        DecisionStage.COMPARISON,
        DecisionStage.FINALIZATION,
    ]
    for signal in CONTINUOUS_PARAMS[stages[0]]:
        for a, b in zip(stages[:-1], stages[1:]):
            mean_a, std_a = CONTINUOUS_PARAMS[a][signal]
            mean_b, std_b = CONTINUOUS_PARAMS[b][signal]
            pooled_std = ((std_a ** 2 + std_b ** 2) / 2.0) ** 0.5
            d = abs(mean_b - mean_a) / pooled_std
            if d < min_d:
                raise ValueError(
                    f"Separability check failed for '{signal}' between "
                    f"{a.value} and {b.value}: d={d:.2f} < {min_d}"
                )


_verify_separability()

# Interaction-type weights per stage (categorical; not used by JSD but kept
# behaviorally consistent with the stage definitions in the proposal).
INTERACTION_TYPE_WEIGHTS: Dict[DecisionStage, Dict[InteractionType, float]] = {
    DecisionStage.EXPLORATION: {
        InteractionType.VIEW: 0.6,
        InteractionType.HOVER: 0.4,
    },
    DecisionStage.COMPARISON: {
        InteractionType.HOVER: 0.3,
        InteractionType.COMPARE_ACTIVATE: 0.4,
        InteractionType.SHORTLIST_ADD: 0.3,
    },
    DecisionStage.FINALIZATION: {
        InteractionType.SHORTLIST_REMOVE: 0.4,
        InteractionType.COMPARE_ACTIVATE: 0.3,
        InteractionType.SELF_REPORT: 0.3,
    },
}

ITEM_POOL = [f"item_{i:02d}" for i in range(10)]
CATEGORY_POOL = ["university"]
PROGRAM_POOL = ["MSc_CS", "MSc_DS", "BSc_Eng"]


def _sample_continuous(rng: np.random.Generator, stage: DecisionStage) -> ContinuousSignals:
    params = CONTINUOUS_PARAMS[stage]
    hover = max(0.0, rng.normal(*params["hover_duration_ms"]))
    dwell = max(0.0, rng.normal(*params["dwell_time_ms"]))
    revisit = max(0, int(round(rng.normal(*params["revisit_count"]))))
    scroll = float(np.clip(rng.normal(*params["scroll_depth_pct"]), 0.0, 100.0))
    return ContinuousSignals(
        hover_duration_ms=hover,
        dwell_time_ms=dwell,
        revisit_count=revisit,
        scroll_depth_pct=scroll,
    )


def _sample_interaction_type(rng: np.random.Generator, stage: DecisionStage) -> InteractionType:
    weights = INTERACTION_TYPE_WEIGHTS[stage]
    types = list(weights.keys())
    probs = list(weights.values())
    idx = rng.choice(len(types), p=probs)
    return types[idx]


def generate_synthetic_dataset(
    n_users: int = 5,
    n_sessions: int = 3,
    events_per_session: int = 10,
    seed: int = MASTER_SEED,
) -> List[InteractionRecord]:
    """
    Generates a tiny, deterministic-stage, strongly-separable synthetic
    interaction dataset conforming to InteractionRecord.

    Parameters
    ----------
    n_users : int
        Number of synthetic users (default 5, per agreed tiny scale).
    n_sessions : int
        Sessions per user (default 3, mapped 1:1 to S1/S2/S3).
    events_per_session : int
        Events per session (default 10).
    seed : int
        Master seed for the numpy Generator (default 42, project convention).

    Returns
    -------
    List[InteractionRecord]
        Flat list of records, ordered by user, then session, then event.
    """
    if n_sessions != 3:
        raise ValueError(
            "This generator assumes exactly 3 sessions (1:1 with S1/S2/S3); "
            "got n_sessions={}.".format(n_sessions)
        )

    rng = np.random.default_rng(seed)
    base_date = datetime(2026, 6, 24, 9, 0, 0)
    records: List[InteractionRecord] = []

    for user_idx in range(n_users):
        user_id = f"u_{user_idx:03d}"

        for session_idx in range(n_sessions):
            stage = SESSION_INDEX_TO_STAGE[session_idx]
            session_id = f"{user_id}_s{session_idx}"
            day_offset = SESSION_INDEX_TO_DAY_OFFSET[session_idx]
            session_start = base_date + timedelta(days=day_offset)

            prev_timestamp = None
            for event_idx in range(events_per_session):
                gap_s = float(rng.uniform(20.0, 90.0)) if event_idx > 0 else 0.0
                timestamp = (
                    session_start
                    if event_idx == 0
                    else prev_timestamp + timedelta(seconds=gap_s)
                )
                prev_timestamp = timestamp

                identifiers = Identifiers(
                    user_id=user_id,
                    item_id=str(rng.choice(ITEM_POOL)),
                    session_id=session_id,
                    event_id=f"{session_id}_e{event_idx:03d}",
                )
                categorical = CategoricalFeatures(
                    item_category=str(rng.choice(CATEGORY_POOL)),
                    program_type=str(rng.choice(PROGRAM_POOL)),
                    interaction_type=_sample_interaction_type(rng, stage),
                )
                continuous = _sample_continuous(rng, stage)
                time_fields = TimeFields(
                    timestamp=timestamp,
                    event_index=event_idx,
                    time_since_start_s=(timestamp - session_start).total_seconds(),
                    time_since_previous_event_s=None if event_idx == 0 else gap_s,
                )
                stage_labels = StageLabels(
                    stage_true=stage,
                    stage_label_source=StageLabelSource.SYNTHETIC_GROUND_TRUTH,
                )

                records.append(
                    InteractionRecord(
                        identifiers=identifiers,
                        categorical=categorical,
                        continuous=continuous,
                        time=time_fields,
                        stage=stage_labels,
                    )
                )

    return records


if __name__ == "__main__":
    data = generate_synthetic_dataset()
    print(f"Generated {len(data)} synthetic records.")
    print("First record (flattened):")
    print(data[0].to_flat_dict())
