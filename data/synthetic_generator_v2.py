"""
synthetic_generator_v2.py

Step 4 ("refine the generator for realism") synthetic data generator.

Builds on the Step 2 minimal generator (data/synthetic_generator.py) but
introduces two changes, confirmed 2026-06-24, to test JSD sensitivity
under more realistic conditions:

    1. Larger scale: 20 users x 3 sessions x ~20 events/session (1,200
       records), giving histogram-based JSD enough samples to be stable
       once separability is weaker.
    2. Probabilistic stage leakage: each event's session has a NOMINAL
       stage (still deterministic: session i -> S_{i+1}, as in Step 2),
       but each event independently has a fixed 20% chance of being
       generated from a NEIGHBORING stage's distribution instead (e.g. a
       Comparison-session event behaving momentarily like Exploration or
       Finalization). `stage_true` records the ACTUAL generating stage
       per event (the realistic ground truth), not the session's nominal
       label -- this is what a real stage-detector must recover.
    3. Weaker separability: continuous-signal distributions are tuned so
       adjacent-stage Cohen's d falls in the target band ~0.5-0.8
       (Cohen, 1988), versus >=1.5 in Step 2.

Explicitly DEFERRED to Step 5 (per agreed sequencing): MDL-based
threshold estimation. This step keeps the Step 3 placeholder threshold
and tests ONLY whether JSD detection sensitivity degrades gracefully
under realistic noise and weaker separation -- changing one variable
group at a time preserves diagnostic power.

References
----------
[1] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.). Routledge. ISBN 978-0-8058-0283-2.
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
    StageLabels,
    StageLabelSource,
    TimeFields,
)
from data.synthetic_generator import (
    CATEGORY_POOL,
    ITEM_POOL,
    PROGRAM_POOL,
    SESSION_INDEX_TO_DAY_OFFSET,
    SESSION_INDEX_TO_STAGE,
    _sample_interaction_type,
)

MASTER_SEED = 42
LEAK_PROBABILITY = 0.20  # fixed 80/20 nominal/leak split, confirmed 2026-06-24

# Stage adjacency for leakage: a leaked event is generated from a
# NEIGHBORING stage's distribution (chain topology S1 - S2 - S3).
STAGE_NEIGHBORS: Dict[DecisionStage, Tuple[DecisionStage, ...]] = {
    DecisionStage.EXPLORATION: (DecisionStage.COMPARISON,),
    DecisionStage.COMPARISON: (DecisionStage.EXPLORATION, DecisionStage.FINALIZATION),
    DecisionStage.FINALIZATION: (DecisionStage.COMPARISON,),
}

# Weaker, realistic continuous-signal parameters (mean, std) per stage.
# Target adjacent-stage Cohen's d ~= 0.6-0.75 (see _verify_separability_band).
CONTINUOUS_PARAMS_V2: Dict[DecisionStage, Dict[str, Tuple[float, float]]] = {
    DecisionStage.EXPLORATION: {
        "hover_duration_ms": (800.0, 150.0),
        "dwell_time_ms": (2000.0, 300.0),
        "revisit_count": (1.0, 1.5),
        "scroll_depth_pct": (30.0, 8.0),
    },
    DecisionStage.COMPARISON: {
        "hover_duration_ms": (900.0, 150.0),
        "dwell_time_ms": (2180.0, 300.0),
        "revisit_count": (2.0, 1.5),
        "scroll_depth_pct": (36.0, 8.0),
    },
    DecisionStage.FINALIZATION: {
        "hover_duration_ms": (1000.0, 150.0),
        "dwell_time_ms": (2360.0, 300.0),
        "revisit_count": (3.0, 1.5),
        "scroll_depth_pct": (42.0, 8.0),
    },
}


def _verify_separability_band(min_d: float = 0.45, max_d: float = 0.85) -> None:
    """Fails fast at import time if adjacent-stage Cohen's d (computed from
    the PARAMETERS, equal-std pooled formula) falls outside the intended
    weak-separability band -- prevents an accidental return to Step 2's
    strong separability or an unintentionally too-weak signal."""
    stages = [DecisionStage.EXPLORATION, DecisionStage.COMPARISON, DecisionStage.FINALIZATION]
    for signal in CONTINUOUS_PARAMS_V2[stages[0]]:
        for a, b in zip(stages[:-1], stages[1:]):
            mean_a, std_a = CONTINUOUS_PARAMS_V2[a][signal]
            mean_b, std_b = CONTINUOUS_PARAMS_V2[b][signal]
            pooled_std = ((std_a ** 2 + std_b ** 2) / 2.0) ** 0.5
            d = abs(mean_b - mean_a) / pooled_std
            if not (min_d <= d <= max_d):
                raise ValueError(
                    f"Separability out of target band for '{signal}' between "
                    f"{a.value} and {b.value}: d={d:.3f}, expected in "
                    f"[{min_d}, {max_d}]"
                )


_verify_separability_band()


def _sample_continuous_v2(rng: np.random.Generator, effective_stage: DecisionStage) -> ContinuousSignals:
    params = CONTINUOUS_PARAMS_V2[effective_stage]
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


def _select_effective_stage(
    rng: np.random.Generator, nominal_stage: DecisionStage, leak_probability: float
) -> DecisionStage:
    """With probability `leak_probability`, returns a randomly chosen
    NEIGHBORING stage instead of the nominal one (bleed/noise)."""
    if rng.random() < leak_probability:
        neighbors = STAGE_NEIGHBORS[nominal_stage]
        idx = int(rng.integers(0, len(neighbors)))
        return neighbors[idx]
    return nominal_stage


def generate_synthetic_dataset_v2(
    n_users: int = 20,
    n_sessions: int = 3,
    events_per_session: int = 20,
    leak_probability: float = LEAK_PROBABILITY,
    seed: int = MASTER_SEED,
) -> List[InteractionRecord]:
    """
    Generates the Step 4 refined synthetic dataset: larger scale,
    probabilistic neighbor-stage leakage, weaker separability.

    `stage_true` on each record holds the ACTUAL generating
    (post-leakage) stage, not the session's nominal stage -- this is the
    realistic ground truth a stage detector must recover.

    Parameters
    ----------
    n_users : int
        Number of synthetic users (default 20).
    n_sessions : int
        Sessions per user (default 3; must be 3, 1:1 with S1/S2/S3).
    events_per_session : int
        Events per session (default 20).
    leak_probability : float
        Per-event probability of generating from a neighboring stage
        instead of the session's nominal stage (default 0.20).
    seed : int
        Master seed for the numpy Generator (default 42).

    Returns
    -------
    List[InteractionRecord]
    """
    if n_sessions != 3:
        raise ValueError("This generator assumes exactly 3 sessions (1:1 with S1/S2/S3).")

    rng = np.random.default_rng(seed)
    base_date = datetime(2026, 6, 24, 9, 0, 0)
    records: List[InteractionRecord] = []

    for user_idx in range(n_users):
        user_id = f"u_{user_idx:03d}"

        for session_idx in range(n_sessions):
            nominal_stage = SESSION_INDEX_TO_STAGE[session_idx]
            session_id = f"{user_id}_s{session_idx}"
            day_offset = SESSION_INDEX_TO_DAY_OFFSET[session_idx]
            session_start = base_date + timedelta(days=day_offset)

            prev_timestamp = None
            for event_idx in range(events_per_session):
                gap_s = float(rng.uniform(20.0, 90.0)) if event_idx > 0 else 0.0
                timestamp = session_start if event_idx == 0 else prev_timestamp + timedelta(seconds=gap_s)
                prev_timestamp = timestamp

                effective_stage = _select_effective_stage(rng, nominal_stage, leak_probability)

                identifiers = Identifiers(
                    user_id=user_id,
                    item_id=str(rng.choice(ITEM_POOL)),
                    session_id=session_id,
                    event_id=f"{session_id}_e{event_idx:03d}",
                )
                categorical = CategoricalFeatures(
                    item_category=str(rng.choice(CATEGORY_POOL)),
                    program_type=str(rng.choice(PROGRAM_POOL)),
                    interaction_type=_sample_interaction_type(rng, effective_stage),
                )
                continuous = _sample_continuous_v2(rng, effective_stage)
                time_fields = TimeFields(
                    timestamp=timestamp,
                    event_index=event_idx,
                    time_since_start_s=(timestamp - session_start).total_seconds(),
                    time_since_previous_event_s=None if event_idx == 0 else gap_s,
                )
                stage_labels = StageLabels(
                    stage_true=effective_stage,  # actual generating stage, not nominal
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
    data = generate_synthetic_dataset_v2()
    print(f"Generated {len(data)} synthetic records (Step 4, realistic).")
    leaked = sum(
        1
        for r in data
        if r.stage.stage_true != SESSION_INDEX_TO_STAGE[int(r.identifiers.session_id.split("_s")[-1])]
    )
    print(f"Empirical leak rate: {leaked / len(data):.3f} (target ~{LEAK_PROBABILITY})")
