"""
synthetic_generator_v3.py

Phase 3, Step 1.5: Unsaturated Synthetic Validation Dataset.

MOTIVATION
----------
Phase 3 Step 1 (stage-stratified NDCG@K) exposed a validity problem in the
Step 4 (`synthetic_generator_v2.py`) dataset: with only 10 items and every
event sampling uniformly from the FULL item pool, all 20 users end up
having interacted with all 10 items globally. Under the frozen WARP-loss
global exclusion scope (Phase 2, Step 3), this leaves the LightFM baseline
with zero valid negatives, so its training loss is identically 0.0 for
every epoch -- the baseline is architecturally blocked from learning, not
fairly out-performed. Any downstream comparison (SW-NDCG, Precision@K,
paired t-test) computed on that dataset would reflect this artifact, not
the CAFE-LightFM vs. baseline architectural difference.

THIS IS A VALIDATION-ONLY DATASET. It does not replace real Prolific data
and MUST NOT be used to support final scientific performance claims
(see Section "Important Documentation" in the accompanying decision log).

DESIGN
------
Same philosophy as synthetic_generator_v2.py (InteractionRecord schema,
S1/S2/S3 structure, 80/20 nominal/neighbor-stage leakage, seed=42), with
one structural change to eliminate saturation:

    v2: each event samples item_id ~ Uniform(FULL item pool)              [10 items]
    v3: each (user, session) first draws a FIXED random subset of
        ~30% of a LARGER 30-item catalog; every event in that session
        then samples item_id ~ Uniform(that subset) only.

This bounds each user's global item coverage deterministically:
    max unique items per user <= 3 stages * 9 items/stage = 27 < 30
guaranteeing >= 3 valid global negatives per user by construction (not
merely with high probability), and exactly 21 valid stage-specific
negatives per (user, stage) pair (30 - 9).

References
----------
[1] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to
    Large Vocabulary Image Annotation. IJCAI 2011. (WARP loss / negative
    sampling exclusion-scope rationale.)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Sequence

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
    PROGRAM_POOL,
    SESSION_INDEX_TO_DAY_OFFSET,
    SESSION_INDEX_TO_STAGE,
    _sample_interaction_type,
)
from data.synthetic_generator_v2 import (
    LEAK_PROBABILITY,
    MASTER_SEED,
    _sample_continuous_v2,
    _select_effective_stage,
)

# --------------------------------------------------------------------------- #
# v3-specific catalog and coverage parameters (confirmed, Phase 3 Step 1.5)
# --------------------------------------------------------------------------- #

N_ITEMS_V3 = 30
COVERAGE_FRACTION = 0.30  # fraction of catalog visible to a user within one stage


def _build_item_pool(n_items: int) -> List[str]:
    return [f"item_{i:03d}" for i in range(n_items)]


def _select_user_stage_item_subset(
    rng: np.random.Generator, item_pool: Sequence[str], coverage_fraction: float
) -> List[str]:
    """Draws a fixed-size random subset (without replacement) of the item
    catalog for one (user, stage) combination. This subset -- not the full
    catalog -- is the sampling universe for every event in that session,
    which is the structural change that eliminates global saturation."""
    subset_size = max(1, round(len(item_pool) * coverage_fraction))
    chosen_idx = rng.choice(len(item_pool), size=subset_size, replace=False)
    return [item_pool[i] for i in sorted(chosen_idx)]


def generate_synthetic_dataset_v3(
    n_users: int = 20,
    n_sessions: int = 3,
    events_per_session: int = 20,
    leak_probability: float = LEAK_PROBABILITY,
    coverage_fraction: float = COVERAGE_FRACTION,
    n_items: int = N_ITEMS_V3,
    seed: int = MASTER_SEED,
) -> List[InteractionRecord]:
    """
    Generates the Phase 3 Step 1.5 unsaturated validation dataset.

    Identical stage semantics, leakage model, and continuous-signal
    parameters to synthetic_generator_v2.py. The only structural change is
    per-(user, stage) item-subset restriction (see module docstring).

    Parameters
    ----------
    n_users : int
        Number of synthetic users (default 20, matches v2).
    n_sessions : int
        Sessions per user (default 3; must be 3, 1:1 with S1/S2/S3).
    events_per_session : int
        Events per session (default 20, matches v2).
    leak_probability : float
        Per-event probability of generating from a neighboring stage's
        distribution instead of the session's nominal stage (default 0.20).
    coverage_fraction : float
        Fraction of the item catalog visible to a user within a single
        (user, stage) session (default 0.30).
    n_items : int
        Total catalog size (default 30, vs. 10 in v2 -- deliberately
        larger to keep per-stage negative counts high: 30*(1-0.3)=21).
    seed : int
        Master seed for the numpy Generator (default 42).

    Returns
    -------
    List[InteractionRecord]
    """
    if n_sessions != 3:
        raise ValueError("This generator assumes exactly 3 sessions (1:1 with S1/S2/S3).")

    rng = np.random.default_rng(seed)
    item_pool = _build_item_pool(n_items)
    base_date = datetime(2026, 6, 24, 9, 0, 0)
    records: List[InteractionRecord] = []

    for user_idx in range(n_users):
        user_id = f"u_{user_idx:03d}"

        for session_idx in range(n_sessions):
            nominal_stage = SESSION_INDEX_TO_STAGE[session_idx]
            session_id = f"{user_id}_s{session_idx}"
            day_offset = SESSION_INDEX_TO_DAY_OFFSET[session_idx]
            session_start = base_date + timedelta(days=day_offset)

            # KEY CHANGE vs. v2: restrict this (user, stage) session to a
            # fixed ~30% subset of the catalog, drawn once per session.
            item_subset = _select_user_stage_item_subset(rng, item_pool, coverage_fraction)

            prev_timestamp = None
            for event_idx in range(events_per_session):
                gap_s = float(rng.uniform(20.0, 90.0)) if event_idx > 0 else 0.0
                timestamp = session_start if event_idx == 0 else prev_timestamp + timedelta(seconds=gap_s)
                prev_timestamp = timestamp

                effective_stage = _select_effective_stage(rng, nominal_stage, leak_probability)

                identifiers = Identifiers(
                    user_id=user_id,
                    item_id=str(rng.choice(item_subset)),  # restricted subset, not full pool
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
                    stage_true=effective_stage,
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
    data = generate_synthetic_dataset_v3()
    print(f"Generated {len(data)} synthetic records (Phase 3 Step 1.5, unsaturated).")
    n_unique_items = len({r.identifiers.item_id for r in data})
    print(f"Catalog size used: {n_unique_items} (pool={N_ITEMS_V3})")
