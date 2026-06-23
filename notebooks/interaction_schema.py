"""
interaction_schema.py

Canonical interaction-record schema for the CAFE-LightFM Phase 1 data pipeline.

Used by:
    - the synthetic data generator (Phase 1, Step 2)
    - the JSD-based stage-boundary detector (Phase 1, Step 3)
    - the Prolific Academic pilot/full ingestion pipeline (Phase 1, Step 4)

Design decisions (confirmed 2026-06-24):
    1. `identifiers` are kept structurally disjoint from `categorical`,
       `continuous`, and `stage` fields. Identifiers route to embedding
       tables; they are never themselves features (Kula, 2015, Eq. 1).
    2. Stage labels are four separate, mutually-constrained fields
       (`stage_true`, `stage_self_report`, `stage_inferred_jsd`,
       `stage_label_source`) to prevent leakage between synthetic ground
       truth and real-participant estimates.
    3. `stage_true` is populated ONLY for synthetic records and MUST be
       None for real participant data (Pilot/Full) -- no ground truth
       exists for human behavior, only self-report and JSD inference.
    4. Absolute timestamps are retained alongside derived relative-time
       fields, since JSD-based stage-boundary detection (Definition 5.1,
       proposal) operates on inter-event dynamics, not absolute time.

References
----------
[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. Proceedings of the 2nd Workshop on New Trends in
    Content-Based Recommender Systems, RecSys 2015. CEUR-WS, Vol. 1448.
[2] Lin, J. (1991). Divergence measures based on the Shannon entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
    DOI: 10.1109/18.61115
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #

class DecisionStage(str, Enum):
    """The three formally-defined decision stages (proposal, Section 5.1)."""
    EXPLORATION = "S1"
    COMPARISON = "S2"
    FINALIZATION = "S3"


class InteractionType(str, Enum):
    """Atomic behavioral event types recorded by the platform."""
    VIEW = "view"
    HOVER = "hover"
    SHORTLIST_ADD = "shortlist_add"
    SHORTLIST_REMOVE = "shortlist_remove"
    COMPARE_ACTIVATE = "compare_activate"
    SELF_REPORT = "self_report"


class StageLabelSource(str, Enum):
    """
    Documents which stage field is authoritative for a given record.

        SYNTHETIC_GROUND_TRUTH -> synthetic records only; stage_true set.
        SELF_REPORT_PILOT      -> real participant records only;
                                   stage_self_report set.
        JSD_INFERRED            -> set for records of EITHER origin, since
                                   JSD inference is validated against
                                   stage_true (synthetic) and against
                                   stage_self_report (real pilot) alike.
    """
    SYNTHETIC_GROUND_TRUTH = "synthetic_ground_truth"
    SELF_REPORT_PILOT = "self_report_pilot"
    JSD_INFERRED = "jsd_inferred"


# --------------------------------------------------------------------------- #
# Disjoint sub-groups
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Identifiers:
    """Record-routing keys. Never enter the feature-embedding space."""
    user_id: str
    item_id: str
    session_id: str
    event_id: str


@dataclass(frozen=True)
class CategoricalFeatures:
    """One-hot-encodable, discrete-valued attributes."""
    item_category: str
    program_type: str
    interaction_type: InteractionType


@dataclass(frozen=True)
class ContinuousSignals:
    """Real-valued behavioral signals feeding the JSD attention-distribution
    estimate (Definition 5.1, proposal). Units fixed per field."""
    hover_duration_ms: float
    dwell_time_ms: float
    revisit_count: int
    scroll_depth_pct: float  # range: 0.0-100.0


@dataclass(frozen=True)
class TimeFields:
    """Absolute timestamp plus derived relative-time fields."""
    timestamp: datetime
    event_index: int
    time_since_start_s: float
    time_since_previous_event_s: Optional[float]  # None for first event in session


@dataclass
class StageLabels:
    """
    Four-field stage-label group with an enforced consistency rule:
    stage_true is set if and only if the record is synthetic.
    """
    stage_true: Optional[DecisionStage] = None
    stage_self_report: Optional[DecisionStage] = None
    stage_inferred_jsd: Optional[DecisionStage] = None
    stage_label_source: StageLabelSource = StageLabelSource.JSD_INFERRED

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raises ValueError on any origin <-> field inconsistency,
        rather than allowing silent leakage between record origins."""
        src = self.stage_label_source

        if src == StageLabelSource.SYNTHETIC_GROUND_TRUTH:
            if self.stage_true is None:
                raise ValueError(
                    "SYNTHETIC_GROUND_TRUTH requires stage_true to be set."
                )
            if self.stage_self_report is not None:
                raise ValueError(
                    "Synthetic records must not carry stage_self_report "
                    "(no participant exists to self-report)."
                )

        elif src == StageLabelSource.SELF_REPORT_PILOT:
            if self.stage_self_report is None:
                raise ValueError(
                    "SELF_REPORT_PILOT requires stage_self_report to be set."
                )
            if self.stage_true is not None:
                raise ValueError(
                    "Real participant records must not carry stage_true "
                    "(no ground truth exists for human behavior)."
                )

        elif src == StageLabelSource.JSD_INFERRED:
            if self.stage_inferred_jsd is None:
                raise ValueError(
                    "JSD_INFERRED requires stage_inferred_jsd to be set."
                )


# --------------------------------------------------------------------------- #
# Composite record
# --------------------------------------------------------------------------- #

@dataclass
class InteractionRecord:
    """
    Canonical interaction record. Composition, not inheritance: the four
    disjoint sub-groups plus StageLabels are kept as named attributes so
    that identifiers and stage ground truth structurally cannot leak into
    the feature space consumed by the Stage-Conditioned Attention layer
    (Section 5.2, CAFE-LightFM proposal).
    """
    identifiers: Identifiers
    categorical: CategoricalFeatures
    continuous: ContinuousSignals
    time: TimeFields
    stage: StageLabels

    def is_synthetic(self) -> bool:
        """True if this record originates from the synthetic generator."""
        return self.stage.stage_label_source == StageLabelSource.SYNTHETIC_GROUND_TRUTH

    def to_flat_dict(self) -> dict:
        """Flattens the nested record into one dict (e.g. a DataFrame row).
        Field names are prefixed by sub-group to preserve provenance."""
        flat: dict = {}
        for prefix, group in (
            ("id", self.identifiers),
            ("cat", self.categorical),
            ("cont", self.continuous),
            ("time", self.time),
            ("stage", self.stage),
        ):
            for k, v in asdict(group).items():
                flat[f"{prefix}__{k}"] = v.value if isinstance(v, Enum) else v
        return flat
