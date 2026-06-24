"""
interaction_matrix.py

Phase 2, Step 1: bridges Phase 1's event-level InteractionRecord schema to
the LightFM-compatible binary implicit-feedback format, per the frozen
Step 0 interface (Phase2_Step0_Model_Interface_Freeze.md).

Produces, from a single pass over the records:
    - user_id_to_idx, item_id_to_idx     : deterministic (sorted) vocabularies
    - item_feature_vocab                 : {feature_name: {value: idx}}
    - item_feature_idx_by_item           : item_idx -> (category_idx, program_idx)
    - positive_pairs                     : stage-collapsed (user_idx, item_idx) set
    - positive_pairs_by_stage            : stage_value -> (user_idx, item_idx) set

Explicit assumption (documented, confirmed 2026-06-25): the Phase 1
synthetic generators sample `item_category` / `program_type` independently
PER EVENT rather than as a fixed catalog property of `item_id`. This module
resolves each item's metadata as the MODE (most frequent value across all
events referencing that item_id; ties broken alphabetically for
determinism) -- a placeholder appropriate for synthetic data only. Real
catalog data (Prolific full study) will supply item metadata as a genuine,
time-invariant per-item property, making this resolution step unnecessary.

Continuous behavioral signals (hover_duration_ms, dwell_time_ms,
revisit_count, scroll_depth_pct) are NEVER read by this module -- per the
Step 0 interface, they are consumed only by the Phase 1 JSD/MDL pipeline,
never by the recommendation model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, Set, Tuple

from data.interaction_schema import InteractionRecord

ITEM_FEATURE_NAMES: Tuple[str, ...] = ("item_category", "program_type")


@dataclass(frozen=True)
class InteractionMatrixBundle:
    user_id_to_idx: Dict[str, int]
    item_id_to_idx: Dict[str, int]
    item_feature_vocab: Dict[str, Dict[str, int]]
    item_feature_idx_by_item: Dict[int, Tuple[int, int]]  # item_idx -> (cat_idx, prog_idx)
    positive_pairs: Set[Tuple[int, int]]
    positive_pairs_by_stage: Dict[str, Set[Tuple[int, int]]]

    @property
    def n_users(self) -> int:
        return len(self.user_id_to_idx)

    @property
    def n_items(self) -> int:
        return len(self.item_id_to_idx)

    @property
    def n_categories(self) -> int:
        return len(self.item_feature_vocab["item_category"])

    @property
    def n_programs(self) -> int:
        return len(self.item_feature_vocab["program_type"])


def _mode(counter: Counter) -> str:
    """Most frequent value; ties broken alphabetically for determinism."""
    max_count = max(counter.values())
    tied = sorted(value for value, count in counter.items() if count == max_count)
    return tied[0]


def build_interaction_matrix(records: Sequence[InteractionRecord]) -> InteractionMatrixBundle:
    """Builds the full Step 0-specified bundle from a list of InteractionRecord.
    Reads ONLY Identifiers, CategoricalFeatures (item_category, program_type),
    and StageLabels (stage_true preferred, falling back to stage_inferred_jsd).
    ContinuousSignals and TimeFields are never accessed."""
    user_ids = sorted({r.identifiers.user_id for r in records})
    item_ids = sorted({r.identifiers.item_id for r in records})
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    category_counts_by_item: Dict[str, Counter] = defaultdict(Counter)
    program_counts_by_item: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        category_counts_by_item[r.identifiers.item_id][r.categorical.item_category] += 1
        program_counts_by_item[r.identifiers.item_id][r.categorical.program_type] += 1

    item_category_by_item = {iid: _mode(category_counts_by_item[iid]) for iid in item_ids}
    item_program_by_item = {iid: _mode(program_counts_by_item[iid]) for iid in item_ids}

    category_values = sorted(set(item_category_by_item.values()))
    program_values = sorted(set(item_program_by_item.values()))
    item_feature_vocab = {
        "item_category": {v: i for i, v in enumerate(category_values)},
        "program_type": {v: i for i, v in enumerate(program_values)},
    }

    item_feature_idx_by_item: Dict[int, Tuple[int, int]] = {}
    for iid in item_ids:
        item_idx = item_id_to_idx[iid]
        cat_idx = item_feature_vocab["item_category"][item_category_by_item[iid]]
        prog_idx = item_feature_vocab["program_type"][item_program_by_item[iid]]
        item_feature_idx_by_item[item_idx] = (cat_idx, prog_idx)

    positive_pairs: Set[Tuple[int, int]] = set()
    positive_pairs_by_stage: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    for r in records:
        u_idx = user_id_to_idx[r.identifiers.user_id]
        i_idx = item_id_to_idx[r.identifiers.item_id]
        positive_pairs.add((u_idx, i_idx))
        stage = r.stage.stage_true if r.stage.stage_true is not None else r.stage.stage_inferred_jsd
        if stage is not None:
            positive_pairs_by_stage[stage.value].add((u_idx, i_idx))

    return InteractionMatrixBundle(
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        item_feature_vocab=item_feature_vocab,
        item_feature_idx_by_item=item_feature_idx_by_item,
        positive_pairs=positive_pairs,
        positive_pairs_by_stage=dict(positive_pairs_by_stage),
    )


if __name__ == "__main__":
    from data.synthetic_generator_v2 import generate_synthetic_dataset_v2

    dataset = generate_synthetic_dataset_v2()
    bundle = build_interaction_matrix(dataset)
    print(f"n_users={bundle.n_users}, n_items={bundle.n_items}, "
          f"n_categories={bundle.n_categories}, n_programs={bundle.n_programs}")
    print(f"positive_pairs (pooled): {len(bundle.positive_pairs)}")
    for stage, pairs in bundle.positive_pairs_by_stage.items():
        print(f"  stage={stage}: {len(pairs)} pairs")
