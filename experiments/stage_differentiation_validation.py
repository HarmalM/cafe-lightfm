"""
stage_differentiation_validation.py

Phase 3, Step 6: Stage-Differentiation Validation
(reframed from "interpretability analysis" -- see
CAFE_LightFM_Phase3_Handoff_Step5_to_Step6.md, Section 8).

--------------------------------------------------------------------------
BINDING SCOPE (confirmed with the researcher, 2026-07-06)
--------------------------------------------------------------------------
- Synthetic v3 dataset ONLY (data/synthetic_generator_v3.py, seed=42).
- Diagnostic / validation-only. This script makes NO claims of:
    * scientific superiority of CAFE-LightFM over any baseline;
    * semantic interpretability -- synthetic v3's `category` and
      `program_type` indices carry no confirmed real-world meaning
      (e.g. no index is known to represent "tuition band" or
      "visa success rate"). Semantic interpretability is deferred to
      Prolific pilot/full-study data (proposal Section 5.3).
- This script validates ONLY that the Stage-Conditioned Attention (SCA)
  layer produces attention allocations over its two item-side metadata
  features that are (a) valid probability distributions and (b)
  numerically differentiated across S1/S2/S3. This is a NECESSARY but
  NOT SUFFICIENT condition for eventual interpretability.

--------------------------------------------------------------------------
CORRECTED DESIGN VS. THE ORIGINAL HANDOFF (Section 8)
--------------------------------------------------------------------------
The handoff document proposed a 30(item) x 3(stage) alpha matrix, under
the assumption that SCA attends over the full 30-item catalog as
"features." Direct inspection of the actual implementation
(models/cafe_lightfm/sca_layer.py, models/cafe_lightfm/cafe_lightfm.py,
supplied 2026-07-06) shows this assumption was incorrect:

    StageConditionedAttention.forward() computes
        logit(f, s_j) = w_base . e_f + w_{s_j} . e_f      (f in {category, program})
        alpha(f, s_j) = softmax_f( logit(f, s_j) )
    over exactly TWO metadata features per item -- item_category and
    program_type (n_features = 2, fixed by construction: see
    CAFELightFM.item_representation(), which stacks exactly
    [category_embedding, program_embedding]). item_id itself is
    deliberately excluded from attention (sca_layer.py docstring: no
    shared vocabulary across items). There is also no stage-only bias
    b_{s_j} in the actual implementation -- it was removed because it is
    shift-invariant under softmax (mathematically inert), per
    sca_layer.py's documented correction to the proposal's Section 5.2
    equation.

    alpha therefore has shape (batch, 2), NOT (batch, 30).

Corrected design (confirmed 2026-07-06):
    1. PRIMARY matrix: 2 features (category, program) x 3 stages,
       averaged across all 30 catalog items per stage.
    2. APPENDIX table: 30 items x 2 features x 3 stages (full item-level
       detail, for transparency / reproducibility).
    3. Item-level diagnostic table: top-5 / bottom-5 items by
       alpha_category, per stage -- replaces the original "top-N
       feature" table (not meaningful with only 2 features). Explicitly
       labeled as an ITEM-level diagnostic of attention allocation
       variation, not feature-semantic interpretability.
    4. Normalization check (corrected): each RAW (item, stage) alpha
       vector over the two features sums to 1.0 (softmax guarantee).
       After averaging across the 30 items, each STAGE COLUMN of the
       aggregated 2x3 matrix also sums to 1.0 (average of vectors that
       each individually sum to 1). Both are checked.
    5. Pairwise Jensen-Shannon divergence (Lin, 1991) [2] between the
       three stage-level 2-element attention distributions, reported as
       a DESCRIPTIVE differentiation measure only -- no p-value or null
       hypothesis is attached to JSD in this script.

--------------------------------------------------------------------------
ZERO RE-IMPLEMENTATION RULE
--------------------------------------------------------------------------
This script reuses CAFELightFM / StageConditionedAttention exactly as
defined in models/cafe_lightfm/cafe_lightfm.py and
models/cafe_lightfm/sca_layer.py. No attention, softmax, or scoring
logic is reimplemented here -- alpha is obtained by calling
CAFELightFM.item_representation() directly, which is the single source
of truth for alpha in the trained model.

--------------------------------------------------------------------------
BUNDLE RECONSTRUCTION (confirmed via direct source inspection, no longer
an assumption -- see data/synthetic_generator_v3.py and
data/interaction_matrix.py, provided 2026-07-06)
--------------------------------------------------------------------------
To compute alpha for the 30 catalog items, this script needs each item's
(category_idx, program_idx). This is a two-step reconstruction, confirmed
directly from source:
    1. `generate_synthetic_dataset_v3(seed=42, ...)` in
       data/synthetic_generator_v3.py returns a List[InteractionRecord]
       (event-level records) -- NOT a bundle directly.
    2. `build_interaction_matrix(records)` in data/interaction_matrix.py
       converts those records into an `InteractionMatrixBundle`, exposing
       `.item_feature_idx_by_item` (item_idx -> (cat_idx, prog_idx)) and
       `.n_items`, matching exactly what this script consumes.

Both functions use module-level defaults (n_users=20, n_items=30,
seed=42 via MASTER_SEED) that match the dimensions independently
inferred from the trained checkpoint (n_users=20, n_items=30) in the
first Colab run of this script -- confirming the default-parameter
reconstruction reproduces the same bundle used at training time.

NOTE on item_category resolution (data/interaction_matrix.py docstring):
`item_category` / `program_type` are sampled independently PER EVENT in
the v3 generator (not as a fixed per-item catalog property), then
resolved to a single value per item_id via mode (most frequent value
across that item's events, ties broken alphabetically). This is a
documented placeholder appropriate for synthetic data only -- it explains
why the checkpoint's inferred `n_categories=1`: with `item_category`
assigned independently at random per event and resolved by mode per
item, the realized category vocabulary can collapse to very few (here,
one) distinct values. This is a property of the synthetic-v3 pipeline,
not a bug in this Step 6 script, and is noted in the Step 6 report as a
factor limiting the diagnostic value of item-level category attention
differentiation on this dataset specifically.

--------------------------------------------------------------------------
References
--------------------------------------------------------------------------
[1] Kula, M. (2015). Metadata Embeddings for User and Item Cold-start
    Recommendations. RecSys 2015 Workshop on New Trends in CBRS.
    CEUR-WS, Vol. 1448, 14-21.
[2] Lin, J. (1991). Divergence Measures Based on the Shannon Entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
    DOI: 10.1109/18.61115.
[3] Jarvelin, K., & Kekalainen, J. (2002). Cumulated Gain-based
    Evaluation of IR Techniques. ACM TOIS, 20(4), 422-446.
    DOI: 10.1145/582415.582418.  (methodological precedent for this
    project's evaluation-protocol documentation style, not directly
    used in this script's computations)
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.cafe_lightfm.cafe_lightfm import CAFELightFM

STAGE_LABELS = ("S1", "S2", "S3")  # assumption: stage_idx 0/1/2 -> S1/S2/S3,
# consistent with project convention (Phase 2 stage encoding). Please
# confirm this ordering matches the encoding used at training time.
FEATURE_LABELS = ("category", "program")


# ==========================================================================
# 1. Dimension inference and model reconstruction (no hard-coded shapes)
# ==========================================================================

@dataclass
class InferredDims:
    n_users: int
    n_items: int
    n_categories: int
    n_programs: int
    n_stages: int
    embedding_dim: int


def infer_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> InferredDims:
    """Infers all CAFELightFM constructor dimensions directly from a
    loaded state_dict's tensor shapes. No dimension is hard-coded.

    Raises
    ------
    KeyError if an expected parameter is missing (reported verbatim so
    the researcher can see exactly which key was not found, rather than
    failing silently or guessing a default).
    """
    required_keys = [
        "user_embedding.weight",
        "item_embedding.weight",
        "category_embedding.weight",
        "program_embedding.weight",
        "sca.w_stage.weight",
    ]
    missing = [k for k in required_keys if k not in state_dict]
    if missing:
        raise KeyError(
            f"Checkpoint state_dict is missing expected key(s): {missing}. "
            f"Available keys: {sorted(state_dict.keys())}"
        )

    n_users, embedding_dim = state_dict["user_embedding.weight"].shape
    n_items = state_dict["item_embedding.weight"].shape[0]
    n_categories = state_dict["category_embedding.weight"].shape[0]
    n_programs = state_dict["program_embedding.weight"].shape[0]
    n_stages = state_dict["sca.w_stage.weight"].shape[0]

    return InferredDims(
        n_users=int(n_users),
        n_items=int(n_items),
        n_categories=int(n_categories),
        n_programs=int(n_programs),
        n_stages=int(n_stages),
        embedding_dim=int(embedding_dim),
    )


def load_model(checkpoint_path: str) -> Tuple[CAFELightFM, InferredDims]:
    """Loads a CAFE-LightFM checkpoint, inferring all dimensions from the
    state_dict itself (per confirmed decision: do not hard-code shapes)."""
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Unexpected checkpoint format at {checkpoint_path}: "
            f"expected a dict or an object with a 'model_state_dict' key, "
            f"got {type(raw)}."
        )

    dims = infer_dims_from_state_dict(state_dict)
    model = CAFELightFM(
        n_users=dims.n_users,
        n_items=dims.n_items,
        n_categories=dims.n_categories,
        n_programs=dims.n_programs,
        n_stages=dims.n_stages,
        embedding_dim=dims.embedding_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, dims


# ==========================================================================
# 2. Bundle loading (item -> category/program mapping)
# ==========================================================================

def load_bundle(seed: int = 42):
    """Rebuilds the synthetic v3 InteractionMatrixBundle to obtain each
    item's (category_idx, program_idx), via the two confirmed source
    functions (data/synthetic_generator_v3.py,
    data/interaction_matrix.py -- see module docstring, "BUNDLE
    RECONSTRUCTION"). No re-implementation of dataset generation or
    bundle-building logic occurs here; both steps call the project's
    existing functions unmodified."""
    from data.synthetic_generator_v3 import generate_synthetic_dataset_v3
    from data.interaction_matrix import build_interaction_matrix

    records = generate_synthetic_dataset_v3(seed=seed)
    return build_interaction_matrix(records)


# ==========================================================================
# 3. Alpha extraction (reuses CAFELightFM.item_representation -- zero
#    re-implementation of attention logic)
# ==========================================================================

def compute_item_stage_alpha(
    model: CAFELightFM,
    item_feature_idx_by_item: Dict[int, Tuple[int, int]],
    n_items: int,
    n_stages: int,
) -> torch.Tensor:
    """Computes alpha(item, stage, feature) for every (item, stage) pair
    in the catalog, by calling the model's own item_representation()
    directly -- no attention math is duplicated here.

    Returns
    -------
    torch.Tensor, shape (n_items, n_stages, 2)
        alpha[:, :, 0] = alpha_category, alpha[:, :, 1] = alpha_program.
    """
    alpha_out = torch.zeros(n_items, n_stages, 2)
    with torch.no_grad():
        for stage_idx in range(n_stages):
            cat_list, prog_list, item_list = [], [], []
            for item_idx in range(n_items):
                cat_idx, prog_idx = item_feature_idx_by_item[item_idx]
                cat_list.append(cat_idx)
                prog_list.append(prog_idx)
                item_list.append(item_idx)

            item_t = torch.tensor(item_list, dtype=torch.long)
            cat_t = torch.tensor(cat_list, dtype=torch.long)
            prog_t = torch.tensor(prog_list, dtype=torch.long)
            stage_t = torch.full((n_items,), stage_idx, dtype=torch.long)

            _, alpha = model.item_representation(item_t, cat_t, prog_t, stage_t)
            alpha_out[:, stage_idx, :] = alpha

    return alpha_out


# ==========================================================================
# 4. Normalization checks (corrected per confirmed decision)
# ==========================================================================

def check_raw_row_normalization(alpha: torch.Tensor, tol: float = 1e-5) -> bool:
    """Each RAW (item, stage) alpha vector over the 2 features must sum
    to 1.0 (softmax guarantee)."""
    row_sums = alpha.sum(dim=-1)  # (n_items, n_stages)
    return bool(torch.all(torch.abs(row_sums - 1.0) < tol))


def aggregate_by_stage(alpha: torch.Tensor) -> torch.Tensor:
    """Averages alpha across all catalog items -> (2 features, n_stages),
    i.e. the PRIMARY 2 x n_stages matrix."""
    return alpha.mean(dim=0).transpose(0, 1)  # (2, n_stages)


def check_aggregated_column_normalization(agg: torch.Tensor, tol: float = 1e-5) -> bool:
    """Each STAGE COLUMN of the aggregated (2, n_stages) matrix must sum
    to 1.0 (average of vectors that each individually sum to 1)."""
    col_sums = agg.sum(dim=0)  # (n_stages,)
    return bool(torch.all(torch.abs(col_sums - 1.0) < tol))


# ==========================================================================
# 5. Jensen-Shannon divergence (descriptive only -- Lin, 1991 [2])
# ==========================================================================

def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two discrete distributions
    (base-2 log, bounded in [0, 1]). Descriptive differentiation measure
    only -- NOT a hypothesis test; no p-value is produced or implied."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.sum(a * torch.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pairwise_jsd(agg: torch.Tensor, stage_labels=STAGE_LABELS) -> Dict[str, float]:
    """Pairwise JSD between the n_stages columns of the aggregated
    (2, n_stages) matrix."""
    n_stages = agg.shape[1]
    labels = stage_labels[:n_stages] if len(stage_labels) >= n_stages else [
        f"S{i+1}" for i in range(n_stages)
    ]
    result = {}
    for i in range(n_stages):
        for j in range(i + 1, n_stages):
            key = f"JSD({labels[i]},{labels[j]})"
            result[key] = jsd(agg[:, i], agg[:, j])
    return result


# ==========================================================================
# 6. Top / bottom item diagnostic table (replaces original top-N feature
#    table -- not meaningful with only 2 features; see module docstring)
# ==========================================================================

def top_bottom_items_by_alpha_category(
    alpha: torch.Tensor, k: int = 5
) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """For each stage, returns the top-k and bottom-k items by
    alpha_category. This is an ITEM-level diagnostic of how attention
    allocation varies across items within a stage -- NOT a feature-
    semantic interpretability claim (synthetic v3 items carry no
    confirmed real-world meaning)."""
    n_items, n_stages, _ = alpha.shape
    out: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for stage_idx in range(n_stages):
        stage_label = STAGE_LABELS[stage_idx] if stage_idx < len(STAGE_LABELS) else f"S{stage_idx+1}"
        cat_vals = alpha[:, stage_idx, 0]  # (n_items,)
        sorted_idx = torch.argsort(cat_vals, descending=True)
        top_k = [(int(i), float(cat_vals[i])) for i in sorted_idx[:k]]
        bottom_k = [(int(i), float(cat_vals[i])) for i in sorted_idx[-k:].flip(0)]
        out[stage_label] = {"top": top_k, "bottom": bottom_k}
    return out


# ==========================================================================
# 7. Output writers (CSV + Markdown + heatmap)
# ==========================================================================

def write_item_level_appendix_csv(alpha: torch.Tensor, out_path: Path) -> None:
    n_items, n_stages, _ = alpha.shape
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "stage", "alpha_category", "alpha_program"])
        for item_idx in range(n_items):
            for stage_idx in range(n_stages):
                stage_label = STAGE_LABELS[stage_idx] if stage_idx < len(STAGE_LABELS) else f"S{stage_idx+1}"
                a_cat = float(alpha[item_idx, stage_idx, 0])
                a_prog = float(alpha[item_idx, stage_idx, 1])
                writer.writerow([item_idx, stage_label, f"{a_cat:.6f}", f"{a_prog:.6f}"])


def write_aggregated_matrix_csv(agg: torch.Tensor, out_path: Path) -> None:
    n_stages = agg.shape[1]
    labels = [STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"S{i+1}" for i in range(n_stages)]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature"] + labels)
        for feat_idx, feat_name in enumerate(FEATURE_LABELS):
            row = [feat_name] + [f"{float(agg[feat_idx, s]):.6f}" for s in range(n_stages)]
            writer.writerow(row)


def write_top_bottom_csv(top_bottom: Dict, out_path: Path, k: int) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "type", "rank", "item_id", "alpha_category"])
        for stage_label, data in top_bottom.items():
            for kind in ("top", "bottom"):
                for rank, (item_id, val) in enumerate(data[kind], start=1):
                    writer.writerow([stage_label, kind, rank, item_id, f"{val:.6f}"])


def write_jsd_csv(jsd_values: Dict[str, float], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage_pair", "jsd_bits"])
        for k, v in jsd_values.items():
            writer.writerow([k, f"{v:.6f}"])


def write_markdown_report(
    agg: torch.Tensor,
    jsd_values: Dict[str, float],
    row_check: bool,
    col_check: bool,
    k: int,
    out_path: Path,
) -> None:
    n_stages = agg.shape[1]
    labels = [STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"S{i+1}" for i in range(n_stages)]

    lines = []
    lines.append("# Phase 3, Step 6 — Stage-Differentiation Validation\n")
    lines.append(
        "**Scope (binding):** synthetic-v3 only, diagnostic/validation-only. "
        "No claims of scientific superiority or semantic interpretability are "
        "made in this report. Attention here refers only to the SCA layer's "
        "allocation over two item-side metadata features (`category`, "
        "`program_type`) — item identity is not attended over "
        "(see `models/cafe_lightfm/sca_layer.py`).\n"
    )
    lines.append(
        "**Dataset note:** if `n_categories == 1` for this checkpoint, "
        "`item_category` was resolved (via mode, per "
        "`data/interaction_matrix.py`) to a single value across the "
        "catalog on this synthetic-v3 run — a documented property of "
        "per-event category sampling in the v3 generator, not a defect in "
        "this script. In that case, `alpha_category` differences across "
        "items reflect only shared-embedding interaction with "
        "`w_base`/`w_stage`, not genuine category-content differences.\n"
    )
    lines.append(
        "**Dataset note:** the loaded checkpoint inferred `n_categories=1`. "
        "Therefore, on this synthetic-v3 run, item-level differences in "
        "`alpha_category` should be interpreted cautiously. They do not "
        "represent meaningful real-world category differences; they are only "
        "a diagnostic artifact of the synthetic-v3 metadata construction.\n"
    )
    lines.append("## 1. Aggregated attention matrix (2 features × {} stages)\n".format(n_stages))
    lines.append("Averaged across all catalog items, per stage.\n")
    header = "| feature | " + " | ".join(labels) + " |"
    sep = "|---" * (n_stages + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for feat_idx, feat_name in enumerate(FEATURE_LABELS):
        row = "| {} | ".format(feat_name) + " | ".join(
            f"{float(agg[feat_idx, s]):.4f}" for s in range(n_stages)
        ) + " |"
        lines.append(row)
    lines.append("")
    lines.append(
        f"**Normalization checks:** raw (item, stage) rows sum to 1.0: "
        f"`{row_check}`. Aggregated stage columns sum to 1.0: `{col_check}`.\n"
    )
    lines.append("## 2. Pairwise Jensen–Shannon divergence (descriptive only)\n")
    lines.append(
        "JSD (Lin, 1991, DOI: 10.1109/18.61115) between the aggregated "
        "2-element attention distributions of each stage pair. This is a "
        "**descriptive differentiation measure only** — no p-value or "
        "significance claim is attached.\n"
    )
    lines.append("| stage pair | JSD (bits) |")
    lines.append("|---|---|")
    for k_pair, v in jsd_values.items():
        lines.append(f"| {k_pair} | {v:.4f} |")
    lines.append("")
    lines.append(f"## 3. Item-level diagnostic (top-{k} / bottom-{k} by alpha_category)\n")
    lines.append(
        "This table shows which catalog items receive the highest/lowest "
        "`category` attention weight per stage. It is an **item-level "
        "diagnostic of attention allocation variation**, not a claim about "
        "feature semantics — see `item_level_top_bottom.csv` for full "
        "values.\n"
    )
    lines.append("## 4. Full item-level attention values\n")
    lines.append("See appendix file `item_level_attention_appendix.csv` "
                  "(30 items × 2 features × {} stages).\n".format(n_stages))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(agg: torch.Tensor, out_path: Path) -> None:
    """Saves a Features x Stages heatmap of the aggregated attention
    matrix. Skips silently (with a printed note) if matplotlib is
    unavailable, since the heatmap is a "nice-to-have" per the confirmed
    Step 6 design ('heatmap if practical'), not a blocking deliverable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping heatmap (non-blocking).")
        return

    n_stages = agg.shape[1]
    labels = [STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"S{i+1}" for i in range(n_stages)]

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(agg.numpy(), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_stages))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(FEATURE_LABELS)))
    ax.set_yticklabels(FEATURE_LABELS)
    ax.set_title("Aggregated SCA attention\n(synthetic-v3, diagnostic only)")
    for i in range(len(FEATURE_LABELS)):
        for j in range(n_stages):
            ax.text(j, i, f"{agg[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="mean alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# 8. Main pipeline
# ==========================================================================

def run(checkpoint_path: str, data_seed: int, output_dir: str, top_k: int = 5) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, dims = load_model(checkpoint_path)
    print(f"Loaded checkpoint. Inferred dims: {dims}")

    bundle = load_bundle(seed=data_seed)
    if bundle.n_items != dims.n_items:
        raise ValueError(
            f"Bundle n_items ({bundle.n_items}) does not match checkpoint "
            f"n_items ({dims.n_items}). Confirm the bundle seed/version "
            f"matches the checkpoint's training run before proceeding."
        )
    if dims.n_categories == 1:
        print(
            "NOTE: n_categories=1 (confirmed data/interaction_matrix.py "
            "behavior -- item_category is sampled per-EVENT, not per-item, "
            "then resolved by mode; the realized vocabulary collapsed to a "
            "single value on this synthetic-v3 run). alpha_category "
            "differentiation across items will therefore reflect only "
            "shared-embedding interaction with w_base/w_stage, not "
            "category-content differences. This is a synthetic-v3 dataset "
            "property, not a script defect; recorded in the Step 6 report."
        )

    alpha = compute_item_stage_alpha(
        model, bundle.item_feature_idx_by_item, dims.n_items, dims.n_stages
    )

    row_ok = check_raw_row_normalization(alpha)
    agg = aggregate_by_stage(alpha)
    col_ok = check_aggregated_column_normalization(agg)
    jsd_values = pairwise_jsd(agg)
    top_bottom = top_bottom_items_by_alpha_category(alpha, k=top_k)

    write_item_level_appendix_csv(alpha, out_dir / "item_level_attention_appendix.csv")
    write_aggregated_matrix_csv(agg, out_dir / "aggregated_feature_stage_matrix.csv")
    write_top_bottom_csv(top_bottom, out_dir / "item_level_top_bottom.csv", k=top_k)
    write_jsd_csv(jsd_values, out_dir / "pairwise_jsd.csv")
    write_markdown_report(agg, jsd_values, row_ok, col_ok, top_k, out_dir / "step6_report.md")
    plot_heatmap(agg, out_dir / "attention_heatmap.png")

    print(f"Row normalization check (raw, per item/stage): {row_ok}")
    print(f"Column normalization check (aggregated, per stage): {col_ok}")
    print(f"Pairwise JSD: {jsd_values}")
    print(f"All outputs written to: {out_dir.resolve()}")


# ==========================================================================
# 9. Lightweight smoke tests (self-contained; do NOT require the real
#    checkpoint or the real dataset bundle -- they build a tiny synthetic
#    CAFELightFM instance and a tiny fake item_feature_idx_by_item map, so
#    the extraction / aggregation / JSD / normalization logic can be
#    verified independently and before Colab access is available).
# ==========================================================================

def _build_dummy_bundle_mapping(n_items: int, n_categories: int, n_programs: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    mapping = {}
    for i in range(n_items):
        cat = int(torch.randint(0, n_categories, (1,), generator=g))
        prog = int(torch.randint(0, n_programs, (1,), generator=g))
        mapping[i] = (cat, prog)
    return mapping


def _smoke_test() -> None:
    torch.manual_seed(42)

    n_users, n_items, n_categories, n_programs, n_stages, dim = 5, 6, 3, 3, 3, 4
    model = CAFELightFM(
        n_users=n_users,
        n_items=n_items,
        n_categories=n_categories,
        n_programs=n_programs,
        n_stages=n_stages,
        embedding_dim=dim,
    )
    # Perturb SCA weights away from zero-init so alpha is non-trivial
    # (uniform 0.5/0.5 alpha would trivially pass sum checks but would
    # not exercise the differentiation logic meaningfully).
    with torch.no_grad():
        model.sca.w_base.copy_(torch.randn(dim))
        model.sca.w_stage.weight.copy_(torch.randn(n_stages, dim))

    mapping = _build_dummy_bundle_mapping(n_items, n_categories, n_programs, seed=42)

    alpha = compute_item_stage_alpha(model, mapping, n_items, n_stages)
    assert alpha.shape == (n_items, n_stages, 2), f"Unexpected alpha shape: {alpha.shape}"
    assert torch.isfinite(alpha).all(), "alpha contains NaN/Inf"

    row_ok = check_raw_row_normalization(alpha)
    assert row_ok, "Raw (item, stage) alpha rows do not sum to 1.0"

    agg = aggregate_by_stage(alpha)
    assert agg.shape == (2, n_stages), f"Unexpected aggregated shape: {agg.shape}"
    col_ok = check_aggregated_column_normalization(agg)
    assert col_ok, "Aggregated stage columns do not sum to 1.0"

    jsd_values = pairwise_jsd(agg, stage_labels=("S1", "S2", "S3"))
    assert len(jsd_values) == 3, f"Expected 3 pairwise JSD values, got {len(jsd_values)}"
    for k, v in jsd_values.items():
        assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"{k} JSD out of [0,1] bounds: {v}"

    # Self-JSD sanity check: JSD of a distribution with itself must be 0.
    self_jsd = jsd(agg[:, 0], agg[:, 0])
    assert math.isclose(self_jsd, 0.0, abs_tol=1e-9), f"Self-JSD should be 0.0, got {self_jsd}"

    top_bottom = top_bottom_items_by_alpha_category(alpha, k=2)
    assert set(top_bottom.keys()) == {"S1", "S2", "S3"}
    for stage_label, data in top_bottom.items():
        assert len(data["top"]) == 2 and len(data["bottom"]) == 2

    print("All Step 6 smoke tests passed (self-contained, no checkpoint/bundle required).")


# ==========================================================================
# 10. CLI entry point
# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3, Step 6: Stage-Differentiation Validation (CAFE-LightFM SCA layer)."
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run self-contained smoke tests and exit.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/cafe_lightfm_v3_seed42.pt")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/step6")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    run(args.checkpoint, args.data_seed, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
