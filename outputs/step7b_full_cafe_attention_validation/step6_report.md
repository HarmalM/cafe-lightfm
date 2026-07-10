# Phase 3, Step 6 — Stage-Differentiation Validation

**Scope (binding):** synthetic-v3 only, diagnostic/validation-only. No claims of scientific superiority or semantic interpretability are made in this report. Attention here refers only to the SCA layer's allocation over two item-side metadata features (`category`, `program_type`) — item identity is not attended over (see `models/cafe_lightfm/sca_layer.py`).

**Dataset note:** if `n_categories == 1` for this checkpoint, `item_category` was resolved (via mode, per `data/interaction_matrix.py`) to a single value across the catalog on this synthetic-v3 run — a documented property of per-event category sampling in the v3 generator, not a defect in this script. In that case, `alpha_category` differences across items reflect only shared-embedding interaction with `w_base`/`w_stage`, not genuine category-content differences.

**Dataset note:** the loaded checkpoint inferred `n_categories=1`. Therefore, on this synthetic-v3 run, item-level differences in `alpha_category` should be interpreted cautiously. They do not represent meaningful real-world category differences; they are only a diagnostic artifact of the synthetic-v3 metadata construction.

## 1. Aggregated attention matrix (2 features × 3 stages)

Averaged across all catalog items, per stage.

| feature | S1 | S2 | S3 |
|---|---|---|---|
| category | 0.6144 | 0.6035 | 0.6724 |
| program | 0.3856 | 0.3965 | 0.3276 |

**Normalization checks:** raw (item, stage) rows sum to 1.0: `True`. Aggregated stage columns sum to 1.0: `True`.

## 2. Pairwise Jensen–Shannon divergence (descriptive only)

JSD (Lin, 1991, DOI: 10.1109/18.61115) between the aggregated 2-element attention distributions of each stage pair. This is a **descriptive differentiation measure only** — no p-value or significance claim is attached.

| stage pair | JSD (bits) |
|---|---|
| JSD(S1,S2) | 0.0001 |
| JSD(S1,S3) | 0.0026 |
| JSD(S2,S3) | 0.0037 |

## 3. Item-level diagnostic (top-5 / bottom-5 by alpha_category)

This table shows which catalog items receive the highest/lowest `category` attention weight per stage. It is an **item-level diagnostic of attention allocation variation**, not a claim about feature semantics — see `item_level_top_bottom.csv` for full values.

## 4. Full item-level attention values

See appendix file `item_level_attention_appendix.csv` (30 items × 2 features × 3 stages).
