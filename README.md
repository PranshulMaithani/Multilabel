# Multilabel

CEFR proficiency scoring: predict CEFR bands/levels from 8–11 selected linguistic features
and emit an interpretable **0–100 score**, using non-regression methods. Target: beat the
77% regression baseline to **≥82%** while keeping per-section interpretability.

## Bands (confirmed)

| Band | CEFR levels |
|---|---|
| 0 | A1, A2 |
| 1 | B1 |
| 2 | B2, C1, C2 |

Pipeline: `features → model → probabilities → 0–100 score → 2 cut-points → 3 bands`.

## Notebook

- **[`notebooks/cefr_10_methods.ipynb`](notebooks/cefr_10_methods.ipynb)** — runs all 10
  methods over the same train/test split (built from the `split` column). Fill in `df` and
  `FEATURE_COLS` at the top; everything else is automatic.

## Documents

- **[Project Discussion & Method Plan](docs/CEFR_Project_Discussion.md)** — the full,
  self-contained plan: cascade framing, the two cases, candidate methods, 0–100 conversion,
  interpretability, validation, open decision points, and sources. *Start here.*
- **[10 Methods — Detailed & Ranked](docs/CEFR_10_Methods_Ranked.md)** — all 10 candidate
  methods (5 core + 5 simpler) with full per-method approach detail, ranked by chance of
  clearing 82% accuracy.
- **[Methods Research](docs/CEFR_Methods_Research.md)** — earlier focused deep-dive on the
  core candidate methods.

## Status

Design phase — resolving requirements with the senior (see the decision points in the
discussion doc). No dataset committed to this repo.
