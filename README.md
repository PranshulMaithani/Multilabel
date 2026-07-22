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

## Notebooks

- **[`notebooks/cefr_2_methods.ipynb`](notebooks/cefr_2_methods.ipynb)** — the focused
  **baseline**: **Ordinal Random Forest** and **Ordinal Boosting (LightGBM)**. Fixed
  defaults (no hyperparameter tuning), reports **train / test / full** accuracy (no CV),
  plus per-method feature importance. *Use this one.* Fill in `df`, `FEATURE_COLS` (and
  optionally `FEATURE_GROUPS`) at the top; everything else is automatic.
- **[`notebooks/cefr_10_methods.ipynb`](notebooks/cefr_10_methods.ipynb)** — the wider
  10-method survey the 2 were chosen from. Same fill-in interface.

## Documents

- **[The Two Methods, Explained in Full](docs/CEFR_2_Methods_Explained.md)** — deep study
  guide for the 2-method notebook: what each approach is, the intuition, what we did, and
  exactly how the two model scores become a 0–100 number and get split into bands. *Read this
  alongside the notebook.*
- **[Project Discussion & Method Plan](docs/CEFR_Project_Discussion.md)** — the full,
  self-contained plan: cascade framing, candidate methods, 0–100 conversion, interpretability,
  validation, decision points, and sources.
- **[10 Methods — Detailed & Ranked](docs/CEFR_10_Methods_Ranked.md)** — all 10 candidate
  methods, ranked by chance of clearing 82% accuracy.
- **[Methods Research](docs/CEFR_Methods_Research.md)** — earlier focused deep-dive.

## Status

Design phase — resolving requirements with the senior (see the decision points in the
discussion doc). No dataset committed to this repo.
