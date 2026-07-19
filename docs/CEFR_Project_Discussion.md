# CEFR Proficiency → 0–100 Score: Project Discussion & Method Plan

> **Purpose:** single, self-contained document to discuss the approach with the senior.
> Captures the corrected cascade framing, the open decision points, all candidate methods,
> the 0–100 conversion, interpretability, validation, and sources.
> **Last updated:** 2026-07-20

---

## Table of contents

0. [TL;DR](#0-tldr-one-screen)
1. [Problem statement & constraints](#1-problem-statement--constraints)
2. [The corrected framing — the cascade](#2-the-corrected-framing--the-cascade)
3. [The two cases & pipeline configurations](#3-the-two-cases--pipeline-configurations)
4. [Decision points to resolve with the senior](#4-decision-points-to-resolve-with-the-senior)
5. [Target treatment: ordinal + Frank–Hall decomposition](#5-target-treatment-ordinal--frankhall-decomposition)
6. [Candidate methods — top 5](#6-candidate-methods--top-5)
7. [Probabilities → 0–100 conversion](#7-probabilities--0100-conversion)
8. [Interpretability (the "sections" requirement)](#8-interpretability-the-sections-requirement)
9. [Validation protocol (small-n)](#9-validation-protocol-small-n)
10. [Experiment plan — the bake-off](#10-experiment-plan--the-bake-off)
11. [The key hypothesis test: score-threshold vs argmax](#11-the-key-hypothesis-test-score-threshold-vs-argmax)
12. [Calibration & class imbalance](#12-calibration--class-imbalance)
13. [Pitfalls checklist](#13-pitfalls-checklist)
14. [Risks & unknowns to raise](#14-risks--unknowns-to-raise)
15. [ML research agent prompt (v2)](#15-ml-research-agent-prompt-v2)
16. [Sources](#16-sources)
17. [Next steps](#17-next-steps)

---

## 0. TL;DR (one screen)

- **Goal:** from 8–11 selected features, produce a **0–100 score** and a **CEFR band/level
  classification**, beating the senior's **77%** regression baseline to **≥82%**, while
  staying **interpretable** (per-section contribution).
- **Corrected framing:** the 0–100 score is **not** just the final business number — it is the
  **intermediate decision variable**. The intended cascade is:
  **multiclass model → confidences → collapse to one calibrated 0–100 scalar → cut-points on
  that scalar → band/level classification.**
- **The bet:** classifying on the aggregated 0–100 score beats reading `argmax` straight off
  the multiclass head. **This is testable and is the crux experiment.**
- **Two open cases** (to confirm with senior): final output is **6 individual levels** (Case 1)
  or the **3 grouped bands** (Case 2). *Both share the same pipeline; only the number of
  final cut-points differs (5 vs 2).*
- **Recommended models (non-regression):** Explainable Boosting Machine (EBM) and ordinal
  gradient boosting are the top picks for the accuracy + interpretability combo; ordinal
  logistic regression is kept only as a transparent baseline.
- **Precedent:** a published CEFR study hit **82% balanced accuracy A1–C2** with well-chosen
  linguistic features — so the bar is realistic *if the feature groups carry signal*.

---

## 1. Problem statement & constraints

Predict CEFR proficiency from linguistic/behavioral features and emit a continuous
**0–100 score** (hard business requirement), with **per-section interpretability**.

| Item | Value |
|---|---|
| Sample size | ~220 labeled samples |
| Features | 8–11 **feature groups**; exactly **one** feature chosen per group → 8–11 model inputs |
| Feature semantics | Within a group, features are alternative proxies for the same construct |
| Labels available | Raw **6-level CEFR** (A1–C2) **and** a fixed **3-band collapse** |
| 3-band collapse | `{A1, A2, B1}` < `{B2}` < `{C1, C2}` (fixed by business req; ~equal distribution) |
| Continuous ground truth | **None** — the 0–100 score must be *derived* from class probabilities |
| Baseline | Senior reached **77% band accuracy** using regression models |
| Mandate | Use **non-regression** methods ("regression is old") |
| Success bar | **≥82% band accuracy** while keeping interpretability |
| Interpretability need | Each feature group = a different assessment "section"; show each section's contribution |

**Note on "non-regression":** the senior's intent is to move past basic regressors, not to ban
anything with "regression" in its name. Classic *ordinal regression* (proportional-odds
logistic) is technically a regression method — kept only as a baseline (see §5, §6).

---

## 2. The corrected framing — the cascade

The 0–100 score is the **engine** for classification, not just the output. Full pipeline:

```
 raw features (8–11, one per group)
        │
        ▼
 [1] MULTICLASS classifier  ───►  class confidences  p = (p_A1, …, p_C2)  or  (p_band1..3)
        │
        ▼
 [2] collapse confidences to ONE calibrated scalar   score ∈ [0, 100]
        │           (expected value over CEFR/GSE anchors — see §7)
        ▼
 [3] CUT-POINTS on the 0–100 axis  ───►  final band / level classification
        │
        ▼
 [4] per-section attribution (EBM / SHAP)  ───►  "Section 3 = +12, Section 7 = −5"
```

**Why the score can help band classification (the senior's hypothesis):**
- It **pools information across all classes** — a sample torn between A2 and B1 lands at a
  score reflecting that ambiguity rather than a hard, possibly-wrong `argmax`.
- The score is a **monotonic ordinal decision variable**; cut-points on it can be **tuned to
  maximize band accuracy directly** (threshold moving), which `argmax` cannot do.
- It **smooths noise** near class boundaries where a small-n multiclass head is least reliable.

**Caveat:** this only *adds* accuracy if the cut-points are tuned and/or the score is better
calibrated than argmax. If not, the score is still the business deliverable but adds no
accuracy. **This must be tested (see §11).**

---

## 3. The two cases & pipeline configurations

"Individual bands" ≈ the **6 ungrouped CEFR levels** (A1…C2). "Three band" ≈ the **grouped 3**.
Live configurations:

| Config | Multiclass trained on | Score from | Final output | Your case |
|---|---|---|---|---|
| **A** | 6 levels | 6-class confidences | **3 bands** (2 cut-points) | Case 2 (richest) |
| **B** | 6 levels | 6-class confidences | **6 levels** (5 cut-points) | Case 1 |
| **C** | 3 bands | 3-class confidences | **3 bands** (2 cut-points) | Case 2 (simpler) |

*(Train-on-3 → predict-6 is impossible: you cannot recover finer granularity from a coarser
model. Ignore that combo.)*

**Key reassurance:** A, B, and C are **the same machinery**. The *only* difference is how many
cut-points sit on the 0–100 axis at the end (**5** for levels, **2** for bands). Build once
behind an `output_granularity` switch and flip it once the senior confirms — **you are not
blocked** by the ambiguity.

**Why prefer training on 6 levels (Config A) even when the target is 3 bands:**
- **Free accuracy:** collapsing 6-level predictions to 3 bands makes within-band confusions
  (A1↔A2, C1↔C2) cost nothing; only cross-band boundaries (B1↔B2, B2↔C1) still hurt. This is a
  plausible source of the 77% → 82% jump — **but not guaranteed; test against direct 3-band.**
- **Smoother score:** six anchors give a finer-grained 0–100 output than three.

---

## 4. Decision points to resolve with the senior

**Bring these to the meeting — they change the build:**

1. **Train granularity** — does the multiclass model predict the **6 individual levels** or the
   **3 bands**? (Config A/B vs C)
2. **Final output** — is the required classification the **6 levels** or the **3 bands**?
   (Case 1 vs Case 2)
3. **Is the 0–100 a required deliverable itself, or purely an internal step** to sharpen band
   classification? (Determines what we report and calibrate against.)
4. **Cut-points** — fixed from **CEFR/GSE anchors**, or **tuned to maximize band accuracy**?
   (This is what lets the score beat `argmax`.)
5. **Accuracy metric** — plain accuracy, **balanced accuracy**, or an ordinal metric
   (Quadratic Weighted Kappa)? (82% "of what" must be pinned down; the 82% precedent is
   *balanced* accuracy.)
6. **Feature-group choice** — is the 1-of-k selection per group **fixed by domain knowledge**,
   or is choosing it part of my task? (Affects leakage risk and search cost; see §9.)
7. **Is the 3-band collapse itself fixed**, or open to a different grouping? (Confirmed fixed
   so far, but worth re-confirming — B2 alone is an unusual middle band.)

---

## 5. Target treatment: ordinal + Frank–Hall decomposition

The bands/levels are **ordered**, so treat the target as **ordinal**, not flat multiclass —
this typically buys accuracy and yields probabilities that map cleanly to a monotonic score.

**Getting ordinal behavior without "regression":** use **ordinal decomposition
(Frank & Hall, 2001)** on top of non-regression base learners. For `K` ordered classes, train
`K−1` binary classifiers predicting `P(y > level k)`, then difference adjacent cumulative
probabilities.

**Worked example — 3 bands (2 binary classifiers):**

```
P(band1) = 1 − P(y > band1)
P(band2) = P(y > band1) − P(y > band2)
P(band3) = P(y > band2)
```

**6 levels** → 5 binary classifiers `P(y > A1), P(y > A2), …, P(y > C1)`, differenced the same
way. This is almost certainly the "multilabel" framing the senior referred to — each cumulative
threshold is a binary label.

Evidence that ordinal decomposition helps: ordinal decision-tree ensembles have been shown to
**beat their non-ordinal counterparts** on ordinal targets (PMC7517475).

---

## 6. Candidate methods — top 5

Ranked for **n≈220 + ≥82% + interpretability + 0–100 output**.

### #1 — Explainable Boosting Machine (EBM / GA2M)
Glass-box **generalized additive model** (cyclic gradient boosting on shallow trees, optional
pairwise interactions = GA2M). Accuracy comparable to Random Forest / XGBoost, but each
feature's contribution is **exact and additive** — ideal for "section 3 contributed +X."
- **Why #1:** best interpretability with little accuracy cost.
- **Library:** `interpret` (InterpretML).

### #2 — Ordinal gradient boosting (OGBoost, or LightGBM/XGBoost + Frank–Hall)
Highest accuracy ceiling and genuinely ordinal. **OGBoost** (2025) is purpose-built with
**CV-based early stopping for small/imbalanced data**. Alternatively wrap LightGBM/XGBoost
(shallow, `max_depth ≤ 3`, strong regularization, **monotonic constraints**) in Frank–Hall.
Interpret via **SHAP**.
- **Why #2:** best shot at clearing 82%.
- **Library:** `ogboost`, `lightgbm`/`xgboost`, `shap`.

### #3 — Ordinal Random Forest (`ordinalForest`, or RF + Frank–Hall)
On small, categorical-ish data, **bagging is often more stable and accurate than boosting**;
forests give well-behaved probabilities + permutation/SHAP importances. Frequently ties or
beats boosting at n≈220.
- **Why #3:** robust; do not assume boosting wins here.
- **Library:** `ordinalForest` (R) or sklearn RF + Frank–Hall wrapper.

### #4 — SVM (RBF) + ordinal decomposition + Platt calibration
The historical CEFR workhorse over grouped linguistic features; strong small-n performer.
Not natively interpretable, but SHAP/permutation recovers per-section attribution; Platt
scaling gives calibrated probabilities for the 0–100 map.
- **Why #4:** strong small-n accuracy, adds diversity to an ensemble.
- **Library:** sklearn `SVC(probability=True)` + `CalibratedClassifierCV`.

### #5 — Proportional-odds ordinal logistic (POM) — BASELINE ONLY
Transparent coefficients, natural cumulative probabilities. **This IS "ordinal regression"**
(what the senior wants to move past) and is unlikely to be top accuracy. Keep as the honest
reference line everything else must beat.
- **Library:** `mord`, `statsmodels`.

### Ruled out on purpose — CORAL / CORN deep ordinal nets
Excellent methods, but need far more than 220 samples; they overfit and are not the
interpretability you want (arXiv 2111.08851). Know they exist; wrong tool here.

### Comparison

| Method | Small-n robustness | Accuracy ceiling | Interpretability | 0–100 fit |
|---|---|---|---|---|
| EBM (GA2M) | High | High | ★★★ native additive | Expected-value map |
| Ordinal GBM / LGBM+FH | Medium–High | **Highest** | ★★ via SHAP | Expected-value map |
| Ordinal RF | **High** | High | ★★ via SHAP/perm | Expected-value map |
| SVM + FH + Platt | Medium–High | High | ★ via SHAP | Calibrated → map |
| POM (baseline) | High | Medium | ★★★ coefficients | Native cumulative |

**Verdict on "is ordinal regression best?":** No. Ordinal *treatment* of the target is right;
ordinal *regression* (POM) as the model is not the best bet. EBM (#1) and ordinal GBM (#2)
dominate it for the 82% + interpretability goal. Let repeated-CV numbers pick the winner —
at n≈220, #1–#4 sit within noise of each other.

---

## 7. Probabilities → 0–100 conversion

No continuous ground truth exists, so the score is **derived** from class probabilities.

### 7.1 Recommended: expected value over per-level anchors

```
score = Σ_level  P(level) × anchor(level)
```

Using **6-level** probabilities gives a smooth, monotonic output.

### 7.2 Defensible anchors — Pearson Global Scale of English (GSE)

Prefer citable anchors over arbitrary equal spacing. Approx. GSE level midpoints:

| CEFR | GSE range | Midpoint |
|---|---|---|
| A1 | 22–29 | ~26 |
| A2 | 30–35 | ~33 |
| B1 | 43–50 | ~47 |
| B2 | 59–66 | ~63 |
| C1 | 76–84 | ~80 |
| C2 | 85–90 | ~88 |

Rescale the GSE span to a full 0–100 if the business wants it:
`score_0_100 = (gse_score − 22) / (90 − 22) × 100`  (clip to [0, 100]).

### 7.3 Alternatives
- **Cumulative-probability integration** — pairs with Frank–Hall:
  `score ∝ P(y > band1) + P(y > band2)`.
- **Isotonic recalibration** — if a real continuous target ever appears, fit a monotonic map
  from raw model score → the true 0–100 scale.

### 7.4 Non-negotiables
- **Monotonicity:** higher predicted proficiency must never lower the score.
- **Calibrate probabilities first** (Platt/isotonic) or the score is biased.

---

## 8. Interpretability (the "sections" requirement)

Because `score = Σ P(level) · anchor`, and EBM gives exact additive feature terms, each
prediction reads as:

> "Section 3 pushed the score **+12**, Section 7 **−5**, …" — contributions summing to the
> final 0–100.

- **EBM:** native additive shape functions per feature (+ pairwise interactions).
- **SHAP:** same additive decomposition for boosting / RF / SVM — cross-check and for
  non-glass-box candidates.
- **Global view:** feature/section importances. **Local view:** per-sample attribution.
- Map each feature back to its group → report **section-level** contribution to stakeholders.

Precedent: interpretable, feature-selection-based CEFR models achieve competitive accuracy
with more explainable, generalizable behavior (arXiv 2602.13102).

---

## 9. Validation protocol (small-n)

At n≈220, a single CV split swings ±~5%, so a one-fold "we hit 82%" will not survive scrutiny.

- **Repeated stratified k-fold** (e.g., 5-fold × 10 repeats). Report **mean ± std** and a CI,
  not a point estimate.
- **Do the 1-of-k feature-group selection *inside* the CV folds.** Selecting features on the
  full data first leaks and inflates the number — the most common small-n mistake.
- **Ordinal metrics alongside accuracy:**
  - **Quadratic Weighted Kappa (QWK)** — penalizes far errors more than adjacent ones.
  - **MAE on band index** — average ordinal distance of errors.
  - **Spearman / Pearson** of predicted-vs-implied 0–100 — monotonic agreement.
  - **Macro-F1 / balanced accuracy** — guards against a dominant band inflating accuracy.
- **Calibration check** (reliability curve) before trusting the 0–100 map.

### Feature-group selection combinatorics
Choosing 1-of-k across 8–11 groups is a search of size `k^(#groups)` (e.g., 3 options × 10
groups = 59,049 combos). Picking the combo that maximizes CV score **outside** nested CV
overfits badly. Options: (a) nested CV, (b) domain knowledge fixes choices, (c) group-aware
selection. **Confirm with senior (Decision point 6) whether this selection is even my job.**

---

## 10. Experiment plan — the bake-off

Pick the winner by **nested repeated stratified CV** across the matrix below.

**Target framings:**
1. Direct 3-band classifier.
2. 6-level classifier → collapse to 3 bands.
3. Frank–Hall ordinal on 3-band.
4. Frank–Hall ordinal on 6-level → collapse.

**Models (× each framing):** EBM, Ordinal GBM (OGBoost / LGBM+FH), Ordinal RF, SVM+FH.
**Baseline:** POM.

**Pipeline per fold:**
```
raw features
  → 1-of-k feature-group selection (INSIDE fold)
  → base model (ordinal framing)
  → probability calibration (Platt / isotonic)
  → per-level probabilities
  → expected-value map with GSE anchors → 0–100
  → cut-points (tuned inside fold) → classification
  → SHAP / EBM attribution → per-section contributions
```

**Report per model:** accuracy (3-band), balanced accuracy, QWK, MAE-band-index, macro-F1,
calibration curve, and one worked 0–100 explanation example.

---

## 11. The key hypothesis test: score-threshold vs argmax

**This experiment justifies the entire 0–100 detour.** For each model, compare band accuracy of:

- **(a) `argmax`** of the multiclass head (the naive baseline), vs
- **(b) tuned cut-points** on the 0–100 score (cut-points fit *inside* each CV fold).

Interpretation:
- **(b) > (a):** the senior's hypothesis holds — the score genuinely adds classification power.
  Report the lift.
- **(b) ≈ (a):** the score is still the business deliverable, but does not improve accuracy —
  say so honestly.
- Also report whether **6-level → collapse** beats **direct 3-band** (framing 2 vs 1 in §10).

---

## 12. Calibration & class imbalance

- **Calibration:** Platt (sigmoid) or isotonic via `CalibratedClassifierCV`; at n≈220 prefer
  Platt/beta (isotonic can overfit with few samples). Calibrate **before** the 0–100 map.
- **Imbalance:** the 3-band collapse is ~balanced by design, but the 6-level distribution may
  not be. Use **class weights** and **threshold moving**; **avoid heavy resampling** at this n
  (it tends to overfit). Report **balanced accuracy / macro-F1** so a dominant class can't
  inflate the headline.

---

## 13. Pitfalls checklist

- [ ] Feature selection done **inside** CV (no leakage).
- [ ] Repeated CV with mean ± std, not a single lucky split.
- [ ] Cut-points tuned **inside** folds, not on the full data.
- [ ] Probabilities calibrated before the 0–100 map.
- [ ] 0–100 map is monotonic; anchors documented/citable.
- [ ] Ordinal metrics reported, not accuracy alone.
- [ ] Class imbalance handled with weights/thresholds, not heavy resampling.
- [ ] No deep ordinal nets (CORAL/CORN) at this n.
- [ ] Each feature traced back to its section for reporting.
- [ ] `argmax` baseline reported alongside score-threshold (the crux test).

---

## 14. Risks & unknowns to raise

- **Feature signal is the ceiling.** If the 8–11 chosen features don't separate the bands,
  no model reaches 82%. The published 82% used carefully engineered features — ask what the
  feature groups actually measure.
- **82% "of what."** Plain vs balanced accuracy vs QWK materially changes the target.
- **B2 as a lone middle band** is unusual and narrow — its boundaries (B1↔B2, B2↔C1) are where
  most cross-band errors will concentrate. Worth confirming the grouping rationale.
- **Combinatorial feature selection** can silently overfit; needs nested CV or fixed choices.
- **Small n** means wide confidence intervals — a 5-point "win" may not be significant; report
  CIs and consider a paired test across folds.

---

## 15. ML research agent prompt (v2)

```
You are an ML research assistant. Produce a rigorous, citation-backed comparison for an
applied problem. Prioritize SMALL-DATA robustness and INTERPRETABILITY. Be concrete about
algorithms, Python libraries, and tradeoffs. Survey and rank — do NOT write a full solution.

PROBLEM
- Predict CEFR proficiency from linguistic features, and emit a continuous 0–100 score
  (business requirement) with per-"section" interpretability. The 0–100 score is used as an
  INTERMEDIATE decision variable: multiclass model → confidences → collapse to one calibrated
  0–100 scalar → cut-points on that scalar → final band/level classification.
- Labels: raw 6-level CEFR (A1–C2) AND a fixed 3-band collapse {A1,A2,B1} < {B2} < {C1,C2}.
  Accuracy is measured on the bands; the 6-level labels are available as extra signal.
- n ≈ 220 total. Features come as 8–11 "feature groups"; exactly one feature is chosen per
  group (each group = alternative proxies for one construct) → 8–11 model inputs.
- Baseline: a senior reached 77% band accuracy with REGRESSION models. Mandate: use
  NON-regression methods; target ≥82% band accuracy WITHOUT losing interpretability.
- No continuous ground-truth score exists — the 0–100 must be DERIVED from class probabilities.

EVALUATE AND RANK THESE CANDIDATES (add others if warranted):
1. Explainable Boosting Machine (EBM/GA2M, InterpretML).
2. Ordinal gradient boosting: OGBoost, and LightGBM/XGBoost under Frank–Hall (2001) ordinal
   decomposition with shallow trees + monotonic constraints.
3. Ordinal Random Forest (ordinalForest / RF + Frank–Hall).
4. SVM (RBF) + ordinal decomposition + Platt calibration.
5. Proportional-odds ordinal logistic — as an interpretable BASELINE only.
Explicitly assess whether CORAL/CORN deep ordinal nets are viable at n≈220 (I expect not).

ANSWER THESE QUESTIONS, each with methods + pros/cons + small-n suitability + citations:
A. Which candidates most reliably clear ~82% band accuracy at n≈220 without overfitting?
   Cite CEFR/proficiency-classification precedents.
B. Does training a 6-level ordinal model then collapsing to 3 bands beat direct 3-band
   training? Cite evidence both ways.
C. Does classifying via cut-points on a probability-derived 0–100 score beat plain argmax of
   the multiclass head? What is the theory/evidence for score-thresholding vs argmax?
D. PROBABILITY → 0–100: expected-value over per-level anchors vs cumulative-probability
   integration vs isotonic recalibration. Recommend defensible CEFR numeric anchors
   (e.g., Pearson Global Scale of English) and how to rescale to 0–100. Emphasize
   monotonicity + calibration.
E. INTERPRETABILITY: faithful additive per-feature (per-section) attribution that sums to the
   0–100 output — EBM native terms vs SHAP (which explainer per model) vs monotonic GBM.
F. VALIDATION at small n with combinatorial 1-of-k feature-group selection: nested/repeated
   stratified CV, leakage risks, and ordinal metrics (QWK, MAE-on-band-index, Spearman).
G. Probability CALIBRATION for small n (Platt vs isotonic vs beta) and class-imbalance
   handling that is safe at n≈220.

DELIVERABLE
- A ranked shortlist of 3–5 end-to-end pipelines (model → calibration → 0–100 map → cut-points
  → interpretability), each with why it fits n≈220, expected interpretability, failure modes.
- Flag anything likely to overfit at this n. Cite papers and libraries throughout.
```

---

## 16. Sources

- [Predicting CEFR levels: microsystem criterial features (Cambridge, ReCALL)](https://www.cambridge.org/core/journals/recall/article/abs/predicting-cefr-levels-in-learners-of-english-the-use-of-microsystem-criterial-features-in-a-machine-learning-approach/C915A35CD69168EDFB80DE8F57A4328C) — ~82% *balanced* accuracy A1–C2 precedent.
- [Towards interpretable models for CEFR level prediction (Estonian, arXiv 2602.13102)](https://arxiv.org/abs/2602.13102) — interpretable feature-selection approach.
- [OGBoost: Ordinal Gradient Boosting (arXiv 2502.13456)](https://arxiv.org/pdf/2502.13456) — CV early stopping for small/imbalanced data.
- [Ordinal decision-tree ensembles beat non-ordinal counterparts (PMC7517475)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7517475/).
- [RF vs GBM on small categorical datasets (PMC8392226)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8392226/) — bagging more stable at small n.
- [Explainable Boosting Machine / InterpretML docs](https://interpret.ml/docs/ebm.html) — glass-box GA2M.
- [CORAL/CORN rank-consistent ordinal networks (arXiv 2111.08851)](https://arxiv.org/abs/2111.08851) — deep ordinal (ruled out at this n).
- [Automated CEFR-J writing assessment: lexical metrics + AI (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2772766125000205).
- Frank, E. & Hall, M. (2001). *A Simple Approach to Ordinal Classification.* ECML — the ordinal decomposition method.
- Pearson **Global Scale of English (GSE)** — CEFR ↔ numeric anchor mapping used in §7.

---

## 17. Next steps

1. **Meeting:** resolve the 7 decision points in §4 (especially train granularity, final
   output granularity, and whether cut-points are tuned).
2. **Data:** obtain the dataset (or schema + band distribution + feature-group options).
3. **Build:** implement the cascade behind an `output_granularity` switch (covers Configs
   A/B/C) with the §10 bake-off and the §11 argmax-vs-score test.
4. **Report:** accuracy + balanced accuracy + QWK, calibration, and worked per-section
   explanations; compare against the 77% baseline with confidence intervals.
