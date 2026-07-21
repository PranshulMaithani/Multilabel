# CEFR Band Prediction → 0–100 Score: Methods Research & Plan

> Working reference for the non-regression, interpretable CEFR scoring task.
> Last updated: 2026-07-20

---

## 1. Problem statement

Predict CEFR proficiency from linguistic/behavioral features and emit a continuous
**0–100 score** (hard business requirement) with **per-section interpretability**.

### Fixed constraints

| Item | Value |
|---|---|
| Sample size | ~220 labeled samples |
| Features | 8–11 **feature groups**; exactly **one** feature chosen per group → 8–11 model inputs |
| Feature semantics | Within a group, features are alternative proxies for the same construct |
| Labels available | Raw **6-level CEFR** (A1–C2) **and** a fixed **3-band collapse** |
| 3-band collapse | `{A1, A2}` < `{B1}` < `{B2, C1, C2}` (confirmed with senior 2026-07-21) |
| Continuous ground truth | **None** — the 0–100 score must be *derived* from class probabilities |
| Baseline | Senior reached **77% band accuracy** using regression models |
| Mandate | Use **non-regression** methods ("regression is old") |
| Success bar | **≥82% band accuracy** while keeping interpretability |
| Interpretability need | Each feature group = a different assessment "section"; show each section's contribution to the score |

### Why the target is ordinal, not plain multiclass

The three bands are **ordered** (`band1 < band2 < band3`). Treating this as flat multiclass
throws away that ordering. Ordinal-aware treatment usually buys a couple of points of accuracy
and produces probabilities that map cleanly onto a monotonic 0–100 scale.

---

## 2. Two structural decisions that shape everything

### 2.1 Model on 6 levels, evaluate/score on 3 bands

You hold the raw A1–C2 labels but only need 3-band accuracy. Exploit that asymmetry:

- **Free accuracy.** Train a 6-level model, then collapse predictions to the 3 bands.
  Confusions *inside* a band (A1↔A2, C1↔C2) stop costing you; only cross-band boundaries
  (B1↔B2, B2↔C1) still hurt. This collapse is a plausible source of the 77% → 82% jump —
  **but it is not guaranteed to beat a direct 3-band model, so test both** (see §7 bake-off).
- **Smoother 0–100 score.** Six ordered anchors give a finer continuous output than three.

### 2.2 "Ordinal, but not regression" — resolved

Classic **ordinal regression = proportional-odds logistic regression (POM)**, which *is* a
regression method. To get the ordinal benefit *without* it being "just regression," use
**ordinal decomposition (Frank & Hall, 2001) on top of non-regression base learners**:

> For `K` ordered classes, train `K−1` binary classifiers predicting `P(y > level k)`
> using **gradient boosting / EBM / random forest** as the engine. Recover per-class
> probabilities by differencing adjacent cumulative probabilities. This is almost certainly
> the "multilabel" framing the senior gestured at.

**Worked example (3 bands):**

```
P(band1) = 1 − P(y > band1)
P(band2) = P(y > band1) − P(y > band2)
P(band3) = P(y > band2)
```

The same pattern extends to the 6-level case with 5 binary classifiers.

---

## 3. Top 5 methods (ranked for n≈220, ≥82%, interpretable, 0–100)

### #1 — Explainable Boosting Machine (EBM / GA2M)
A glass-box **generalized additive model** (cyclic gradient boosting on shallow trees, plus
optional pairwise interactions = GA2M). Accuracy is comparable to Random Forest / XGBoost,
but every feature's contribution is **exact and additive** — ideal for "section 3 contributed
+X to the 0–100." Run as a 6-level classifier or wrapped in Frank–Hall.
- **Pick because:** maximizes interpretability with little accuracy cost. Best interpretability/accuracy trade-off.
- **Library:** `interpret` (InterpretML).

### #2 — Ordinal gradient boosting (OGBoost, or LightGBM/XGBoost + Frank–Hall)
Highest accuracy ceiling and genuinely ordinal. **OGBoost** (2025) is purpose-built and uses
**CV-based early stopping designed for small/imbalanced data**. Alternatively wrap
LightGBM/XGBoost (shallow trees, `max_depth ≤ 3`, strong regularization, **monotonic
constraints**) in Frank–Hall. Interpret via **SHAP** (additive per-section attribution).
- **Pick because:** best shot at actually clearing 82%.
- **Library:** `ogboost`, `lightgbm`/`xgboost`, `shap`.

### #3 — Ordinal Random Forest (`ordinalForest`, or RF + Frank–Hall)
On small datasets with categorical-ish features, **bagging is often more stable and accurate
than boosting**, and forests give well-behaved probabilities plus permutation/SHAP
importances. Lower variance than #2 at n≈220 — frequently ties or beats boosting here.
- **Pick because:** robust; do not assume boosting wins at this n.
- **Library:** `ordinalForest` (R) or scikit-learn RF + Frank–Hall wrapper.

### #4 — SVM (RBF) + ordinal decomposition + Platt calibration
The historical CEFR workhorse over grouped linguistic features; strong small-n performer.
Not natively interpretable, but SHAP/permutation recovers per-section attribution, and Platt
scaling yields calibrated probabilities for the 0–100 map.
- **Pick because:** strong small-n accuracy; good diversity for an ensemble.
- **Library:** scikit-learn `SVC(probability=True)` + `CalibratedClassifierCV`.

### #5 — Proportional-odds ordinal logistic (POM) — interpretable BASELINE only
Transparent coefficients, natural cumulative probabilities. **But this IS "ordinal
regression"** (what the senior wants to move past) and is unlikely to be your top accuracy.
Keep it as the honest reference line everything else must beat.
- **Use because:** transparent baseline, sanity check, calibration reference.
- **Library:** `mord`, `statsmodels`.

### Ruled out on purpose — CORAL / CORN deep ordinal networks
Excellent methods, but they need far more than 220 samples; they will overfit and are not the
interpretability you want. Know they exist; wrong tool here.

### Comparison

| Method | Small-n robustness | Accuracy ceiling | Interpretability | 0–100 fit |
|---|---|---|---|---|
| EBM (GA2M) | High | High | ★★★ native additive | Expected-value map |
| Ordinal GBM / LGBM+FH | Medium–High | **Highest** | ★★ via SHAP | Expected-value map |
| Ordinal RF | **High** | High | ★★ via SHAP/perm | Expected-value map |
| SVM + FH + Platt | Medium–High | High | ★ via SHAP | Calibrated → map |
| POM (baseline) | High | Medium | ★★★ coefficients | Native cumulative |

### Is ordinal regression "the best"? — No
Ordinal *treatment* of the target is right; ordinal *regression* (POM) as the model is not
your best bet. For the 82% bar **and** interpretability, **EBM (#1) and ordinal GBM (#2)
dominate it.** Run the bake-off, keep POM only as the transparent baseline, and let the
repeated-CV numbers pick the winner — at n≈220, #1–#4 sit within noise of each other.

---

## 4. Converting probabilities → 0–100 score

No continuous ground truth exists, so the score must be **derived** from class probabilities.

### 4.1 Recommended: expected value over per-level anchors

```
score = Σ_level  P(level) × anchor(level)
```

Using the **6-level** probabilities gives a smooth, monotonic output.

### 4.2 Defensible anchors — Pearson Global Scale of English (GSE)

Prefer citable anchors over arbitrary equal spacing. Approx. GSE level midpoints:

| CEFR | GSE range | Midpoint |
|---|---|---|
| A1 | 22–29 | ~26 |
| A2 | 30–35 | ~33 |
| B1 | 43–50 | ~47 |
| B2 | 59–66 | ~63 |
| C1 | 76–84 | ~80 |
| C2 | 85–90 | ~88 |

Rescale the GSE span to 0–100 if the business wants the full range:
`score_0_100 = (gse_score − 22) / (90 − 22) × 100` (clip to [0, 100]).

### 4.3 Alternatives
- **Cumulative-probability integration** — pairs naturally with Frank–Hall:
  `score ∝ P(y > band1) + P(y > band2)`.
- **Isotonic recalibration** — if a real continuous target ever appears, fit a monotonic map
  from raw model score → the true 0–100 scale.

### 4.4 Non-negotiables
- **Monotonicity:** higher predicted proficiency must never lower the score.
- **Calibrate probabilities first** (Platt/isotonic) or the score will be biased.

---

## 5. Interpretability (the "sections" requirement)

Because `score = Σ P(level) · anchor`, and EBM gives exact additive feature terms, each
prediction reads as:

> "Section 3 pushed the score **+12**, Section 7 **−5**, …" — contributions summing to the
> final 0–100.

- **EBM:** native additive shape functions per feature (and pairwise interactions).
- **SHAP:** same additive decomposition for the boosting / RF / SVM models — use as a
  cross-check and for the non-glass-box candidates.
- **Global view:** feature/section importances; **Local view:** per-sample attribution.
- Map each feature back to its group → report **section-level** contribution for stakeholders.

---

## 6. Validation protocol (making 82% credible)

At n≈220 a single CV split swings ±~5%, so a one-fold "we hit 82%" will not survive scrutiny.

- **Repeated stratified k-fold** (e.g., 5-fold × 10 repeats). Report **mean ± std** and a CI,
  not a point estimate.
- **Do the 1-of-k feature-group selection *inside* the CV folds.** Selecting features on the
  full data first leaks and inflates the number — the most common small-n mistake.
- **Ordinal metrics alongside accuracy:**
  - **Quadratic Weighted Kappa (QWK)** — penalizes far errors more than adjacent ones.
  - **MAE on band index** — average ordinal distance of errors.
  - **Spearman / Pearson** of predicted-vs-implied 0–100 — monotonic agreement.
  - **Macro-F1** — guards against a dominant band inflating accuracy.
- **Calibration check** (reliability curve) before trusting the 0–100 map.
- **Class imbalance:** class weights and threshold moving are safe; heavy resampling at n≈220
  tends to overfit.

### Feature-group selection caveat
Choosing 1-of-k across 8–11 groups is a large combinatorial search
(`k^(#groups)`). Selecting the combo that maximizes CV score **outside** nested CV overfits
badly. Options: (a) nested CV, (b) domain knowledge to fix choices, (c) group-aware selection.

---

## 7. Experiment plan — the bake-off

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
  → SHAP / EBM attribution → per-section contributions
```

**Report:** accuracy (3-band), QWK, MAE-band-index, macro-F1, calibration, and a worked
0–100 explanation example per model.

---

## 8. Pitfalls checklist

- [ ] Feature selection done **inside** CV (no leakage).
- [ ] Repeated CV with mean ± std, not a single lucky split.
- [ ] Probabilities calibrated before the 0–100 map.
- [ ] 0–100 map is monotonic; anchors are documented/citable.
- [ ] Ordinal metrics reported, not accuracy alone.
- [ ] Class imbalance handled with weights/thresholds, not heavy resampling.
- [ ] No deep ordinal nets (CORAL/CORN) at this n.
- [ ] Each feature traced back to its section for reporting.

---

## 9. Prompt for the ML research agent (v2)

```
You are an ML research assistant. Produce a rigorous, citation-backed comparison for an
applied problem. Prioritize SMALL-DATA robustness and INTERPRETABILITY. Be concrete about
algorithms, Python libraries, and tradeoffs. Survey and rank — do NOT write a full solution.

PROBLEM
- Predict CEFR proficiency from linguistic features, and emit a continuous 0–100 score
  (hard business requirement) with per-"section" interpretability.
- Labels: raw 6-level CEFR (A1–C2) AND a fixed 3-band collapse {A1,A2} < {B1} < {B2,C1,C2}.
  Accuracy is measured on the 3 bands; the 6-level labels are available as extra signal.
- n ≈ 220 total. Features come as 8–11 "feature groups"; exactly one feature is chosen per
  group (each group = alternative proxies for one construct) → 8–11 model inputs.
- Baseline: a senior reached 77% band accuracy with REGRESSION models. Mandate: use
  NON-regression methods; target ≥82% band accuracy WITHOUT losing interpretability.
- No continuous ground-truth score exists — the 0–100 must be DERIVED from class
  probabilities in a principled, monotonic way.

EVALUATE AND RANK THESE CANDIDATES (add others if warranted):
1. Explainable Boosting Machine (EBM/GA2M, InterpretML).
2. Ordinal gradient boosting: OGBoost, and LightGBM/XGBoost under Frank–Hall (2001) ordinal
   decomposition with shallow trees + monotonic constraints.
3. Ordinal Random Forest (ordinalForest / RF + Frank–Hall).
4. SVM (RBF) + ordinal decomposition + Platt calibration.
5. Proportional-odds ordinal logistic — as an interpretable BASELINE only.
Explicitly assess whether CORAL/CORN deep ordinal nets are viable at n≈220 (I expect not).

ANSWER THESE QUESTIONS, each with methods + pros/cons + small-n suitability + citations:
A. Which candidates most reliably clear ~82% 3-band accuracy at n≈220 without overfitting?
   Cite CEFR/proficiency-classification precedents (e.g., microsystem criterial features,
   SVM over grouped linguistic features, interpretable feature-selection studies).
B. Does training a 6-level ordinal model then collapsing to 3 bands beat direct 3-band
   training? Cite evidence both ways.
C. PROBABILITY → 0–100: expected-value over per-level anchors vs cumulative-probability
   integration vs isotonic recalibration. Recommend defensible CEFR numeric anchors
   (e.g., Pearson Global Scale of English) and how to rescale to 0–100. Emphasize
   monotonicity + calibration.
D. INTERPRETABILITY: faithful additive per-feature (per-section) attribution that sums to
   the 0–100 output — EBM native terms vs SHAP (which explainer per model) vs monotonic GBM.
E. VALIDATION at small n with combinatorial 1-of-k feature-group selection: nested/repeated
   stratified CV, leakage from selecting features outside CV, and ordinal metrics
   (Quadratic Weighted Kappa, MAE-on-band-index, Spearman of predicted-vs-true 0–100).
F. Probability CALIBRATION for small n (Platt vs isotonic vs beta) and class-imbalance
   handling that is safe at n≈220.

DELIVERABLE
- A ranked shortlist of 3–5 end-to-end pipelines (model → calibration → 0–100 map →
  interpretability), each with why it fits n≈220, expected interpretability, failure modes.
- Flag anything likely to overfit at this n. Cite papers and libraries throughout.
```

---

## 10. Sources

- [Predicting CEFR levels: microsystem criterial features (Cambridge, ReCALL)](https://www.cambridge.org/core/journals/recall/article/abs/predicting-cefr-levels-in-learners-of-english-the-use-of-microsystem-criterial-features-in-a-machine-learning-approach/C915A35CD69168EDFB80DE8F57A4328C) — 82% balanced accuracy A1–C2 precedent
- [Towards interpretable models for CEFR level prediction (Estonian, arXiv 2602.13102)](https://arxiv.org/abs/2602.13102)
- [OGBoost: Ordinal Gradient Boosting (arXiv 2502.13456)](https://arxiv.org/pdf/2502.13456)
- [Ordinal decision-tree ensembles beat non-ordinal counterparts (PMC7517475)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7517475/)
- [RF vs GBM on small categorical datasets (PMC8392226)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8392226/)
- [Explainable Boosting Machine / InterpretML docs](https://interpret.ml/docs/ebm.html)
- [CORAL/CORN rank-consistent ordinal networks (arXiv 2111.08851)](https://arxiv.org/abs/2111.08851)
- [Automated CEFR-J writing assessment: lexical metrics + AI (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2772766125000205)

---

## 11. Next step

Write the bake-off code: EBM + ordinal-GBM + ordinal-RF (+ SVM, POM baseline), Frank–Hall
decomposition, nested repeated stratified CV, expected-value 0–100 with GSE anchors, and
SHAP/EBM per-section attribution. **Needs:** the dataset (or a schema + band distribution +
example feature-group options).
