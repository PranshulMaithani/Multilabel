# CEFR: 10 Modeling Methods — Detailed & Ranked by Chance of Clearing 82%

> Deep-dive on 10 candidate methods for the cascade
> (**multiclass → confidences → 0–100 score → cut-points → band/level classification**),
> ranked by realistic probability of reaching **≥82% band accuracy** at **n≈220**, 8–11
> features, ordinal target, with the 0–100 output and per-section interpretability.
>
> **Ranking is a prior, not a verdict.** At n≈220 the top 4–5 sit within noise; the
> repeated nested-CV bake-off (see `CEFR_Project_Discussion.md` §10–11) is the arbiter.
> The single biggest lever is **feature signal quality**, not model choice — the published
> 82% precedent came from carefully engineered features.
>
> **Last updated:** 2026-07-20

---

## Ranking at a glance

| # | Method | Family | Chance to clear 82% | Interpretability | Overfit risk @ n≈220 |
|---|---|---|---|---|---|
| 1 | Ordinal Gradient Boosting (OGBoost / LGBM+Frank–Hall) | Boosted trees | **High** | ★★ (SHAP) | Medium (needs regularization) |
| 2 | Explainable Boosting Machine (EBM / GA2M) | Additive (GAM) | **High** | ★★★ native | Low–Medium |
| 3 | SVM (RBF) + ordinal decomposition + calibration | Kernel margin | **High** | ★ (SHAP) | Low–Medium |
| 4 | Ordinal Random Forest | Bagged trees | Medium–High | ★★ (SHAP/perm) | **Low** |
| 5 | Linear / Quadratic Discriminant Analysis (LDA/QDA) | Probabilistic linear | Medium–High | ★★ coefficients | **Low** (LDA) |
| 6 | Proportional-Odds Ordinal Logistic (POM) — *baseline* | Ordinal regression | Medium | ★★★ coefficients | Low |
| 7 | Gaussian Naive Bayes | Probabilistic | Medium | ★★ per-feature | **Low** |
| 8 | k-Nearest Neighbors (distance-weighted, ordinal) | Instance-based | Low–Medium | ★ case-based | Medium (scaling-sensitive) |
| 9 | Single Decision Tree (shallow CART) | Tree / rules | Low | ★★★ rules | High (unstable) |
| 10 | Composite Index / PCA scoring (rubric) | Unsupervised index | Low | ★★★ rubric | **Very low** |

Methods **1–5** are the realistic 82% contenders. **6–7** are strong, honest baselines.
**8–10** are simple/transparent tools — most valuable for interpretability, sanity checks, or
as the human-readable "official" 0–100 score even if a stronger model drives the classification.

**Legend for each method below:** *How it works · Applied here · 0–100 score · Interpretability
· Small-n behavior · Config & library · Why this rank.*

---

## 1. Ordinal Gradient Boosting — OGBoost / LightGBM + Frank–Hall  ·  Chance: **High**

**How it works.** Builds an additive ensemble of *shallow* decision trees sequentially; each
new tree fits the gradient of the loss left by the current ensemble. It captures non-linear
feature effects and interactions automatically. Ordinal behavior comes either from **OGBoost's
native ordinal loss** (2025 package, with cross-validated early stopping specifically for small
/ imbalanced data) or from **Frank–Hall decomposition**: train `K−1` binary boosters for
`P(y > level k)` and difference adjacent cumulative probabilities.

**Applied here.** Train on the **6 levels** via Frank–Hall (5 binary boosters) — this gives the
finest probability vector for the 0–100 map and the "free accuracy" collapse to 3 bands. Fight
overfitting at n≈220 aggressively: `max_depth = 2–3`, `learning_rate = 0.01–0.05`, large
`n_estimators` with **early stopping**, `min_child_samples ≈ 15–30`, `subsample ≈ 0.7`,
`colsample`/`feature_fraction ≈ 0.7`, and L1/L2 leaf regularization. Add **monotonic
constraints** on any feature known to move monotonically with proficiency — this both improves
generalization and guarantees a sensible score.

**0–100 score.** Frank–Hall → per-level probabilities → expected value over GSE anchors
(`score = Σ P(level)·anchor(level)`).

**Interpretability.** **SHAP TreeExplainer** gives exact additive per-feature contributions
that sum to the prediction; aggregate features back to their groups for **per-section**
attribution. Also: gain-based importance and partial-dependence plots.

**Small-n behavior.** Highest accuracy *ceiling* of all 10, but also the highest overfit risk —
so the regularization above and OGBoost's CV early stopping matter. Always evaluate with
repeated nested CV, never a single split.

**Config & library.** `ogboost`; or `lightgbm` / `xgboost` (binary objective per Frank–Hall
booster); `shap` for attribution.

**Why #1.** With disciplined regularization it has the best odds of *exceeding* 82% when the
features carry signal. It's ranked above EBM only for raw accuracy headroom — EBM wins on
interpretability (see #2).

---

## 2. Explainable Boosting Machine — EBM / GA2M  ·  Chance: **High**

**How it works.** A **glass-box generalized additive model**: `g(E[y]) = β₀ + Σ f_j(x_j) +
Σ f_ij(x_i, x_j)`. Each `f_j` is a **shape function** for one feature, learned by cyclic
gradient boosting on shallow trees in round-robin over features (so no single feature dominates
the boosting). Optional pairwise terms `f_ij` = **GA2M**. Accuracy is typically within a point
or two of full gradient boosting / random forest.

**Applied here.** Use a multiclass EBM on the 6 levels, or Frank–Hall binary EBMs for a cleaner
ordinal decomposition. Keep interactions few (e.g., top 5–10) at n≈220 to avoid overfitting.
Because it's additive, the score's per-section decomposition is **exact by construction**.

**0–100 score.** Class probabilities (reasonably calibrated) → expected value over GSE anchors.

**Interpretability.** **Best of the 10.** Every feature has a plottable shape function (global
view), and each prediction decomposes into per-feature term values that **sum exactly** to the
output logit/score (local view) — precisely the "Section 3 = +12, Section 7 = −5" story, with
no post-hoc approximation like SHAP.

**Small-n behavior.** More stable than full GBM (additive, shallow, round-robin); limit
interactions and use `outer_bags` for variance reduction.

**Config & library.** `interpret` → `ExplainableBoostingClassifier`; tune `max_bins`,
`interactions`, `learning_rate`, `outer_bags`, `min_samples_leaf`.

**Why #2.** Essentially ties #1 on accuracy odds while being the most interpretable model
here. **For the combined "82% *and* interpretability" goal, this is arguably co-#1** — it leads
#1 only if raw accuracy is not the sole criterion.

---

## 3. SVM (RBF) + Ordinal Decomposition + Calibration  ·  Chance: **High**

**How it works.** Finds the maximum-margin decision boundary in a high-dimensional feature
space induced by the RBF kernel; regularization parameter `C` trades margin width against
errors. Probabilities come from **Platt scaling** (a sigmoid fit on the decision function).

**Applied here.** **Standardize features first (critical for RBF).** Wrap in Frank–Hall
(binary `y > k` SVMs) for ordinal structure, or use one-vs-one multiclass. Tune `C` and `gamma`
by nested CV; set `class_weight='balanced'`. This is the **historical CEFR workhorse** —
grouped linguistic features + SVM is a well-trodden, strong path.

**0–100 score.** Calibrated per-level probabilities → expected value over GSE anchors.

**Interpretability.** Not native. Recover per-section attribution with **SHAP KernelExplainer**
(slow but model-agnostic) or **permutation importance**. Weaker/coarser than EBM.

**Small-n behavior.** Excellent — SVMs are among the strongest classifiers on small tabular
data and resist overfitting with a well-chosen `C`. Sensitive to feature scaling and to
irrelevant features (another reason the 1-of-k group selection matters).

**Config & library.** `sklearn` `SVC(probability=True)` inside a `Pipeline` with
`StandardScaler`, wrapped in `CalibratedClassifierCV`.

**Why #3.** Accuracy odds comparable to #1–#2 and a proven CEFR track record; ranked just
below them because interpretability is post-hoc and calibration adds moving parts.

---

## 4. Ordinal Random Forest  ·  Chance: **Medium–High**

**How it works.** Bagging: many *de-correlated* deep trees trained on bootstrap samples with
random feature subsets; predictions are averaged/voted. Ordinal versions (R's `ordinalForest`)
optimize a score-based partition of the ordinal target; alternatively use RF inside Frank–Hall.

**Applied here.** 500–1000 trees, `max_features ≈ sqrt(p)`, `min_samples_leaf` moderate
(5–10) for smoothing. Handles non-linearity with **low variance** — often the most *stable*
strong model at n≈220.

**0–100 score.** Vote-proportion probabilities (calibrate — RF probabilities are often
under-confident) → expected value over GSE anchors.

**Interpretability.** **Permutation importance** (prefer over impurity importance, which is
biased toward high-cardinality features) and **SHAP TreeExplainer** for per-section attribution.

**Small-n behavior.** Very robust to overfitting thanks to bagging; the main cost is a slightly
lower ceiling than boosting because it can under-fit sharp class boundaries. On small,
categorical-ish data, bagging is frequently *more accurate and stable* than boosting — do not
assume boosting wins.

**Config & library.** `sklearn` `RandomForestClassifier` (+ Frank–Hall wrapper) or R
`ordinalForest` / `ranger`.

**Why #4.** A reliable near-top performer; ranked below the top 3 mainly on ceiling, but it may
well win the bake-off by being the least variance-prone.

---

## 5. Linear / Quadratic Discriminant Analysis — LDA / QDA  ·  Chance: **Medium–High**

**How it works.** Models each class as a multivariate Gaussian and applies Bayes' rule. **LDA**
assumes a *shared* covariance → linear boundaries and very few parameters. **QDA** allows a
*per-class* covariance → quadratic boundaries but needs more samples per class.

**Applied here.** Standardize features; treat as multiclass on the 6 levels, then collapse.
With only 8–11 features that are roughly Gaussian construct-proxies, **LDA is a genuinely
competitive small-n method that often beats fancier models.** Use **shrinkage LDA**
(Ledoit-Wolf) for extra stability. QDA is borderline here (≈37 samples/level for 6 levels) —
try it, but expect LDA to be safer.

**0–100 score.** Posterior probabilities are smooth and well-behaved → expected value over GSE
anchors.

**Interpretability.** LDA's **linear discriminant coefficients** are per-feature weights along
the discriminant direction — readable and mappable to sections; less granular than EBM/SHAP but
transparent.

**Small-n behavior.** LDA has **low variance** and is a classic strong baseline when n is
small; QDA is riskier due to per-class covariance estimation.

**Config & library.** `sklearn` `LinearDiscriminantAnalysis(solver='lsqr',
shrinkage='auto')` / `QuadraticDiscriminantAnalysis(reg_param=...)`.

**Why #5.** The strongest of the "simple" methods and a realistic ~80%+ contender that can
occasionally clear 82% — an important, cheap benchmark that the complex models must beat to
justify themselves.

---

## 6. Proportional-Odds Ordinal Logistic (POM) — *baseline*  ·  Chance: **Medium**

**How it works.** A single latent linear predictor `η = wᵀx` plus `K−1` ordered thresholds;
cumulative logits `P(y ≤ k) = σ(θ_k − η)`. The "proportional odds" assumption: one shared `w`
across all thresholds.

**Applied here.** The natural ordinal reference model. Standardize features. Report it as the
line every non-regression method must beat.

**0–100 score.** Cumulative probabilities natively, or per-class differences → expected value.

**Interpretability.** **Fully transparent** — each coefficient is a log-odds effect per feature.

**Small-n behavior.** Stable and interpretable, but limited by its linearity and the
proportional-odds assumption → moderate ceiling.

**Caveat.** This **is "ordinal regression"** — technically what the senior wants to move past.
Keep it strictly as a benchmark, not the deliverable.

**Config & library.** `mord` (`LogisticAT`), `statsmodels` `OrderedModel`.

**Why #6.** A solid, honest baseline with a moderate ceiling — the reference line for the whole
project, not a likely winner.

---

## 7. Gaussian Naive Bayes  ·  Chance: **Medium**

**How it works.** Assumes features are conditionally independent given the class; multiplies
per-feature Gaussian likelihoods by class priors. Extremely few parameters.

**Applied here.** Multiclass on the 6 levels. Its robustness at tiny n is a real asset, but the
**independence assumption is violated** here — features within/across groups are correlated
construct proxies — which caps accuracy.

**0–100 score.** Posteriors → expected value, **but calibrate first** (NB probabilities are
often over-confident / poorly calibrated).

**Interpretability.** Nice additive story in **log space**: each feature contributes a
log-likelihood-ratio term to each class — a natural per-section decomposition.

**Small-n behavior.** Very low variance, fast, hard to overfit.

**Config & library.** `sklearn` `GaussianNB` + `CalibratedClassifierCV`.

**Why #7.** Robust and interpretable, but correlated features keep its ceiling below LDA — a
good fast baseline rather than a contender.

---

## 8. k-Nearest Neighbors (distance-weighted, ordinal-aware)  ·  Chance: **Low–Medium**

**How it works.** Classifies a point by its `k` nearest training exemplars, weighted by
distance. Naturally ordinal if you **average the neighbors' band index** instead of majority
voting.

**Applied here.** **Standardize features (critical).** Small `k` (5–15), distance weighting.
A neat trick: use **`KNeighborsRegressor` on the band index** → it outputs a smooth continuous
value that *is* your 0–100 score after scaling, no anchor mapping needed. Consider feature
weighting/selection since kNN is hurt by irrelevant features.

**0–100 score.** Distance-weighted mean neighbor band index → directly a 0–100 value; or
class-vote probabilities → expected value.

**Interpretability.** **Case-based**: "this learner scored like these most-similar exemplars."
Intuitive for stakeholders but *not* an additive per-section decomposition.

**Small-n behavior.** Workable at 8–11 dimensions (curse of dimensionality is mild), but very
sensitive to scaling, metric choice, and irrelevant/redundant features.

**Config & library.** `sklearn` `KNeighborsClassifier` / `KNeighborsRegressor`.

**Why #8.** Moderate ceiling and fragile to preprocessing; valuable mainly as an intuitive,
case-based complement and a diversity member for an ensemble.

---

## 9. Single Decision Tree (shallow CART)  ·  Chance: **Low**

**How it works.** Recursively splits the feature space to maximize class purity; each leaf
predicts a class distribution.

**Applied here.** Keep it shallow (`max_depth = 3–4`) with cost-complexity pruning to control
overfitting. Not ordinal-native, but fine as a **rule-extraction** tool.

**0–100 score.** Leaf class proportions → expected value, but with few leaves the score takes a
**small number of discrete values** (chunky, not smooth).

**Interpretability.** **Best-in-class for human-readable rules** — literal if-then paths a
non-technical stakeholder can follow. But no additive per-section contribution.

**Small-n behavior.** **High variance / unstable** — a single tree overfits and its structure
swings with small data changes (the very reason forests and boosting exist).

**Config & library.** `sklearn` `DecisionTreeClassifier(ccp_alpha=...)`.

**Why #9.** Low accuracy ceiling; best used as an *explanation aid* on top of a stronger model,
not as the classifier itself.

---

## 10. Composite Index / PCA-based Scoring (rubric)  ·  Chance: **Low**

**How it works.** Normalize the selected features (e.g., min-max to 0–1), combine them into a
single index with weights, and scale to 0–100. To stay **non-regression**, derive weights from
**domain knowledge**, **feature-label correlation**, or **PCA (first principal component)** —
*not* least-squares fitting. Threshold the index into bands.

**Applied here.** The simplest possible realization of the 0–100 requirement, monotone by
construction. Effectively a transparent scoring **rubric**.

**0–100 score.** The index **is** the 0–100 score directly — no probability step needed.

**Interpretability.** **Maximal and business-friendly** — each section's weighted contribution
to the final score is explicit and fixed, exactly like a grading rubric.

**Small-n behavior.** Extremely robust (few or zero *fitted* parameters), essentially cannot
overfit — but its linear, unsupervised nature gives the **lowest accuracy ceiling**.

**Config & library.** `sklearn` `PCA`, `MinMaxScaler` / `StandardScaler`; numpy.

**Why #10.** Unlikely to clear 82% on its own, but the **most transparent** option and
genuinely useful: it can serve as the human-readable "official" 0–100 score and a sanity anchor,
even while a stronger model (#1–#5) does the actual classification.

---

## How to actually maximize the odds of 82%

1. **Fix the features first.** Model choice moves accuracy a few points; feature quality moves
   it a lot. Confirm what each group measures and get the 1-of-k selection right (inside CV).
2. **Run the bake-off** (`CEFR_Project_Discussion.md` §10) across #1–#5 + the #6 baseline with
   repeated nested CV; report mean ± std, QWK, and balanced accuracy.
3. **Ensemble the top 2–3 diverse winners** (e.g., EBM + SVM + Ordinal RF) via soft-voting on
   the per-level probabilities — diversity typically adds **1–3 points** and stabilizes the
   0–100 score.
4. **Tune the cut-points** on the 0–100 score inside each fold, and always compare against the
   plain `argmax` baseline (the crux test, §11) so any lift is real.
5. **Keep an interpretable spine** — EBM or the composite index — so the final deliverable
   explains itself per section regardless of which model wins on accuracy.

---

## Sources

- [OGBoost: Ordinal Gradient Boosting (arXiv 2502.13456)](https://arxiv.org/pdf/2502.13456)
- [Explainable Boosting Machine / InterpretML docs](https://interpret.ml/docs/ebm.html)
- [Ordinal decision-tree ensembles beat non-ordinal counterparts (PMC7517475)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7517475/)
- [RF vs GBM on small categorical datasets (PMC8392226)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8392226/)
- [Predicting CEFR levels: microsystem criterial features (Cambridge, ReCALL)](https://www.cambridge.org/core/journals/recall/article/abs/predicting-cefr-levels-in-learners-of-english-the-use-of-microsystem-criterial-features-in-a-machine-learning-approach/C915A35CD69168EDFB80DE8F57A4328C)
- [Towards interpretable models for CEFR level prediction (arXiv 2602.13102)](https://arxiv.org/abs/2602.13102)
- [CORAL/CORN rank-consistent ordinal networks (arXiv 2111.08851)](https://arxiv.org/abs/2111.08851)
- Frank, E. & Hall, M. (2001). *A Simple Approach to Ordinal Classification.* ECML.
- Pearson **Global Scale of English (GSE)** — CEFR ↔ numeric anchor mapping.
