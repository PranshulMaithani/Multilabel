# The Two Methods, Explained in Full

> Study guide for `notebooks/cefr_2_methods.ipynb`.
> Covers **what** each approach is, the **intuition**, **what we actually did**, and exactly
> **how the two model scores become a 0-100 number and get split into the three bands.**
>
> Last updated: 2026-07-22

> **Note on the baseline notebook.** `cefr_2_methods.ipynb` currently runs a **simplified
> baseline**: no hyperparameter tuning, cut-points fitted on the **train** scores, and it
> reports **train / test / full** accuracy only (no cross-validation number). Sections 6, 8
> and 9 below describe the more rigorous **out-of-fold + tuning** approach — the recommended
> upgrade once you are past establishing a baseline. `test_acc` is an honest held-out number
> in both versions.

---

## Table of contents

1. [The problem and the pipeline](#1-the-problem-and-the-pipeline)
2. [Shared machinery: Frank-Hall ordinal decomposition](#2-shared-machinery-frank-hall-ordinal-decomposition)
3. [Method 1 - Ordinal Random Forest](#3-method-1--ordinal-random-forest)
4. [Method 2 - Ordinal Boosting (LightGBM)](#4-method-2--ordinal-boosting-lightgbm)
5. [From two scores to a 0-100 number](#5-from-two-scores-to-a-0-100-number)
6. [The two split points, in full](#6-the-two-split-points-in-full)
7. [Feature importance: two views](#7-feature-importance-two-views)
8. [Hyperparameter tuning](#8-hyperparameter-tuning)
9. [Validation and the four accuracy columns](#9-validation-and-the-four-accuracy-columns)
10. [Cheat sheet: the two methods compared](#10-cheat-sheet-the-two-methods-compared)
11. [What to say to the senior](#11-what-to-say-to-the-senior)

---

## 1. The problem and the pipeline

### The task
From 8-11 features (one chosen per feature group, each group representing an assessment
**section**), predict which of three **ordered** CEFR bands a learner belongs to:

| Band | CEFR levels |
|---|---|
| 0 | A1, A2 |
| 1 | B1 |
| 2 | B2, C1, C2 |

- ~220 rows total, split into train/test by the `split` column.
- Baseline to beat: **77%** (senior's regression models). Target: **>=82%**.

### The business constraint that shapes everything
The deliverable is **not** a class label. It is a **0-100 score**, which is then cut into the
three bands by **two split points**. So both methods, however they differ internally, are
funnelled into the same shape:

```
   features
      |
      v
 [1] MODEL  ------------------->  probabilities  P(band0), P(band1), P(band2)
      |
      v
 [2] COLLAPSE to one number  --->  score in [0, 100]
      |
      v
 [3] TWO CUT-POINTS  ----------->  band 0 / 1 / 2
      |
      v
 [4] EXPLAIN  ------------------>  per-section contribution
```

Stage 1 is the only part that differs between the two methods. Stages 2-4 are **identical**,
which is what makes the comparison fair.

### Why the target is treated as ordinal
The bands have an order: `A1-A2 < B1 < B2-C1-C2`. A plain 3-class classifier treats them as
three unrelated labels, so it considers mistaking A1-A2 for B2-C1-C2 exactly as bad as
mistaking it for B1. That is wrong - one is a far worse error. Ordinal handling keeps that
information, and it makes the probabilities line up naturally on a one-dimensional 0-100 scale.

---

## 2. Shared machinery: Frank-Hall ordinal decomposition

Both methods use this. **It is the trick that turns an ordinal problem into ordinary binary
classification**, and it is the reason each method produces *two* scores.

### The intuition
Instead of asking *"which of three bands is this learner in?"*, ask a series of simpler
**cumulative** yes/no questions:

1. "Is this learner **above band 0**?" (better than A1-A2?)
2. "Is this learner **above band 1**?" (better than B1?)

This is how a human grader thinks - work up the scale asking "have they cleared this bar?"
And the answers must be **consistent**: anyone above band 1 is necessarily above band 0.

### The mechanics
For `K` ordered classes you train `K-1` binary models. With `K = 3`:

| Model | Question | Trained with positive label = |
|---|---|---|
| model 1 | `y > 0` | rows whose band is B1 or B2-C1-C2 |
| model 2 | `y > 1` | rows whose band is B2-C1-C2 |

Write their outputs as `c0 = P(y > 0)` and `c1 = P(y > 1)`. **These are the two scores** that
section 5 turns into the 0-100 number. Recover the three band probabilities by
**differencing**:

```
P(band 0) = 1  - c0        "not above band 0"
P(band 1) = c0 - c1        "above band 0 but not above band 1"
P(band 2) = c1             "above band 1"
```

These always sum to 1. Worked example with `c0 = 0.92`, `c1 = 0.31`:

```
P(band 0) = 1    - 0.92 = 0.08
P(band 1) = 0.92 - 0.31 = 0.61
P(band 2) =        0.31 = 0.31      (sums to 1.00)
```

### The safeguard (why the code has extra lines)
The two models are trained **independently**, so nothing forces `c0 >= c1` even though it must
hold logically. If a noisy fit returned `c0 = 0.40`, `c1 = 0.55`, then
`P(band 1) = 0.40 - 0.55 = -0.15` - a **negative probability**.

The code prevents this with `np.minimum.accumulate`, which forces the cumulative sequence to
be non-increasing (clamping `c1` down to `0.40`, so `P(band 1) = 0`). It then clips to a tiny
positive floor and renormalises so the three sum to exactly 1. That is the whole reason
`FrankHallOrdinal.predict_proba` is longer than the three-line formula.

---

## 3. Method 1 - Ordinal Random Forest

**Current best performer.**

### The intuition
A single decision tree is a flowchart of yes/no questions on features, ending in a verdict.
Trees are flexible but notoriously **unstable**: change a handful of training rows and you can
get a completely different tree. That instability is *variance*, and at 220 rows it is the
dominant source of error.

Random Forest's insight: **build hundreds of deliberately different trees and average them.**
Each tree is individually overfitted and unreliable, but their errors point in different
directions, so averaging cancels most of the noise while the real signal - which all trees
agree on - survives.

### How the differences are manufactured
Two sources of randomness, both essential:

1. **Bootstrap resampling (bagging).** Each tree is trained on a random sample of the rows
   drawn *with replacement*, so each sees a slightly different dataset.
2. **Random feature subsets.** At *every split*, a tree may only consider a random subset of
   features (`max_features`, e.g. `sqrt(9) = 3`).

Point 2 matters more than it looks. Without it, if one feature is strongly predictive every
tree would split on it first and all the trees would look nearly identical - averaging
near-identical things gains nothing. Forcing trees to sometimes ignore the best feature
**de-correlates** them.

### The maths of why it works
Averaging `B` estimators each with variance `s^2` and pairwise correlation `rho`:

```
Var(average) = rho * s^2  +  (1 - rho)/B * s^2
```

The second term vanishes as `B` grows - the free win from adding trees. But the first term
does **not** depend on `B`: it is floored by how correlated the trees are. This is exactly why
random feature subsets matter - they lower `rho` and thus lower the floor. It also explains
why **adding more trees never overfits** a forest: more trees only shrink the second term.

### What we actually did
- Frank-Hall wrapper -> **two independent forests**, one per cumulative question.
- Each forest: 400-800 trees, `max_features` in `{sqrt, 0.5, 0.8}`,
  `min_samples_leaf` in `{1,2,3,5,8}`, `max_depth` in `{None, 6, 10}`.
- `class_weight="balanced_subsample"` so a larger band cannot dominate.
- **No feature scaling** - trees split on thresholds, so any monotonic rescaling of a feature
  produces the identical tree. Only median imputation is applied.
- Probability = the fraction of trees voting "yes".

### Why it suits this problem
- Bagging attacks **variance**, the main failure mode at n~220.
- Your feature groups are **correlated proxies** of the same constructs. Correlated inputs
  destabilise linear models; a forest barely notices.
- It captures non-linear effects and interactions automatically, with almost no tuning.

### Reading its feature importance
Native importance = total **impurity reduction** contributed by each feature across all
splits, reported **per cumulative question**. The two columns often differ, and that
difference is informative: it tells you which sections discriminate at the **low end**
(`P(y>0)`, separating A1-A2 from the rest) versus the **high end** (`P(y>1)`, separating
B2-C1-C2). A section can matter enormously for one boundary and not at all for the other.

*Caveat:* impurity importance is biased toward high-cardinality / continuous features, which
is exactly why the notebook also reports **permutation importance** (section 7).

---

## 4. Method 2 - Ordinal Boosting (LightGBM)

### The intuition
Boosting takes the opposite approach to bagging. Rather than building many independent trees
and averaging, it builds trees **one at a time, each one fixing what the previous ones got
wrong**. It is deliberate, sequential self-correction: fit a weak model, look at the errors,
fit the next model specifically to those errors, repeat hundreds of times.

Each individual tree is intentionally feeble (depth 2-4, a "stump-like" weak learner). The
power comes from combining hundreds of small corrections, each applied at a low learning rate.

### The mechanics
Start from a constant, then repeatedly add a small correction:

```
F_0(x)  = constant
F_m(x)  = F_{m-1}(x)  +  nu * h_m(x)
```

where `h_m` is a shallow tree fitted to the **negative gradient** of the loss at the current
predictions (for squared error this is literally the residual; for log-loss it is the
probability error), and `nu` is the **learning rate**, typically 0.01-0.05. Small `nu` means
each tree nudges the prediction only slightly, so you need many trees - but the result
generalises far better than a few large steps.

### Bagging vs boosting - the contrast worth understanding
| | Random Forest (bagging) | Gradient boosting |
|---|---|---|
| Trees built | in **parallel**, independently | **sequentially**, each depends on the last |
| Each tree is | deep, low-bias, high-variance | shallow, high-bias, low-variance |
| Combining reduces | **variance** | **bias** |
| More trees | never overfits | **can** overfit |
| Key knobs | `max_features`, `min_samples_leaf` | `learning_rate`, `max_depth`, `n_estimators` |

**This is why both are in the notebook.** They have opposite failure modes, so when they agree
you can trust the result, and an ensemble of the two is genuinely diverse rather than two
flavours of the same idea.

Because boosting *can* overfit, at n~220 the regularisation knobs matter far more than they
would on a large dataset - hence the deliberately conservative search space below.

### What we actually did
- Same Frank-Hall wrapper -> **two independent LightGBM models**.
- Searched: `n_estimators` {200,400,600}, `learning_rate` {0.01,0.03,0.05},
  `num_leaves` {3,7,15}, `max_depth` {2,3,4}, `min_child_samples` {5,10,15,20},
  `reg_lambda` {0,1,5}.
- `subsample=0.8` + `colsample_bytree=0.8` add a bagging-flavoured randomness on top.
- `min_child_samples` is the single most important guard here: it forbids leaves built from a
  handful of rows, which is how boosting memorises noise on small data.

### Reading its feature importance
LightGBM **gain** importance = the total improvement in the loss delivered by every split on
that feature. This is more meaningful than a raw split *count*, because a feature might be
split on frequently while barely helping. Gain weights each split by how much it actually
bought. Reported per cumulative question, same as Method 1.

---

## 5. From two scores to a 0-100 number

This is the heart of the pipeline. Each method emits **two** numbers - `c0 = P(y>0)` and
`c1 = P(y>1)` - and we collapse them into **one** 0-100 score.

### Step 1 - two scores to three band probabilities
By the Frank-Hall differencing from section 2:

```
P(A1-A2)    = 1  - c0
P(B1)       = c0 - c1
P(B2-C1-C2) =      c1
```

### Step 2 - three probabilities to one score
Take an **expected value** using an anchor value per band:

```
score = P(A1-A2) x 0  +  P(B1) x 50  +  P(B2-C1-C2) x 100
```

### Step 3 - the algebra collapses to a one-liner
Substitute the differences from step 1:

```
score = (1 - c0) x 0  +  (c0 - c1) x 50  +  c1 x 100
      = 50*c0 - 50*c1 + 100*c1
      = 50*c0 + 50*c1
```

So the whole thing is just:

```
score = 50 x ( P(y>0) + P(y>1) )
```

### Why the score is always in [0, 100]
Two independent reasons, both worth understanding:

1. **Direct:** `c0` and `c1` are each probabilities in [0, 1], so `c0 + c1` is in [0, 2], and
   `x 50` lands in [0, 100].
2. **Structural:** the score is a **weighted average** of the anchors 0, 50, 100 - the weights
   are the three band probabilities, which are non-negative and sum to 1. A weighted average
   can never fall outside the range of the values being averaged, so it is trapped in
   [min anchor, max anchor] = [0, 100]. This is why the property survives *any* anchor choice.

### The intuition to remember
**Each of the two proficiency bars a learner clears is worth 50 points, weighted by how
confident the model is that they cleared it.** Cleared both with certainty -> 100; neither ->
0; a coin-flip on the first bar only -> ~25.

### Worked example
`c0 = 0.90`, `c1 = 0.30`:
```
direct : 50 * (0.90 + 0.30)          = 60.0
bands  : P = (0.10, 0.60, 0.30)
         0*0.10 + 50*0.60 + 100*0.30 = 60.0    (agree)
```

### Why an expected value rather than `argmax`
`argmax` discards confidence. Two learners both labelled "B1" are identical to `argmax`, even
though one may sit a hair below B2-C1-C2 and the other a hair above A1-A2:

| learner | P(A1-A2) | P(B1) | P(B2-C1-C2) | argmax | score |
|---|---|---|---|---|---|
| X | 0.45 | 0.50 | 0.05 | B1 | 30.0 |
| Y | 0.05 | 0.50 | 0.45 | B1 | 70.0 |

Same label, 40 points apart. Distinguishing them is *the entire point* of a 0-100 business
score - and it is what allows tuned cut-points to recover accuracy `argmax` leaves on the
table (the `lift_vs_argmax` column measures exactly this).

### Properties of the score
- **Bounded** in [0, 100] - shown above.
- **Monotone** - moving probability mass toward a higher band can only raise it.
- **Smooth** - small changes in model confidence produce small changes in the score. No jumps,
  which matters if a learner is ever re-scored.

### Anchors: spacing matters, absolute values do not
This is subtle and easy to get wrong.

**Equally spaced anchors are interchangeable.** `[0,50,100]`, `[0,1,2]` and `[10,55,100]` are
affine transformations of each other, so they rank learners **identically**. After the
cut-points are re-tuned, accuracy is **exactly** the same and only the printed number changes.
Restyling an equally spaced scale for the business is free.

**Unequally spaced anchors are NOT free.** The GSE-flavoured set `[29, 50, 74]` has gaps of
**21 and 24**, so it weights the top band relatively more, and it **can reorder learners**:

| learner | P(A1-A2) | P(B1) | P(B2-C1-C2) | score `[0,50,100]` | score `[29,50,74]` |
|---|---|---|---|---|---|
| A | 0.601 | 0.00 | 0.399 | **39.90** | **46.96** |
| B | 0.200 | 0.80 | 0.000 | **40.00** | **45.80** |

Under equal spacing **A < B**; under GSE **A > B**. The ranking flips, so band assignments -
and therefore accuracy - can differ. Adopting CEFR/GSE-flavoured anchors is a genuine
modelling decision: **re-run and compare, don't assume it is cosmetic.**

---

## 6. The two split points, in full

### Why not just use 33.3 and 66.7?
Because the score distribution is not uniform, and model confidence is not calibrated to the
band prior. Scores **clump** - a model might push most learners into the 55-75 range, so a
fixed threshold at 66.7 would slice straight through the densest region and misclassify
heavily. The thresholds have to sit where the model **actually separates** the bands, and that
is an empirical question, not a matter of arithmetic.

### The search algorithm
1. **Build candidates.** Take 120 evenly spaced **percentiles** of the score distribution and
   deduplicate. Percentiles rather than a flat 0-100 grid because they concentrate resolution
   where the data actually lies - no candidates wasted on empty stretches of the scale.
2. **Enumerate every ordered pair** `(t1, t2)` with `t1 < t2` - roughly 7,000 combinations.
3. **Score each pair** by the accuracy of the resulting assignment:
   `band = 0 if s <= t1, else 1 if s <= t2, else 2`.
4. **Keep the best pair.** The search is exhaustive over that grid, so it finds the grid's
   global optimum - there is no local-minimum risk and no optimiser to babysit.

`OPTIMIZE_METRIC` controls step 3. Use `"accuracy"` when the bands are balanced; switch to
`"balanced_accuracy"` if they are not, otherwise the thresholds drift to favour whichever band
is largest.

### The critical detail: fitted on OUT-OF-FOLD scores
The cut-points are chosen from **5-fold out-of-fold** predictions on the training set, never
from in-sample ones. This is not a technicality - it is the difference between a working model
and a broken one.

An in-sample score is what the model outputs for a row it was **trained on**, so it is
unrealistically confident. Thresholds tuned against inflated scores sit in the wrong place for
genuinely unseen data.

**This is not hypothetical - it happened in an earlier version of this project.** kNN with
distance weighting makes every training point its own zero-distance neighbour, so its
in-sample probabilities were a perfect 1.0 on the true class. Cut-points fitted to that scored
**0.500** on test, while the model's own `argmax` scored **0.868**. Switching to out-of-fold
scores fixed it to **0.838**.

Random Forest has the same disease in milder form: `train_acc ~0.99` against `oof_acc ~0.82`
is exactly that gap. Anything fitted on top of a model's own predictions - cut-points,
calibration, stacking - must use out-of-fold predictions.

### No leakage into the test set
The chosen `(t1, t2)` are **frozen** and applied to the test scores unchanged. The test set
plays no part in selecting them. In production these are simply two constants.

### Check it is a plateau, not a spike
A single best pair means little if accuracy collapses the moment a threshold moves by a point.
The notebook reports the top pairs and measures how wide the near-optimal region is:

- **Broad plateau** -> the thresholds are stable and will transfer to new data.
- **Narrow spike** -> they are fitted to noise; treat them, and the accuracy they produce,
  with suspicion.

This is also the honest answer when someone asks *"why 30.8 and 74.8, not round numbers?"* -
you can show the range over which they could move without hurting.

---

## 7. Feature importance: two views

Each method reports **both**, then rolls them up to sections via `FEATURE_GROUPS`.

### View 1 - native importance (model-specific)
Whatever the model itself exposes:

| Method | Native importance | Units |
|---|---|---|
| Ordinal RF | impurity reduction across splits | relative, per cumulative question |
| Ordinal Boosting | **gain** - loss improvement per split | relative, per cumulative question |

Cheap, and it reveals internal structure - especially the per-question split (low-end vs
high-end discrimination). But the units differ per model, so native importances are **not
directly comparable across the two methods**.

### View 2 - permutation importance on the final band prediction
Shuffle one feature, re-run the **entire pipeline** (model -> 0-100 score -> cut-points), and
measure how much band accuracy drops.

This is the one to quote, for three reasons:
1. It is **model-agnostic**, so both methods are on the same scale and directly comparable.
2. It measures the **actual deliverable** (the final band), not an internal quantity.
3. It has an unambiguous interpretation: *"shuffling this section costs us N accuracy points."*

A feature with importance <= 0 contributes nothing **in that model** and is a drop candidate.
The cross-method comparison table adds an `agreement` column - how many of the two methods
found each feature useful. Useful in **both** = core signal; useful in **neither** = a genuine
drop candidate.

---

## 8. Hyperparameter tuning

Each method gets a **randomised search inside 5-fold cross-validation**
(`RandomizedSearchCV`, `TUNE_ITER` candidates, scored by `OPTIMIZE_METRIC`). Randomised rather
than exhaustive grid search because at these grid sizes random sampling finds near-optimal
settings far more cheaply - most hyperparameters do not matter much, and random search spends
its budget exploring the ones that do.

**Honest caveat.** The hyperparameters are selected using the same CV folds later used to fit
the cut-points, so `oof_acc` is mildly optimistic. A fully rigorous setup would use **nested**
cross-validation (an inner loop for tuning, an outer for evaluation). At n~220 that is
expensive and noisy, so this is a deliberate trade-off - just don't present `oof_acc` as if it
were untouched by selection. `test_acc` remains clean, because the test set is used nowhere in
tuning.

Set `TUNE = False` to fall back to the hand-picked defaults.

---

## 9. Validation and the four accuracy columns

| Column | What it is | Honest generalisation estimate? |
|---|---|---|
| `train_acc` | fitted on train, predicted on train | **No** - optimistic by construction |
| `test_acc` | held-out test set | **Yes** |
| `full_acc` | train + test combined | **No** - ~2/3 of it is the in-sample part |
| `oof_acc` | out-of-fold predictions on train | **Yes** - the most stable |

**How to use them.**
- `train_acc` and `full_acc` are **fit diagnostics, not performance claims.** Never quote
  `full_acc` to the business as "our accuracy" - it is inflated by in-sample rows. A method
  that memorises its training data can show a spectacular `full_acc` and still be poor.
- The gap `train_acc - oof_acc` is your **overfitting meter**. Expect it to be large for both
  tree ensembles (they are high-capacity), which is exactly why the cut-points are fitted on
  out-of-fold scores.
- **Choose the method on `oof_acc`; report `test_acc`.** Picking the winner by highest
  `test_acc` is itself overfitting to the test set - with ~65 test rows, a 3-4 point gap is
  well inside noise.

With a test set this small, a difference of one or two learners moves accuracy by ~1.5 points.
Treat close finishes as ties and prefer the simpler, more stable model.

---

## 10. Cheat sheet: the two methods compared

| | Ordinal RF | Ordinal Boosting |
|---|---|---|
| **Family** | bagged trees | boosted trees |
| **Ordinal via** | Frank-Hall (2 forests) | Frank-Hall (2 LightGBM models) |
| **Trees built** | parallel, independent | sequential, corrective |
| **Reduces** | variance | bias |
| **More trees** | never overfits | can overfit |
| **Needs scaling** | no | no |
| **Overfit risk @ n~220** | low-medium | **medium-high** |
| **Native importance** | impurity reduction | gain |
| **Best trait** | robust, stable all-rounder | highest accuracy ceiling |
| **Watch out for** | impurity-importance bias | needs careful regularisation |

**Ensemble.** The notebook soft-votes the two by averaging their probability vectors. Because
one reduces variance and the other reduces bias, averaging can help - though on a small test
set verify it actually does rather than assuming.

**Which to pick?** If they land within ~2 points on `oof_acc` (likely), prefer the **forest**:
it is more stable, harder to overfit, and needs less tuning to defend. Use boosting if it wins
`oof_acc` by a clear margin.

---

## 11. What to say to the senior

> Each learner's section features go into a model that estimates two probabilities: that they
> are above the A1-A2 boundary, and that they are above the B1 boundary. Those two numbers are
> added and multiplied by 50 to give a single **0-100 score** - effectively, each proficiency
> threshold a learner clears is worth 50 points, weighted by the model's confidence that they
> cleared it. The score is then split into the three bands by two cut-points, chosen by
> exhaustively testing every candidate pair against held-out cross-validation predictions on
> the training data, and keeping the pair that classified the most learners correctly. Those
> cut-points are then fixed constants, applied unchanged to new learners. We tried two
> different tree ensembles - a random forest (which reduces variance) and gradient boosting
> (which reduces bias) - so the result does not hinge on one modelling choice.

**Be ready to defend:**

1. **Does the 0-100 step actually help?** Point at `lift_vs_argmax`. Positive means the score
   plus tuned thresholds genuinely beat reading the model's raw prediction. If it is ~0, say
   so plainly: the score is still the required business output, it just isn't adding accuracy.
2. **Why aren't the thresholds round numbers?** They sit where the model actually separates the
   bands. The plateau analysis shows how far they could move without hurting.
3. **Which accuracy number is real?** `test_acc` and `oof_acc`. Never `full_acc`.
4. **Why two models?** Different failure modes (variance vs bias). Agreement between them is
   evidence the signal is real rather than an artefact of one algorithm.
5. **What actually limits performance?** Almost certainly **feature quality**, not model
   choice. If both land in a similar range, that is the classic signature of being limited by
   the information in the features rather than by the algorithm.

---

## Sources

- Frank, E. & Hall, M. (2001). *A Simple Approach to Ordinal Classification.* ECML.
- Breiman, L. (2001). *Random Forests.* Machine Learning 45(1).
- Friedman, J. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
