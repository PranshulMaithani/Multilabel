# Reshaping the 0-100 Score - Two Variations

> Why the raw scores pile up at the extremes, and the **two** reshaping variations in
> `cefr_2_methods_reshaped.ipynb` - both plotted, both leaving **every band and the accuracy
> unchanged**:
>
> - **A - Global bell:** one symmetric bell centred at 50 (most learners 35-65).
> - **B - Per-band ranges:** band 0 -> ~2-40, B1 -> 40-60, band 2 -> ~60-98.
>
> The notebook computes both, graphs both, and prints a comparison table showing that
> train/test/full accuracy are **identical** across raw / A / B (only the split points differ),
> so the choice is purely about the distribution you want to present.
>
> Last updated: 2026-07-24

---

## 1. The symptom

The raw 0-100 scores are **U-shaped**: a peak at **0**, a peak at **100**, and a flat, sparse
shelf in between. Business does not want scores of exactly 0 or 100. The desired output:

- **Band 0 (A1-A2)** in the **low** range (~0-40)
- **Band 1 (B1)** around **50**
- **Band 2 (B2-C1-C2)** in the **high** range (~60-100)
- and **no pile-up at 0 or 100**.

## 2. Why it happens (the cause)

The score is built from the two Frank-Hall probabilities:

```
score = 50 * ( c0 + c1 ),   c0 = P(y>0) = "above A1-A2",  c1 = P(y>1) = "above B1"
```

**Tree ensembles are overconfident.** For a clearly-A1-A2 learner essentially all trees vote
"no", so `c0 ~ 0, c1 ~ 0 -> score ~ 0`. For a clearly-strong learner `c0 ~ 1, c1 ~ 1 ->
score ~ 100`. B1 lands near 50. Only the genuinely borderline learners fall in the middle, and
there aren't many - hence the two extreme peaks and the flat middle. The U-shape is honest;
reshaping it is a **presentation choice**.

## 3a. Variation A - global bell (the chosen output)

Map each learner onto a symmetric **Beta(5, 5)** bell scaled to 0-100. Two modes:

- **Balanced (`GLOBAL_BALANCED=True`, default).** Each band is placed into its own equal 1/3
  **slice** of the bell by its within-band rank. This centres **B1 exactly on 50**, makes the
  mapping **neutral to the 4:7:7 prevalence** (band 0 being rarer doesn't skew placement), and
  because the three slices are disjoint the **band boundaries are exactly preserved** - a B1
  learner can never get an A-score that reads as band 0. Result: band 0 low, B1 ~50, band 2
  high, all on one bell.
- **Pooled (`GLOBAL_BALANCED=False`).** One percentile map over all scores; reflects the real
  prevalence (band 0 lands in the bottom ~22%, so B1 sits a little below 50).

Either way the histogram still shows the real 4:7:7 counts - that is true data, not a bias;
balancing only de-biases the *mapping*. Tune the bell width with `GLOBAL_BETA` (higher =
tighter around 50).

## 3b. Variation B - per-band remap (keeps bands separated)

Rather than one global bell, remap **each band into its own target range** and spread its
learners smoothly inside it.

**Target ranges (`BAND_RANGES`, tunable):**

| Band | CEFR | Target score range | Soft hump around |
|---|---|---|---|
| 0 | A1, A2 | **2 - 40** | ~20 |
| 1 | B1 | **40 - 60** | **50** |
| 2 | B2, C1, C2 | **60 - 98** | ~80 |

**Two steps, per band, fitted on TRAIN only:**

1. **Within-band percentile.** For the learners in a band, convert each raw score to its rank
   *within that band* -> `p` in (0, 1). (`QuantileTransformer(output_distribution="uniform")`,
   one per band.)
2. **Map into the band's range with a soft taper.** Push `p` through a symmetric Beta and scale
   into `[lo, hi]`:

   ```
   reshaped_score = lo + (hi - lo) * Beta.ppf( p ; a = BETA_A, b = BETA_A )
   ```

   `BETA_A = 2` gives a gentle hump centred in the band's range, tapering at the edges so
   **nothing piles up** - not at 0/100 and not at the internal boundaries.
   (`BETA_A = 1` would spread learners flatly across the range; higher = more centred.)

**Result on real-style data:** band 0 -> ~3-39 (median ~20), B1 -> ~40-60 (median ~50),
band 2 -> ~61-97 (median ~80). No learner at 0 or 100.

## 4. Why it is safe - and the trade-off

**Bands and accuracy are unchanged.** The remap is monotonic and done band-by-band, and each
band is placed in a disjoint, ordered interval (0-40 < 40-60 < 60-100). So the adjusted
cut-points are simply the interval boundaries **40 / 60**, and every learner keeps the **exact
same band** - the notebook checks this (0 rows shift). Accuracy, confusion matrices, importance:
all untouched.

**Trade-off: the score becomes relative *within its band*.** A learner's reshaped number now
says *"where they rank among others in the same band"*, mapped into that band's range - not an
absolute probability. A clearly-C2 learner who scored ~98 raw now scores ~80-97 (high in band
2), not 100. That is the intended effect. The absolute value is preserved in the final table as
`{model}_raw_score` if anyone needs it.

**Fit on train, apply everywhere.** The per-band mappers are learned from the training scores
and applied unchanged to test and any new learner (its band decides which mapper is used), so it
is reproducible and leak-free.

## 5. How to tune it

All in the reshaping cell:

- **Move a band's range:** edit `BAND_RANGES`, e.g. `{0:(0,35), 1:(45,55), 2:(65,100)}` to make
  B1 tighter around 50 and band 2 reach 100. (Keep them ordered and non-overlapping so bands
  stay separable.)
- **Within-band shape:** `BETA_A = 1.0` = flat spread across the range; `2.0` = soft hump; `4.0`
  = strongly centred.
- **Turn it off:** `RESHAPE = False` restores the raw scores.

## 6. Where it lives in the notebook

`notebooks/cefr_2_methods_reshaped.ipynb` (a copy of the baseline) adds one section,
**"Reshape the score distribution - per band"**, immediately **before** the final prediction
table. It fits the per-band mappers, stores `score_train_adj` / `score_test_adj` and
`cuts_adj = (40, 60)`, checks no band moved, prints each band's resulting score range, and plots
raw (U-shaped) vs reshaped (three humps). The **final prediction table** then shows both:

- `{model}_raw_score` = `50 * (m1 + m2)` (absolute)
- `{model}_score` = the per-band reshaped score (business deliverable)

alongside `m1`, `m2`, the band, and `ciid` / `split` / `region`.

## 7. Tools

- `sklearn.preprocessing.QuantileTransformer` - within-band percentile mapping.
- `scipy.stats.beta` - the soft within-band taper (`Beta(a, a)` quantile function).
