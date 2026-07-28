# Hierarchical Bell-Curve Score (alternative approach)

> A different route to a bell-shaped 0-100 score, in `notebooks/cefr_hierarchical_bell.ipynb`.
> The bell comes from the **models' own confidence** - **no** uniform/quantile reshaping.
>
> Last updated: 2026-07-24

## The idea

Two **hierarchical** binary classifiers instead of Frank-Hall:

```
                 all learners
                      |
    [Stage 1]  band 0  vs  {band 1, band 2}       -> p_up  = P(upper)
                                  |
                    [Stage 2]  band 1 vs band 2    -> p_two = P(band2 | upper)
                    (trained on the upper subset)
```

- `P(band0) = 1 - p_up`
- `P(band1) = p_up * (1 - p_two)`
- `P(band2) = p_up * p_two`
- predicted band = argmax

This is **not** Frank-Hall: Stage 2 is a *conditional* model trained only on the band-1/band-2
rows, so it is a genuinely harder, lower-confidence problem.

## Why it produces a bell (no reshaping)

Two facts combine:

1. **The log-odds of a probability is ~normally distributed**, even though the raw probability
   piles up at 0 and 1. So scoring on `logit(confidence)` instead of the probability gives a
   bell, not a U.
2. **Band 1 vs band 2 is hard**, so Stage 2's confidence is low for many learners (`p_two`
   near 0.5) - they land in the **middle**, filling the centre of the bell.

**Score** = standardised confidence margin:

```
margin = logit(p_up) + p_up * logit(p_two)
score  = 50 + SPREAD * (margin - mean_train) / std_train,   clipped to [1, 99]
```

Centred at 50, band 0 low, B1 ~50, band 2 high, **nobody at 0 or 100** - straight from the
model, with no quantile/uniform transform anywhere.

Prototype on realistic synthetic data (band 0 separable, band 1/2 overlapping):

| score | shape |
|---|---|
| expected-value `50*(P1+2*P2)` | U-shaped, band 0 piled at ~0 |
| **hierarchical confidence margin** | **bell**, band medians ~23 / ~51 / ~65, ends empty |

## Trade-offs vs the Frank-Hall + reshaping notebook

- **Pro:** the bell is intrinsic to the model's confidence - no post-hoc reshaping to justify.
- **Con:** Stage 2 trains on a subset (the upper group), so it sees less data; accuracy can be
  a touch lower than the shared Frank-Hall models, especially at small n.
- The `score` is a continuous **confidence** measure, so bands can overlap slightly on the
  score axis (honest uncertainty). Report `pred_band` (the hierarchical argmax) as the class;
  `score` is the 0-100 presentation.

## Knobs

- `SPREAD` - bell width (score = 50 +/- SPREAD * z). Larger spreads the score over more of 0-100.
- `CALIBRATE` - calibrate the two classifiers (sigmoid) for smoother confidence and a cleaner
  bell. On by default.
- Rigour upgrade: standardise the margin using **out-of-fold** train predictions (the notebook
  uses in-sample train stats for simplicity; the held-out test bell is unaffected).

## Tools

`RandomForestClassifier`, `CalibratedClassifierCV` (scikit-learn). Standalone notebook - does
not use `cefr_common.py` or the Frank-Hall pipeline.
