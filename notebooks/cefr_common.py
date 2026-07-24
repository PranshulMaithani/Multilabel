"""Shared code for the CEFR 0-100 scoring pipeline.

Holds the ordinal model class, the score helpers, the two reshapers, and a single
`score_dataframe(bundle, df)` inference call. Imported by BOTH the training notebook
(`cefr_2_methods_reshaped.ipynb`) and the inference notebook (`cefr_inference.ipynb`)
so that models saved with joblib load cleanly in either.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy.stats import beta as _beta

# ---- band definition (must match the training notebook) ----
BAND_MAP = {"A1": 0, "A2": 0, "B1": 1, "B2": 2, "C1": 2, "C2": 2}
BAND_NAMES = ["A1-A2", "B1", "B2-C1-C2"]
N_BANDS = 3
BAND_ANCHORS = np.array([0.0, 50.0, 100.0])
_EPS = 1e-3


class FrankHallOrdinal(BaseEstimator, ClassifierMixin):
    """Ordinal decomposition (Frank & Hall, 2001).

    For K ordered classes, fit K-1 binary models P(y > k), then difference the
    cumulative probabilities into per-class probabilities.

    Kept identical to the training notebook so pickled instances load here.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.K_ = len(self.classes_)
        self.models_ = []
        for k in range(self.K_ - 1):
            yk = (y > self.classes_[k]).astype(int)
            if len(np.unique(yk)) < 2:
                self.models_.append(("const", float(yk[0])))
            else:
                m = clone(self.base_estimator)
                m.fit(X, yk)
                self.models_.append(("model", m))
        return self

    def predict_proba(self, X):
        n = len(X)
        cum = np.zeros((n, self.K_ - 1))
        for k, (kind, m) in enumerate(self.models_):
            cum[:, k] = m if kind == "const" else m.predict_proba(X)[:, 1]
        cum = np.minimum.accumulate(cum, axis=1)   # P(y>k) must be non-increasing

        p = np.zeros((n, self.K_))
        p[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, self.K_ - 1):
            p[:, k] = cum[:, k - 1] - cum[:, k]
        p[:, -1] = cum[:, -1]

        p = np.clip(p, 1e-9, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---- scoring helpers ----
def proba_to_score(proba, anchors=BAND_ANCHORS):
    return np.asarray(proba) @ np.asarray(anchors, dtype=float)


def apply_cutpoints(scores, t1, t2):
    scores = np.asarray(scores)
    return np.where(scores <= t1, 0, np.where(scores <= t2, 1, 2))


def frankhall_cumulative(pipe, X):
    """The two sub-model outputs per row: c0 = P(y>0), c1 = P(y>1) (monotonic-adjusted)."""
    fh = pipe.steps[-1][1]
    Xt = X
    for _, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    cols = [np.full(len(X), m) if kind == "const" else m.predict_proba(Xt)[:, 1]
            for kind, m in fh.models_]
    return np.minimum.accumulate(np.column_stack(cols), axis=1)


# ---- reshapers (apply pre-fitted transformers loaded from the bundle) ----
def _bell(p, beta_a):
    return 100.0 * _beta.ppf(np.clip(p, _EPS, 1 - _EPS), beta_a, beta_a)


def apply_global(model, s, beta_a, bands=None):
    """Global-bell reshape. `model` is ("pooled", qt) or ("balanced", {band: qt}).

    balanced: each band is placed into its own equal 1/K slice of the bell by within-band rank,
    so B1 is centred on 50, the mapping is neutral to the 4:7:7 prevalence, AND the band
    boundaries are exactly preserved (disjoint slices). Needs `bands` (the raw-score band of
    each row).
    """
    s = np.asarray(s, float)
    kind, obj = model
    if kind == "pooled":
        return _bell(obj.transform(s.reshape(-1, 1)).ravel(), beta_a)
    if bands is None:
        raise ValueError("balanced global reshape needs `bands`")
    bands = np.asarray(bands)
    K = len(obj)
    p = np.empty(len(s), float)
    for b, qt in obj.items():
        m = bands == b
        if not m.any():
            continue
        if qt is None:
            r = np.full(int(m.sum()), 0.5)
        else:
            r = np.clip(qt.transform(s[m].reshape(-1, 1)).ravel(), 0.0, 1.0)
        p[m] = (b + _EPS + r * (1.0 - 2 * _EPS)) / K     # strictly inside the b-th slice
    return _bell(p, beta_a)


def balanced_bell_cuts(beta_a, n_bands=N_BANDS):
    """Bell-scale values of the K-1 band boundaries for the balanced global reshape."""
    return tuple(float(_bell(np.array([j / n_bands]), beta_a)[0]) for j in range(1, n_bands))


def apply_perband(trans, s, bands, band_ranges, beta_a):
    s = np.asarray(s, float)
    bands = np.asarray(bands)
    out = np.empty(len(s), float)
    for b in range(N_BANDS):
        m = bands == b
        if not m.any():
            continue
        lo, hi = band_ranges[b]
        qt = trans.get(b)
        if qt is None:
            out[m] = (lo + hi) / 2.0
        else:
            p = np.clip(qt.transform(s[m].reshape(-1, 1)).ravel(), _EPS, 1 - _EPS)
            out[m] = lo + (hi - lo) * _beta.ppf(p, beta_a, beta_a)
    return out


# ---- one-call inference ----
def score_dataframe(bundle, df, keep_cols=("ciid", "location", "split")):
    """Run every saved model on a raw feature dataframe and return one prediction table.

    For each model `k` the output has: {k}_m1, {k}_m2, {k}_raw_score, {k}_bell_score,
    {k}_perband_score, {k}_band. Any of `keep_cols` present in `df` are passed through.
    """
    feats = bundle["feature_cols"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing feature columns: {missing}")

    X = df[feats].astype(float)
    names = bundle.get("band_names", BAND_NAMES)
    anchors = bundle.get("band_anchors", BAND_ANCHORS)
    cfg = bundle["reshape_cfg"]

    out = pd.DataFrame(index=df.index)
    for c in keep_cols:
        if c in df.columns:
            out[c] = df[c].values

    for key, mdl in bundle["models"].items():
        pipe = mdl["pipe"]
        cuts = mdl["cuts"]
        proba = pipe.predict_proba(X)
        raw = proba_to_score(proba, anchors)
        cum = frankhall_cumulative(pipe, X)
        bands = apply_cutpoints(raw, *cuts)
        bell = apply_global(mdl["reshaper_global"], raw, cfg["global_beta"], bands)
        pband = apply_perband(mdl["reshaper_perband"], raw, bands,
                              cfg["band_ranges"], cfg["band_beta"])
        out[f"{key}_m1"] = np.round(cum[:, 0], 4)
        out[f"{key}_m2"] = np.round(cum[:, 1], 4)
        out[f"{key}_raw_score"] = np.round(raw, 2)
        out[f"{key}_bell_score"] = np.round(bell, 2)
        out[f"{key}_perband_score"] = np.round(pband, 2)
        out[f"{key}_band"] = [names[i] for i in bands]
    return out
