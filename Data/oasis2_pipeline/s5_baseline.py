"""Step 5: quick tabular baselines to prove the pipeline end-to-end and give a first answer to RQ1-RQ3.

Cohort = landmark dataset (one row per subject, so every split is automatically by subject).
CV = repeated stratified k-fold; metrics = ROC AUC and balanced accuracy.
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import config

warnings.filterwarnings("ignore")
S = lambda cols: [f"s_{c}" for c in cols]
CLIN = S(["age", "sex", "educ", "ses", "mmse", "cdr"])
IMG = S(["age", "sex", "nwbv", "etiv"])                 # imaging-derived + age/sex covariates
STATIC = S(["age", "sex", "educ", "ses", "mmse", "cdr", "nwbv", "etiv"])


def models():
    return {"LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "RandForest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                                 random_state=0, n_jobs=1)}


def evaluate(df, cols, name, exp, rows, folds, repeats):
    y = df["progressed"].values
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
    for mname, model in models().items():
        pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()), ("clf", model)])
        r = cross_validate(pipe, df[cols], y, cv=cv, scoring=["roc_auc", "balanced_accuracy"])
        rows.append({"experiment": exp, "features": name, "model": mname,
                     "auc": r["test_roc_auc"].mean(), "auc_std": r["test_roc_auc"].std(),
                     "bal_acc": r["test_balanced_accuracy"].mean()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmark", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    df = pd.read_csv(config.OUT_DIR / f"landmark{a.landmark}.csv")
    n_pos = int(df["progressed"].sum())
    print(f"Cohort: {len(df)} subjects, {n_pos} progressed, {len(df) - n_pos} stable")
    if min(n_pos, len(df) - n_pos) < 6:
        print("Too few subjects in one class for a meaningful CV. Try --landmark 1.")
        return
    folds = min(5, n_pos, len(df) - n_pos)
    LONG = STATIC + [c for c in df.columns if c.startswith(("l_", "b_"))]
    rows = []
    # RQ1: unimodal vs multimodal (OASIS-2 only has clinical + MRI-derived volumes)
    for n, c in [("clinical", CLIN), ("imaging-derived", IMG), ("clinical+imaging", STATIC)]:
        evaluate(df, c, n, "RQ1 modality", rows, folds, a.repeats)
    # RQ2: static (visit L only) vs longitudinal (visits 1..L), same subjects
    if a.landmark >= 2:
        evaluate(df, STATIC, "static", "RQ2 visits", rows, folds, a.repeats)
        evaluate(df, LONG, "longitudinal", "RQ2 visits", rows, folds, a.repeats)
        # RQ3: simulate a missing earlier visit for a fraction p of subjects
        rng = np.random.default_rng(0)
        for p in (0.25, 0.5, 0.75):
            d = df.copy()
            hide = rng.random(len(d)) < p
            d.loc[hide, [c for c in LONG if c.startswith(("l_", "b_"))]] = np.nan
            evaluate(d, LONG, f"longitudinal, {int(p * 100)}% missing visit 1", "RQ3 missing", rows, folds, a.repeats)
    res = pd.DataFrame(rows)
    res.to_csv(config.OUT_DIR / f"baseline_results_L{a.landmark}.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
