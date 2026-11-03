"""
run_experiments.py - baseline answers to the three research questions (tabular, CPU only).

RQ1  multimodal vs unimodal         : feature sets dem / cli / mri / bio / combinations
RQ2  longitudinal vs static         : static features vs static + history (earlier visits)
RQ3  robustness to missing data     : drop a whole modality / random values at test time,
                                      model trained normally vs trained with missing-data augmentation

Evaluation: repeated StratifiedGroupKFold (all visits of a subject stay in the same fold -> no leakage),
metric = out-of-fold AUROC, reported as mean +- sd over repeats.

Usage:
    pip install scikit-learn pandas numpy matplotlib
    python run_experiments.py                              # ADNI, label 'progress'
    python run_experiments.py --cohort oasis2
    python run_experiments.py --label conv --repeats 10
Outputs in Data_small/tables/results_<cohort>_<label>/ : rq1.csv rq2.csv rq3.csv rq3.png summary.txt
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ap.add_argument("--data", default=PROJECT_ROOT / "Data_small", type=Path)
ap.add_argument("--cohort", default="adni", choices=["adni", "oasis2"])
ap.add_argument("--label", default="progress", choices=["progress", "conv"])
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--use-dx", action="store_true", help="ADNI: also give the current diagnosis as a feature")
ap.add_argument("--no-cdr", action="store_true", help="drop CDR features (OASIS-2 label is built from CDR)")
a = ap.parse_args()

T = Path(a.data) / "tables"
df = pd.read_csv(T / f"{a.cohort}_visits.csv")
G = json.load(open(T / f"feature_groups_{a.cohort}.json"))
if a.use_dx and a.cohort == "adni":
    G["dx"] = ["dx0_rank"]
if a.no_cdr:
    G["cli"] = [c for c in G["cli"] if "cdr" not in c]
df = df[df[a.label].notna()].reset_index(drop=True)
y = df[a.label].astype(int).values
groups = df["subject_id"].values
out = T / f"results_{a.cohort}_{a.label}"; out.mkdir(exist_ok=True)
lines = []


def say(*x):
    s = " ".join(str(i) for i in x); print(s); lines.append(s)


say(f"cohort={a.cohort} label={a.label}: {len(df)} visits, {df.subject_id.nunique()} subjects, "
    f"positives={y.sum()} ({y.mean()*100:.0f}%)")
if min(y.sum(), (1 - y).sum()) < a.splits:
    raise SystemExit("Too few positives/negatives for cross-validation: lower --splits or change the horizon.")

MODELS = {
    "HGB": lambda s: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                    min_samples_leaf=5, l2_regularization=1.0,
                                                    class_weight="balanced", random_state=s),
    "LogReg": lambda s: make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                      LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")),
}


def cols_of(*names):
    return [c for n in names for c in G.get(n, [])]



def cv_auc(model_name, cols, rows=None, repeats=None, train_aug=False, mask_fn=None, X_override=None):
    """Repeated grouped CV. Removes constant features from HGB using training data only."""
    X = df[cols].values.astype(float) if X_override is None else X_override
    idx_all = np.arange(len(df)) if rows is None else np.where(rows)[0]

    if len(idx_all) < 30:
        say(f"   (only {len(idx_all)} labelled visits here: too few for a reliable AUROC, skipped)")
        return np.nan, np.nan

    aucs = []

    for rep in range(repeats or a.repeats):
        seed = a.seed + rep
        rng = np.random.default_rng(seed)
        n_sp = min(
            a.splits,
            len(set(groups[idx_all][y[idx_all] == 1])),
            len(set(groups[idx_all][y[idx_all] == 0]))
        )

        if n_sp < 2:
            say("   (not enough labelled subjects of both classes in this subset)")
            return np.nan, np.nan

        cv = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=seed)
        pred = np.full(len(df), np.nan)

        for tr, te in cv.split(idx_all, y[idx_all], groups[idx_all]):
            tr, te = idx_all[tr], idx_all[te]
            Xtr, ytr = X[tr], y[tr]
            Xte = X[te].copy()

            if mask_fn is not None:
                Xte = mask_fn(Xte, rng)

            if model_name == "HGB":
                keep = np.array([
                    np.unique(col[np.isfinite(col)]).size >= 2
                    for col in Xtr.T
                ])
                Xtr = Xtr[:, keep]
                Xte = Xte[:, keep]

            if Xtr.shape[1] == 0:
                raise ValueError("No usable features remain in this training fold.")

            if train_aug:
                aug = [mask_fn(Xtr.copy(), rng) for _ in range(2)]
                Xtr = np.vstack([Xtr] + aug)
                ytr = np.concatenate([ytr] * 3)

            m = MODELS[model_name](seed).fit(Xtr, ytr)
            pred[te] = m.predict_proba(Xte)[:, 1]

        ok = ~np.isnan(pred)
        aucs.append(roc_auc_score(y[ok], pred[ok]))

    return float(np.mean(aucs)), float(np.std(aucs))



def fmt(m, s):
    return "n/a" if np.isnan(m) else f"{m:.3f} +- {s:.3f}"


# ------------------------------------------------------------------ RQ1
say("\n" + "=" * 70, "\nRQ1 - multimodal vs unimodal (static features, AUROC)\n" + "=" * 70)
sets = {"demographics": ["dem"], "clinical (MMSE/CDR/ADAS)": ["cli"], "MRI-derived": ["mri"],
        "biomarkers (CSF/plasma/PET)": ["bio"], "clinical + MRI": ["cli", "mri"],
        "MULTIMODAL: dem+cli+mri+bio": ["dem", "cli", "mri", "bio"]}
if "dx" in G:
    sets = {**sets, "MULTIMODAL + diagnosis": ["dem", "cli", "mri", "bio", "dx"]}
rq1 = []
for name, g in sets.items():
    cols = cols_of(*g)
    if not cols:
        continue
    row = {"feature_set": name, "n_features": len(cols)}
    for mn in MODELS:
        m, s = cv_auc(mn, cols); row[mn] = fmt(m, s); row[mn + "_mean"] = m
    rq1.append(row); say(f"{name:34s} k={len(cols):3d}  " + "  ".join(f"{mn}: {row[mn]}" for mn in MODELS))
pd.DataFrame(rq1).to_csv(out / "rq1.csv", index=False)

# ------------------------------------------------------------------ RQ2
say("\n" + "=" * 70, "\nRQ2 - longitudinal (history of earlier visits) vs static\n" + "=" * 70)
static = cols_of("dem", "cli", "mri", "bio")
longi = static + cols_of("hist")
has_hist = (df["hist_n_prev"] >= 1).values
say(f"visits with >=1 earlier visit: {has_hist.sum()} / {len(df)}")
rq2 = []
for scope, rows in (("all visits", None), ("only visits with history", has_hist)):
    for name, cols in (("static (1 visit)", static), ("longitudinal (static + history)", longi)):
        row = {"scope": scope, "features": name}
        for mn in MODELS:
            m, s = cv_auc(mn, cols, rows=rows); row[mn] = fmt(m, s)
        rq2.append(row); say(f"{scope:26s} {name:32s} " + "  ".join(f"{mn}: {row[mn]}" for mn in MODELS))
pd.DataFrame(rq2).to_csv(out / "rq2.csv", index=False)

# ------------------------------------------------------------------ RQ3
say("\n" + "=" * 70, "\nRQ3 - robustness to missing data (multimodal+history features)\n" + "=" * 70)
cols = longi
N_SPLITS = max(2, min(a.splits, len(set(groups[y == 1])), len(set(groups[y == 0]))))
col_idx = {g: [cols.index(c) for c in G.get(g, []) if c in cols] for g in ("dem", "cli", "mri", "bio", "hist")}


def drop_group(g):
    def f(X, rng):
        X[:, col_idx[g]] = np.nan; return X
    return f


def drop_random(p):
    def f(X, rng):
        X[rng.random(X.shape) < p] = np.nan; return X
    return f


def aug_mask(X, rng):                       # training augmentation: random modality OR random-value dropout
    for g, ix in col_idx.items():
        if ix and rng.random() < 0.3:
            X[:, ix] = np.nan
    X[rng.random(X.shape) < 0.1] = np.nan
    return X



def run_cond(mn, test_mask, aug):
    X = df[cols].values.astype(float)
    aucs = []

    for rep in range(a.repeats):
        seed = a.seed + rep
        rng = np.random.default_rng(seed)
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed
        )
        pred = np.full(len(df), np.nan)

        for tr, te in cv.split(X, y, groups):
            Xtr, ytr = X[tr].copy(), y[tr]

            keep = np.ones(Xtr.shape[1], dtype=bool)

            if mn == "HGB":
                keep = np.array([
                    np.unique(col[np.isfinite(col)]).size >= 2
                    for col in Xtr.T
                ])

            if not keep.any():
                raise ValueError("No usable features remain in this training fold.")

            if aug:
                Xtr = np.vstack(
                    [Xtr] + [aug_mask(Xtr.copy(), rng) for _ in range(2)]
                )
                ytr = np.concatenate([ytr] * 3)

            Xte = X[te].copy()

            if test_mask is not None:
                Xte = test_mask(Xte, rng)

            Xtr = Xtr[:, keep]
            Xte = Xte[:, keep]

            model = MODELS[mn](seed)
            model.fit(Xtr, ytr)
            pred[te] = model.predict_proba(Xte)[:, 1]

        aucs.append(roc_auc_score(y, pred))

    return float(np.mean(aucs)), float(np.std(aucs))



conds = [("no missing", None)] + [(f"drop all '{g}'", drop_group(g)) for g in ("cli", "mri", "bio", "hist") if col_idx[g]] \
        + [(f"random {int(p*100)}% values", drop_random(p)) for p in (0.1, 0.2, 0.3, 0.5)]
rq3 = []
for cname, mask in conds:
    row = {"condition": cname}
    for mn in MODELS:
        for aug in (False, True):
            m, s = run_cond(mn, mask, aug)
            row[f"{mn}{'+aug' if aug else ''}"] = round(m, 3)
    rq3.append(row)
    say(f"{cname:22s} " + "  ".join(f"{k}: {v:.3f}" for k, v in row.items() if k != "condition"))
R3 = pd.DataFrame(rq3); R3.to_csv(out / "rq3.csv", index=False)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rnd = R3[R3.condition.str.startswith("random") | (R3.condition == "no missing")].copy()
    rnd["p"] = [0] + [10, 20, 30, 50][: len(rnd) - 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in [c for c in R3.columns if c != "condition"]:
        ax.plot(rnd.p, rnd[k], marker="o", label=k, ls="--" if "aug" in k else "-")
    ax.set_xlabel("% of feature values removed at test time"); ax.set_ylabel("AUROC")
    ax.set_title(f"RQ3 robustness ({a.cohort}, {a.label})"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "rq3.png", dpi=120)
except Exception as e:
    say("(plot skipped:", e, ")")

say(f"\nSaved to {out}")
say("Reminder: with ~100 labelled visits, differences < ~0.05 AUROC are within noise; compare the +- values.")
(out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
