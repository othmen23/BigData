"""
build_tables.py - turn Data_small/ into clean modeling tables (one row = one MRI visit).

Outputs in Data_small/tables/:
    adni_visits.csv, oasis2_visits.csv   features + labels + image_path
    feature_groups_adni.json, feature_groups_oasis2.json   which column belongs to which modality
    tables_report.txt                    label prevalence, censoring, missingness

Label (Project RQ: "progression"), evaluated at a horizon H (default 36 months) after the visit:
    progress = 1 if the diagnosis gets WORSE within H  (ADNI: CN->MCI/AD, MCI->AD ; OASIS-2: CDR increases)
    conv     = 1 if the subject REACHES dementia within H (ADNI: AD ; OASIS-2: CDR >= 1)
    0 only if follow-up is long enough to be sure (>= H - 90 days), else NaN (= censored, dropped in training).
    Subjects already at the last stage (AD / CDR 2) are NaN (cannot progress).

Usage:  python build_tables.py [--data "...\\Data_small"] [--horizon-months 36]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ap.add_argument("--data", default=PROJECT_ROOT / "Data_small", type=Path)
ap.add_argument("--horizon-months", type=float, default=36)
ap.add_argument("--censor-slack-days", type=int, default=90)
a = ap.parse_args()

D = Path(a.data)
T = D / "ADNI_Tables"
OUT = D / "tables_assessment"; OUT.mkdir(exist_ok=True)
H = pd.Timedelta(days=a.horizon_months * 30.44)
SLACK = pd.Timedelta(days=a.censor_slack_days)
report = []


def say(*x):
    s = " ".join(str(i) for i in x); print(s); report.append(s)


def rd(pattern):
    f = next(T.glob(pattern), None)
    if f is None:
        say(f"  [warn] table not found: {pattern}"); return None
    return pd.read_csv(f, low_memory=False)


def num(df, cols, nonneg=True):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if nonneg:
                df.loc[df[c] < 0, c] = np.nan      # ADNI codes missing as -4 / -1
    return df


def attach(assessments, tbl, datecol, cols, rename, tol_days, tag):
    """Attach the latest available record on or before each clinical assessment."""
    if tbl is None:
        return assessments

    cols = [c for c in cols if c in tbl.columns]
    if not cols or datecol not in tbl.columns:
        return assessments

    t = tbl[["PTID", datecol] + cols].copy()
    t["_date"] = pd.to_datetime(t[datecol], errors="coerce")
    t = t.dropna(subset=["PTID", "_date"])
    t = t[t[cols].notna().any(axis=1)]
    t = t.rename(columns={c: rename.get(c, c) for c in cols})
    t = t.sort_values("_date")

    left = assessments.copy()
    left = left.sort_values("assessment_date")

    result = pd.merge_asof(
        left,
        t.drop(columns=datecol),
        left_on="assessment_date",
        right_on="_date",
        by="PTID",
        tolerance=pd.Timedelta(days=tol_days),
        direction="backward"
    )

    result[f"{tag}_gap_days"] = (
        result["assessment_date"] - result["_date"]
    ).dt.days

    return result.drop(columns="_date").sort_values(
        ["PTID", "assessment_date"]
    ).reset_index(drop=True)


def label_scan(t_scan, ev, max_rank, top_rank, tol=pd.Timedelta(days=180)):
    nan = dict(
        dx0_rank=np.nan,
        dx0_gap_days=np.nan,
        followup_days=np.nan,
        progress=np.nan,
        conv=np.nan
    )

    if ev is None or ev.empty:
        return nan

    baseline = ev[ev.date <= t_scan]

    if baseline.empty:
        return nan

    baseline = baseline[baseline.date >= t_scan - tol]

    if baseline.empty:
        return nan

    t0 = baseline.date.iloc[-1]
    r0 = baseline["rank"].iloc[-1]

    later = ev[
        (ev.date > t_scan) &
        (ev.date <= t_scan + H)
    ]

    follow = ev.date.max() - t_scan

    out = dict(
        dx0_rank=r0,
        dx0_gap_days=(t_scan - t0).days,
        followup_days=follow.days
    )

    enough = follow >= H - SLACK

    if r0 >= max_rank:
        out.update(progress=np.nan, conv=np.nan)
        return out

    out["progress"] = (
        1.0 if (later["rank"] > r0).any()
        else (0.0 if enough else np.nan)
    )

    if r0 >= top_rank:
        out["conv"] = np.nan
    else:
        out["conv"] = (
            1.0 if (later["rank"] >= top_rank).any()
            else (0.0 if enough else np.nan)
    )

    return out


def add_history(df, subj_col, date_col, vars_, prefix="hist"):
    """Longitudinal trajectory features using the current and previous visits: number of prior visits, time since first,
    change since previous visit, slope since first visit (per year)."""
    df = df.sort_values([subj_col, date_col]).reset_index(drop=True)
    new = {f"{prefix}_n_prev": np.zeros(len(df)), f"{prefix}_yrs_since_first": np.full(len(df), np.nan)}
    for v in vars_:
        new[f"{prefix}_{v}_d_prev"] = np.full(len(df), np.nan)
        new[f"{prefix}_{v}_slope"] = np.full(len(df), np.nan)
    for _, idx in df.groupby(subj_col).indices.items():
        idx = list(idx)
        for k, i in enumerate(idx):
            new[f"{prefix}_n_prev"][i] = k
            if k == 0:
                continue
            yrs = (df.loc[i, date_col] - df.loc[idx[0], date_col]).days / 365.25
            new[f"{prefix}_yrs_since_first"][i] = yrs
            for v in vars_:
                now, prev, first = df.loc[i, v], df.loc[idx[k - 1], v], df.loc[idx[0], v]
                new[f"{prefix}_{v}_d_prev"][i] = now - prev
                if yrs > 0.2:
                    new[f"{prefix}_{v}_slope"][i] = (now - first) / yrs
    return pd.concat([df, pd.DataFrame(new)], axis=1)


# ======================================================================= ADNI
say("=" * 70, "\nADNI\n" + "=" * 70)
man = pd.read_csv(D / "manifest_adni.csv")
scans = man[["PTID", "date", "stratum", "subset_path"]].rename(columns={"subset_path": "image_path"})
scans["scan_date"] = pd.to_datetime(scans.date)
scans = scans.drop(columns="date").sort_values(["PTID", "scan_date"]).reset_index(drop=True)
say(f"{scans.PTID.nunique()} subjects, {len(scans)} MRI visits")

# --- clinical scores
mmse = num(rd("*MMSE*.csv"), ["MMSCORE"])
cdr = num(rd("*_CDR_*.csv"), ["CDGLOBAL", "CDRSB"])
adas = num(rd("*ADAS*.csv"), ["TOTAL13"])
scans = attach(scans, mmse, "VISDATE", ["MMSCORE"], {"MMSCORE": "cli_mmse"}, 120, "mmse")
scans = attach(scans, cdr, "VISDATE", ["CDGLOBAL", "CDRSB"], {"CDGLOBAL": "cli_cdr", "CDRSB": "cli_cdrsb"}, 120, "cdr")
scans = attach(scans, adas, "VISDATE", ["TOTAL13"], {"TOTAL13": "cli_adas13"}, 120, "adas")

# --- MRI-derived (FreeSurfer volumes, normalised by ICV)
fs = rd("*UCSFFSX7*.csv")
fs_cols = []
if fs is not None:
    if "OVERALLQC" in fs.columns:
        fs = fs[~fs.OVERALLQC.astype(str).str.lower().str.startswith("fail")]
    vol_cols = [c for c in fs.columns if re.fullmatch(r"ST\d+(CV|SV)", c)]
    fs = num(fs, vol_cols)
    icv = fs["ST10CV"] if "ST10CV" in fs.columns else None
    ren = {}
    for c in vol_cols:
        if icv is not None and c != "ST10CV":
            fs[c] = fs[c] / icv * 1000.0           # per-mille of ICV
        ren[c] = "mri_" + c
    if {"ST29SV", "ST88SV"} <= set(vol_cols):
        fs["mri_hippo"] = fs.ST29SV + fs.ST88SV
    if {"ST24CV", "ST83CV"} <= set(vol_cols):
        fs["mri_entorhinal"] = fs.ST24CV + fs.ST83CV
    if {"ST37SV", "ST96SV"} <= set(vol_cols):
        fs["mri_ventricles"] = fs.ST37SV + fs.ST96SV
    extra = [c for c in ("mri_hippo", "mri_entorhinal", "mri_ventricles") if c in fs.columns]
    fs_cols = vol_cols + extra
    scans = attach(scans, fs, "EXAMDATE", fs_cols, ren, 120, "fs")
    fs_cols = [ren.get(c, c) for c in fs_cols]

# --- demographics (static per subject)
dem = rd("*PTDEMOG*.csv")
apoe = rd("*APOERES*.csv")
sub = pd.DataFrame(index=scans.PTID.unique()); sub.index.name = "PTID"
if dem is not None:
    dem = num(dem, ["PTGENDER", "PTEDUCAT", "PTDOBYY"], nonneg=True)
    if "PTDOB" in dem.columns:
        yr = dem.PTDOB.astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)
        dem["PTDOBYY"] = dem.get("PTDOBYY", pd.Series(np.nan, index=dem.index)).fillna(yr)
    g = dem.groupby("PTID")[["PTGENDER", "PTEDUCAT", "PTDOBYY"]].first()
    sub = sub.join(g)
if apoe is not None and "GENOTYPE" in apoe.columns:
    apoe["dem_apoe4"] = apoe.GENOTYPE.astype(str).str.count("4").where(apoe.GENOTYPE.notna())
    sub = sub.join(apoe.groupby("PTID").dem_apoe4.first())
scans = scans.merge(sub.reset_index(), on="PTID", how="left")
for c in ("PTGENDER", "PTEDUCAT", "PTDOBYY", "dem_apoe4"):
    if c not in scans.columns:
        scans[c] = np.nan
scans["dem_age"] = scans.scan_date.dt.year + scans.scan_date.dt.dayofyear / 365.25 - scans.get("PTDOBYY")
scans["dem_sex_female"] = (scans.get("PTGENDER") == 2).astype(float).where(scans.get("PTGENDER").notna())
scans["dem_educ"] = scans.get("PTEDUCAT")
scans = scans.drop(columns=[c for c in ("PTGENDER", "PTEDUCAT", "PTDOBYY") if c in scans.columns])

# --- biomarkers
bio_cols = []
csf = num(rd("*ROCHE*.csv"), ["ABETA42", "TAU", "PTAU"])
if csf is not None:
    csf["bio_ptau_ab42"] = csf.PTAU / csf.ABETA42
    scans = attach(scans, csf, "EXAMDATE", ["ABETA42", "TAU", "PTAU", "bio_ptau_ab42"],
                   {"ABETA42": "bio_abeta42", "TAU": "bio_tau", "PTAU": "bio_ptau"}, 365, "csf")
pl = num(rd("*PLASMA*.csv"), ["pT217_F", "AB42_AB40_F", "NfL_Q", "GFAP_Q"])
scans = attach(scans, pl, "EXAMDATE", ["pT217_F", "AB42_AB40_F", "NfL_Q", "GFAP_Q"],
               {"pT217_F": "bio_pt217", "AB42_AB40_F": "bio_ab42_40", "NfL_Q": "bio_nfl", "GFAP_Q": "bio_gfap"}, 365, "plasma")
amy = rd("*AMY_6MM*.csv")
if amy is not None:
    amy = num(amy, ["SUMMARY_SUVR", "AMYLOID_STATUS"]); amy = num(amy, ["CENTILOIDS"], nonneg=False)
scans = attach(scans, amy, "SCANDATE", ["CENTILOIDS", "SUMMARY_SUVR", "AMYLOID_STATUS"],
               {"CENTILOIDS": "bio_amy_centiloid", "SUMMARY_SUVR": "bio_amy_suvr", "AMYLOID_STATUS": "bio_amy_status"}, 365, "amy")
tau = num(rd("*TAU_6MM*.csv"), ["META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"])
scans = attach(scans, tau, "SCANDATE", ["META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"],
               {"META_TEMPORAL_SUVR": "bio_tau_meta", "CTX_ENTORHINAL_SUVR": "bio_tau_ent"}, 365, "taupet")

# --- labels from the diagnosis table
dxs = rd("*DXSUM*.csv")
dxs = dxs[dxs.DIAGNOSIS.isin([1, 2, 3])].copy()
dxs["date"] = pd.to_datetime(dxs.EXAMDATE, errors="coerce")
dxs["rank"] = dxs.DIAGNOSIS.map({1: 0, 2: 1, 3: 2})          # 0=CN 1=MCI 2=AD
dxs = dxs.dropna(subset=["date"]).sort_values("date")
events = {k: g[["date", "rank"]].reset_index(drop=True) for k, g in dxs.groupby("PTID")}
lab = [label_scan(r.scan_date, events.get(r.PTID), max_rank=2, top_rank=2) for r in scans.itertuples()]
scans = pd.concat([scans, pd.DataFrame(lab)], axis=1)
scans["dx0"] = scans.dx0_rank.map({0: "CN", 1: "MCI", 2: "AD"})

# --- history (RQ2)
hist_vars = [c for c in ["cli_mmse", "cli_cdrsb", "cli_adas13", "mri_hippo", "mri_entorhinal", "mri_ventricles"] if c in scans.columns]
scans = add_history(scans, "PTID", "scan_date", hist_vars)
scans["subject_id"] = scans.PTID
scans["visit_idx"] = scans.groupby("PTID").cumcount()

groups = {
    k: [c for c in scans.columns if c.startswith(k + "_") and pd.api.types.is_numeric_dtype(scans[c])]
    for k in ("dem", "cli", "mri", "bio", "hist")
}
json.dump(groups, open(OUT / "feature_groups_adni.json", "w"), indent=1)
scans.to_csv(OUT / "adni_visits.csv", index=False)

say(f"\nhorizon = {a.horizon_months} months")
say("diagnosis at scan:", scans.dx0.value_counts(dropna=False).to_dict())
for L in ("progress", "conv"):
    v = scans[L]
    say(f"label '{L}': positive={int((v==1).sum())} negative={int((v==0).sum())} "
        f"censored/excluded={int(v.isna().sum())}  (subjects with a label: {scans[v.notna()].PTID.nunique()})")
say("feature groups:", {k: len(v) for k, v in groups.items()})
say("\nmissing % per modality (share of visits with NO value in the whole group):")
for k, cols in groups.items():
    if k in ("hist",):
        say(f"  {k:5s}: {(scans.hist_n_prev == 0).mean()*100:.0f}% of visits have no earlier visit"); continue
    say(f"  {k:5s}: {scans[cols].isna().all(axis=1).mean()*100:.0f}%")

# ===================================================================== OASIS-2
say("\n" + "=" * 70, "\nOASIS-2\n" + "=" * 70)
o = pd.read_csv(D / "manifest_oasis2.csv")
o["scan_date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(o["MR Delay"], unit="D")
o = o.rename(columns={"Subject ID": "subject_id", "MRI ID": "session_id", "subset_path": "image_path"})
o["dem_age"] = o.Age; o["dem_sex_female"] = (o["M/F"] == "F").astype(float)
o["dem_educ"] = o.EDUC; o["dem_ses"] = o.SES
o["cli_mmse"] = o.MMSE; o["cli_cdr"] = o.CDR
o["mri_etiv"] = o.eTIV; o["mri_nwbv"] = o.nWBV; o["mri_asf"] = o.ASF
ev = {k: pd.DataFrame({"date": g.scan_date.values, "rank": g.CDR.values}).sort_values("date").reset_index(drop=True)
      for k, g in o.groupby("subject_id")}
lab = [label_scan(r.scan_date, ev.get(r.subject_id), max_rank=2.0, top_rank=1.0) for r in o.itertuples()]
o = pd.concat([o.reset_index(drop=True), pd.DataFrame(lab)], axis=1)
o = add_history(o, "subject_id", "scan_date", ["cli_mmse", "mri_nwbv"])
o["visit_idx"] = o.groupby("subject_id").cumcount()
gro = {k: [c for c in o.columns if c.startswith(k + "_") and pd.api.types.is_numeric_dtype(o[c])]
       for k in ("dem", "cli", "mri", "hist")}
gro["bio"] = []
json.dump(gro, open(OUT / "feature_groups_oasis2.json", "w"), indent=1)
o.to_csv(OUT / "oasis2_visits.csv", index=False)
say(f"{o.subject_id.nunique()} subjects, {len(o)} MRI visits, group labels: {o.drop_duplicates('subject_id').Group.value_counts().to_dict()}")
for L in ("progress", "conv"):
    v = o[L]
    say(f"label '{L}': positive={int((v==1).sum())} negative={int((v==0).sum())} censored/excluded={int(v.isna().sum())}")
say("NOTE: OASIS-2 labels come from CDR (not a clinical diagnosis) -> do not pool blindly with ADNI.")

(OUT / "tables_report.txt").write_text("\n".join(report), encoding="utf-8")
say(f"\nSaved to {OUT}")
