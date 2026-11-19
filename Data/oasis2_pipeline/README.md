# OASIS-2 data pipeline

Goal: a reusable pipeline (clean data -> labels -> features -> MRI volumes -> baselines) built on OASIS-2, then reused on ADNI and AIBL by only changing the column mapping.

## Setup

```bash
pip install -r requirements.txt
```

Data structure:

```text
data/oasis2/oasis_longitudinal_demographics-*.xlsx
data/oasis2/raw/   <- extract OAS2_RAW_PART1.tar.gz and OAS2_RAW_PART2.tar.gz here
```

`data/` and `outputs/` are in `.gitignore` (the OASIS data use agreement forbids redistribution).

### Windows Spark environment

For the tested Windows configuration:

- Python 3.11
- PySpark 3.5.4
- Java
- Hadoop 3.3.0 Windows binaries

Configure the environment in Git Bash:

```bash
export HADOOP_HOME="C:/Program Files/hadoop-3.3.0"
export PATH="$HADOOP_HOME/bin:$PATH"
export PYSPARK_PYTHON="/path/to/python311/python.exe"
export PYSPARK_DRIVER_PYTHON="/path/to/python311/python.exe"
##Replace `/path/to/python311/python.exe` with the actual Python 3.11 executable path on your computer.
```

For Spark ML model persistence, use:

```bash
JAVA_TOOL_OPTIONS="-Djava.library.path=C:/PROGRA~1/hadoop-3.3.0/bin"
```

Adapt the paths if Hadoop or Python is installed elsewhere. These environment variables must be configured again when starting a new terminal session.

## Run (in order)

| Step | Command | Output |
|---|---|---|
| 1 | `python s1_load_demographics.py` | `outputs/visits_clean.csv` + summary |
| 2a | `python s2_features_pandas.py` | `outputs/landmark{1,2,3}.csv` |
| 2b | `python s2_features_spark.py --landmark 2 --check` | Spark landmark table, checked against Pandas |
| 2c | `JAVA_TOOL_OPTIONS="-Djava.library.path=C:/PROGRA~1/hadoop-3.3.0/bin" py -3.11 s6_spark_ml.py --landmark 1 --folds 5` | Spark ML models, metrics and predictions |
| 3 | `python s3_mri_index.py` | `outputs/mri_index.csv` |
| 4 | `python s4_mri_preprocess.py --size 96` | `outputs/mri_npy/*.npy` |
| 5 | `python s5_baseline.py --landmark 2` | `outputs/baseline_results_L2.csv` |

Look at a scan: `python view_scan.py path/to/mpr-1.nifti.img`

## Spark ML implementation

The Spark ML experiment uses the OASIS-2 landmark 1 dataset.

- Input: one row per eligible subject.
- Cohort: 137 subjects, including 28 progression cases and 109 stable cases.
- Features: age, sex, education, socioeconomic status, MMSE, CDR, nWBV and eTIV.
- Target: `progressed`.
- Models: Logistic Regression and Random Forest.
- Evaluation: stratified 5-fold cross-validation, with out-of-fold predictions.
- Class imbalance: inverse-frequency class weights are calculated from each training fold.
- Final models are trained on all eligible subjects and saved separately.

Generated files:

```text
outputs/
    spark_ml_landmark1_metrics.csv
    spark_ml_landmark1_predictions.csv
    spark_models/
        landmark1_LogisticRegression/
        landmark1_RandomForest/
```

### Preliminary Spark ML results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.5845 | 0.5229 |
| PR-AUC | 0.2507 | 0.2373 |
| Balanced accuracy | 0.5798 | 0.5470 |
| Precision | 0.2679 | 0.2917 |
| Recall | 0.5357 | 0.2500 |
| F1-score | 0.3571 | 0.2692 |

Logistic Regression performs better overall in this experiment. These are preliminary results, not clinical validation.

## The task (landmark design, no leakage)

- Features use only visits 1..L. Label = CDR at any later visit is higher than CDR at visit L (`progressed`).
- Cohort: subjects with >= L+1 visits and CDR <= 0.5 at visit L.
- One row per subject, so splits are by subject.
- Never use the `Group` column as a feature (it is built from all visits = leakage). `group_converted` is kept only as an alternative label.
- Exclude `subject_id` and future-derived information from model features.

## Mapping to the research questions

- RQ1 modality: clinical vs imaging-derived vs both. OASIS-2 has no PET/biomarkers, so this is only partial: finish it on ADNI.
- RQ2 longitudinal: static (visit L) vs visits 1..L, same subjects.
- RQ3 missing data: hide visit-1 information for 25/50/75% of subjects.

## Raw MRI format

Each session has `mpr-1/2/3.nifti.img` + `.hdr` (Analyze 7.5 pair, 256x256x128). Always load the `.img` with nibabel; the `.hdr` must sit next to it. Step 4 uses mpr-1 only.

Not done yet: skull stripping, registration, averaging the 3 scans, bias-field correction (use ANTs/FSL if you train on images).

## Limits to state in your report

- OASIS-2 has only 150 subjects: few progressors, wide confidence intervals.
- The landmark 1 Spark ML cohort has 137 eligible subjects and only 28 progression cases.
- Follow-up duration varies; the target is progression at any observed future visit, not progression within a fixed time horizon.
- Model performance is modest and needs further validation.
- Check the printed cohort sizes before trusting any result.
- Do not interpret these experiments as clinical prediction tools.

---

# Reducing the data (read this first if disk space is a problem)

Formats: OASIS-2 = Analyze pair (`.hdr` + `.img`, one file per scan). ADNI = DICOM (one `.dcm` file per slice, one folder per scan: `ADNI/<PTID>/<sequence>/<scan date>/<Image ID>/*.dcm`). Open both with 3D Slicer, or use `mri_utils.py`.

## Plan

1. Tables carry most of the signal and are small (~100 MB). Use ALL rows of the ADNI tables. MRI volumes (FreeSurfer, `UCSFFSX7`) and PET (`UCBERKELEY_AMY_6MM`, `UCBERKELEY_TAU_6MM`) already exist there as numbers: no need for PET images.
2. Images: keep only a SUBSET converted to small `.npy` (96^3 float16 = 1.7 MB each), then archive/delete the raw files.
3. Dev set = 10 subjects (minutes to run). Experiment set = 100-200 subjects.

## Commands

```bash
python inspect_data.py --adni "Data/T1w_Imaging_Cohort_MRI_1" --oasis "Data/OAS2_RAW_PART1"
python prune_oasis_raw.py
python select_subset.py --n-subjects 10 --out outputs/subset_dev.csv
python convert_dicom.py --subset outputs/subset_dev.csv --size 96
python check_npy.py --dir outputs/adni_npy --n 12
```

`select_subset.py --labels labels.csv` (columns `rid,label`) balances progressors / stable subjects.

Match scans to table visits with PTID (or RID = last number of PTID) + scan date vs EXAMDATE (nearest, within ~90 days).

Limits: DICOM must be uncompressed; resizing to a cube ignores voxel spacing (fine for a baseline).