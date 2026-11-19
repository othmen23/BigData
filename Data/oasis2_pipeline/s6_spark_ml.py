import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.functions import vector_to_array

import config


FEATURES = [
    "s_age",
    "s_sex",
    "s_educ",
    "s_ses",
    "s_mmse",
    "s_cdr",
    "s_nwbv",
    "s_etiv"
]

TARGET = "progressed"


def add_class_weights(df):
    counts = {
        row[TARGET]: row["count"]
        for row in df.groupBy(TARGET).count().collect()
    }

    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)

    if n0 == 0 or n1 == 0:
        raise ValueError("Both target classes must be present.")

    w0 = (n0 + n1) / (2.0 * n0)
    w1 = (n0 + n1) / (2.0 * n1)

    return df.withColumn(
        "class_weight",
        F.when(F.col(TARGET) == 1, F.lit(w1)).otherwise(F.lit(w0))
    )


def build_pipeline(model_name):
    imputed = [f"{c}_imputed" for c in FEATURES]

    stages = [
        Imputer(
            inputCols=FEATURES,
            outputCols=imputed,
            strategy="median"
        ),
        VectorAssembler(
            inputCols=imputed,
            outputCol="raw_features"
        ),
        StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=True,
            withStd=True
        )
    ]

    if model_name == "LogisticRegression":
        classifier = LogisticRegression(
            featuresCol="features",
            labelCol=TARGET,
            weightCol="class_weight",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.0
        )

    elif model_name == "RandomForest":
        classifier = RandomForestClassifier(
            featuresCol="features",
            labelCol=TARGET,
            weightCol="class_weight",
            numTrees=100,
            maxDepth=4,
            minInstancesPerNode=2,
            seed=42
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    stages.append(classifier)
    return Pipeline(stages=stages)


def evaluate_predictions(y_true, y_pred, scores):
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    input_path = config.OUT_DIR / f"landmark{args.landmark}.csv"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    required = FEATURES + [TARGET, "subject_id"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()
    df[TARGET] = df[TARGET].astype(int)

    if df[FEATURES].isna().all().any():
        raise ValueError("At least one feature is completely empty.")

    y = df[TARGET].values
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive

    if min(n_positive, n_negative) < args.folds:
        raise ValueError("Not enough examples for the requested folds.")

    print("\n========== OASIS-2 SPARK ML ==========")
    print(f"Dataset: {input_path.name}")
    print(f"Subjects: {len(df)}")
    print(f"Progression: {n_positive}")
    print(f"Stable: {n_negative}")
    print(f"Features: {len(FEATURES)}")
    print(f"Cross-validation folds: {args.folds}")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("OASIS2_Spark_ML")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    models = ["LogisticRegression", "RandomForest"]
    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=42
    )

    all_metrics = []
    all_predictions = []

    start = time.time()

    try:
        for model_name in models:
            print(f"\n========== {model_name} ==========")

            oof = np.zeros(len(df))
            fold_predictions = np.zeros(len(df))

            for fold, (train_idx, test_idx) in enumerate(
                cv.split(df[FEATURES], y), start=1
            ):
                train_pdf = df.iloc[train_idx][required].copy()
                test_pdf = df.iloc[test_idx][required].copy()

                train_sdf = spark.createDataFrame(train_pdf)
                test_sdf = spark.createDataFrame(test_pdf)

                train_sdf = add_class_weights(train_sdf)

                pipeline = build_pipeline(model_name)
                fitted = pipeline.fit(train_sdf)

                predictions = (
                    fitted.transform(test_sdf)
                    .select(
                        "subject_id",
                        TARGET,
                        "prediction",
                        vector_to_array("probability")[1].alias("score")
                    )
                    .toPandas()
                )

                for _, row in predictions.iterrows():
                    idx = df.index[df["subject_id"] == row["subject_id"]][0]
                    oof[idx] = float(row["score"])
                    fold_predictions[idx] = float(row["prediction"])

                print(
                    f"Fold {fold}/{args.folds} completed | "
                    f"train={len(train_idx)} | test={len(test_idx)}"
                )

            metrics = evaluate_predictions(
                y,
                fold_predictions,
                oof
            )

            metrics["model"] = model_name
            metrics["subjects"] = len(df)
            metrics["positive_cases"] = n_positive
            all_metrics.append(metrics)

            print("\nOut-of-fold evaluation:")
            for key, value in metrics.items():
                if isinstance(value, (float, np.floating)):
                    print(f"{key}: {value:.4f}")

            tn, fp, fn, tp = confusion_matrix(
                y, fold_predictions, labels=[0, 1]
            ).ravel()

            print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

            for i in range(len(df)):
                all_predictions.append({
                    "subject_id": df.iloc[i]["subject_id"],
                    "model": model_name,
                    "actual": int(y[i]),
                    "predicted": int(fold_predictions[i]),
                    "progression_probability": float(oof[i])
                })

            print("\nTraining final model on all subjects...")

            full_sdf = spark.createDataFrame(df[required])
            full_sdf = add_class_weights(full_sdf)

            final_model = build_pipeline(model_name).fit(full_sdf)

            model_dir = config.OUT_DIR / "spark_models"
            model_dir.mkdir(exist_ok=True)

            model_path = model_dir / (
                f"landmark{args.landmark}_{model_name}"
            )

            if model_path.exists():
                raise FileExistsError(
                    f"Model already exists: {model_path}. "
                    "Move or rename it before running again."
                )

            final_model.write().save(str(model_path))
            print(f"Saved model: {model_path}")

        metrics_path = config.OUT_DIR / (
            f"spark_ml_landmark{args.landmark}_metrics.csv"
        )
        predictions_path = config.OUT_DIR / (
            f"spark_ml_landmark{args.landmark}_predictions.csv"
        )

        pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
        pd.DataFrame(all_predictions).to_csv(
            predictions_path, index=False
        )

        print("\n========== FINAL RESULTS ==========")
        print(pd.DataFrame(all_metrics).round(4).to_string(index=False))
        print(f"\nMetrics saved: {metrics_path}")
        print(f"Predictions saved: {predictions_path}")
        print(f"Total time: {time.time() - start:.1f} seconds")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
