"""Step 2b: same landmark dataset with PySpark (the 'Big Data' version, scales to many subjects).

Usage: python s2_features_spark.py --landmark 2 [--check]   (--check compares with the pandas result)
"""
import argparse
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window, functions as F
import config
from features import (LANDMARK_COLS, STATIC_COLS, TREND_VARS, load_visits, build_landmark_pandas)


def build_landmark_spark(spark, csv_path, L=2, max_cdr=0.5):
    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)
    w = Window.partitionBy("subject_id").orderBy("visit")
    df = (df.withColumn("rn", F.row_number().over(w))
            .withColumn("n", F.count("*").over(Window.partitionBy("subject_id")))
            .filter(F.col("n") >= L + 1))
    cur = df.filter(F.col("rn") == L)
    first = df.filter(F.col("rn") == 1).select(
        "subject_id", *[F.col(c).alias(f"f_{c}") for c in TREND_VARS + ["mr_delay_days", "group"]])
    fut = df.filter(F.col("rn") > L).groupBy("subject_id").agg(
        F.max("cdr").alias("fut_max_cdr"), F.max("mr_delay_days").alias("fut_delay"))
    j = (cur.join(first, "subject_id").join(fut, "subject_id")
            .filter(F.col("cdr").isNotNull() & (F.col("cdr") <= max_cdr) & F.col("fut_max_cdr").isNotNull()))
    years = (F.col("mr_delay_days") - F.col("f_mr_delay_days")) / 365.25
    cols = [F.col("subject_id"), F.lit(L).alias("landmark")]
    cols += [F.col(c).cast("double").alias(f"s_{c}") for c in STATIC_COLS]
    cols += [years.alias("l_years"), F.lit(L).alias("l_n_visits")]
    cols += [F.col(f"f_{c}").cast("double").alias(f"b_{c}") for c in TREND_VARS]
    for c in TREND_VARS:
        delta = F.col(c) - F.col(f"f_{c}")
        ok = (years > 0) if L >= 2 else F.lit(False)
        cols.append(F.when(ok, delta).cast("double").alias(f"l_{c}_delta"))
        cols.append(F.when(ok, delta / years).cast("double").alias(f"l_{c}_slope"))
    cols += [((F.col("fut_delay") - F.col("mr_delay_days")) / 365.25).alias("future_years"),
             (F.col("fut_max_cdr") > F.col("cdr")).cast("int").alias("progressed"),
             (F.col("f_group") == "Converted").cast("int").alias("group_converted")]
    return j.select(*cols).select(*LANDMARK_COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmark", type=int, default=2)
    ap.add_argument("--max-cdr", type=float, default=0.5)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    spark = (SparkSession.builder.master("local[2]").appName("oasis2_features")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    t = time.time()
    sdf = build_landmark_spark(spark, config.VISITS_CSV, a.landmark, a.max_cdr)
    out = sdf.toPandas().sort_values("subject_id").reset_index(drop=True)
    print(f"Spark: {len(out)} subjects in {time.time() - t:.1f}s")
    out.to_csv(config.OUT_DIR / f"landmark{a.landmark}_spark.csv", index=False)

    if a.check:
        ref = (build_landmark_pandas(load_visits(config.VISITS_CSV), a.landmark, a.max_cdr)
               .sort_values("subject_id").reset_index(drop=True))
        assert list(ref["subject_id"]) == list(out["subject_id"]), "different subjects selected"
        for c in LANDMARK_COLS[1:]:
            assert np.allclose(ref[c].astype(float), out[c].astype(float), equal_nan=True, atol=1e-6), c
        print("CHECK OK: Spark and pandas outputs are identical")
    spark.stop()


if __name__ == "__main__":
    main()
