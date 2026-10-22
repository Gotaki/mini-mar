"""PySpark feature pipeline: raw telemetry -> model-ready features.

Input : data/telemetry.csv
Output: data/features.parquet

Feature families (m = each of the 6 metrics):
  zc_<m>      robust z-score of the service's deviation from its COHORT median (removes load seasonality)
  zs_<m>      robust z-score vs the service's own hour-of-day / weekday-or-weekend baseline
  zc3h_/zc24h_/zs3h_/zs24h_<m>   trailing 3-hour and 24-hour means of the above (to catch slow burns)
"""
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

DATA = Path("data")
LOG_METRICS = ["drop_rate", "setup_fail_rate", "attach_fail_rate", "throughput_mbps", "volume_mb"]
METRICS = LOG_METRICS + ["mos"]
MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation for normal data
EPS = 1e-6


def median(col):
    return F.percentile_approx(col, 0.5, 10000)


def robust_z(df, mapping, keys):
    """For each {out_col: in_col}: out_col = (x - median) / (1.4826 * MAD), median/MAD computed per group `keys`.
    All metrics are handled in one aggregation per step so the Spark plan stays small."""
    ins = list(mapping.values())
    meds = df.groupBy(*keys).agg(*[median(c).alias(f"_med_{c}") for c in ins])
    df = df.join(F.broadcast(meds), keys)
    mads = df.groupBy(*keys).agg(*[median(F.abs(F.col(c) - F.col(f"_med_{c}"))).alias(f"_mad_{c}") for c in ins])
    df = df.join(F.broadcast(mads), keys)
    for out, c in mapping.items():
        df = df.withColumn(out, (F.col(c) - F.col(f"_med_{c}")) / (MAD_SCALE * F.greatest(F.col(f"_mad_{c}"), F.lit(EPS))))
    return df.drop(*[f"_med_{c}" for c in ins], *[f"_mad_{c}" for c in ins])


def main() -> None:
    spark = (SparkSession.builder.master("local[*]").appName("mini-mar-features")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.driver.memory", "2g")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(str(DATA / "telemetry.csv"), header=True, inferSchema=True)
    df = (df.withColumn("ts", F.col("ts").cast("timestamp"))
            .withColumn("ts_unix", F.col("ts").cast("long"))
            .withColumn("hour", F.hour("ts"))
            .withColumn("is_weekend", F.dayofweek("ts").isin(1, 7).cast("int")))

    # 1. Variance-stabilise: multiplicative metrics -> log scale; MOS stays linear
    for m in METRICS:
        df = df.withColumn(f"v_{m}", F.log(F.col(m)) if m in LOG_METRICS else F.col(m))

    # 2. Cohort-relative deviation: subtract the cohort's median at the same timestamp
    cohort_med = df.groupBy("cohort", "ts").agg(*[median(f"v_{m}").alias(f"cmed_{m}") for m in METRICS])
    df = df.join(F.broadcast(cohort_med), ["cohort", "ts"])
    for m in METRICS:
        df = df.withColumn(f"dev_{m}", F.col(f"v_{m}") - F.col(f"cmed_{m}"))
    df = df.cache()

    # 3. Robust z-scores
    df = robust_z(df, {f"zc_{m}": f"dev_{m}" for m in METRICS}, ["service_id"])
    df = robust_z(df, {f"zs_{m}": f"v_{m}" for m in METRICS}, ["service_id", "hour", "is_weekend"])
    df = df.cache()

    # 4. Trailing windows (time-based, so missing rows don't distort them)
    w3 = Window.partitionBy("service_id").orderBy("ts_unix").rangeBetween(-3 * 3600, 0)
    w24 = Window.partitionBy("service_id").orderBy("ts_unix").rangeBetween(-24 * 3600, 0)
    for m in METRICS:
        for prefix in ("zc", "zs"):
            df = (df.withColumn(f"{prefix}3h_{m}", F.avg(f"{prefix}_{m}").over(w3))
                    .withColumn(f"{prefix}24h_{m}", F.avg(f"{prefix}_{m}").over(w24)))

    feature_cols = [f"{p}{w}_{m}" for m in METRICS for p in ("zc", "zs") for w in ("", "3h", "24h")]
    out = df.select("service_id", "cohort", "ts", "incident_id", *METRICS, *feature_cols)
    out.coalesce(1).write.mode("overwrite").parquet(str(DATA / "features.parquet"))

    # Sanity check: features should be larger on incident rows than normal rows
    res = (out.withColumn("is_incident", (F.col("incident_id") > 0).cast("int"))
              .groupBy("is_incident")
              .agg(F.count("*").alias("rows"),
                   F.round(F.avg(F.abs("zc_drop_rate")), 2).alias("mean_abs_zc_drop"),
                   F.round(F.avg(F.abs("zs_drop_rate")), 2).alias("mean_abs_zs_drop"),
                   F.round(F.avg(F.abs("zc_throughput_mbps")), 2).alias("mean_abs_zc_tput"))
              .orderBy("is_incident"))
    print(f"\nfeature columns: {len(feature_cols)}")
    res.show()
    spark.stop()


if __name__ == "__main__":
    main()