"""Baseline detectors: static thresholds and robust z-scores. Isolation Forest comes next."""
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate, load_truth

DATA, RESULTS = Path("data"), Path("results")
# direction of "bad" for each metric: +1 = higher is worse, -1 = lower is worse (volume is not scored)
BAD_SIGN = {"drop_rate": 1, "setup_fail_rate": 1, "attach_fail_rate": 1, "throughput_mbps": -1, "mos": -1}


def bad_z(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Worst directional z-score across metrics, e.g. prefix='zc24h' uses zc24h_drop_rate, ..."""
    return np.column_stack([s * df[f"{prefix}_{m}"].to_numpy() for m, s in BAD_SIGN.items()]).max(axis=1)


def static_threshold(df: pd.DataFrame, q: float = 0.999) -> np.ndarray:
    """What an analyst does with no modelling: one global limit per metric at the 99.9th percentile of all data."""
    flag = np.zeros(len(df), dtype=bool)
    for m, s in BAD_SIGN.items():
        lim = df[m].quantile(q if s > 0 else 1 - q)
        flag |= (df[m] > lim).to_numpy() if s > 0 else (df[m] < lim).to_numpy()
    return flag


DETECTORS = {
    "static_threshold": lambda d: static_threshold(d),
    "seasonal_z>4": lambda d: bad_z(d, "zs") > 4,
    "cohort_z>4": lambda d: bad_z(d, "zc") > 4,
    "cohort_z_3h>2": lambda d: bad_z(d, "zc3h") > 2,
    "cohort_z_24h>1": lambda d: bad_z(d, "zc24h") > 1,
}


def main() -> None:
    df = pd.read_parquet(DATA / "features.parquet")
    truth = load_truth()
    summaries, per_incident = [], []
    for name, fn in DETECTORS.items():
        d = df[["service_id", "cohort", "ts", "incident_id"]].copy()
        d["flag"] = fn(df)
        s, per = evaluate(name, d, truth)
        summaries.append(s)
        per_incident += per

    summary = pd.DataFrame(summaries).set_index("detector")
    per = pd.DataFrame(per_incident)
    pd.set_option("display.width", 200)
    print("\n=== Summary ===")
    print(summary.round(3).to_string())
    print("\n=== Per incident (hours_to_detect; NaN = missed) ===")
    print(per.pivot(index="incident_id", columns="detector", values="hours_to_detect").round(1).to_string())
    RESULTS.mkdir(exist_ok=True)
    summary.round(4).to_csv(RESULTS / "baseline_summary.csv")
    per.round(4).to_csv(RESULTS / "baseline_per_incident.csv", index=False)


if __name__ == "__main__":
    main()