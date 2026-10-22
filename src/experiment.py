"""Step 5: tune every detector on DEV cohorts, then report on unseen TEST cohorts. Adds Isolation Forest.

Split (by cohort, so no incident straddles the boundary and test services are never seen during tuning):
  DEV  = cohorts B, C  -> incident 1 (outage) and incident 2 (slow burn)
  TEST = cohorts A, D  -> incident 3 (localised), 4 (outage) and 5 (slow burn); also contains the data-quality gap

Every detector is turned into a continuous anomaly SCORE per row (higher = more anomalous). A row is flagged when
score > threshold. Thresholds are chosen on DEV only, using this rule (an 'alert fatigue' constraint):
  fewest false alarms first, then the highest row-level F1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from baselines import BAD_SIGN, bad_z
from evaluate import evaluate, load_truth

DATA, RESULTS = Path("data"), Path("results")
DEV_COHORTS, TEST_COHORTS = ["B", "C"], ["A", "D"]
Q_GRID = [0.99, 0.995, 0.999, 0.9995, 0.9999]      # quantile grid for score-based detectors


# ---------- scorers: each factory is fitted on DEV only and returns a function df -> score array ----------
def ecdf_scorer(dev: pd.DataFrame):
    """Static-threshold detector as a score: for each metric, how extreme is this value (in the bad direction)
    relative to the DEV distribution? Score = the most extreme metric. Threshold q == 'beyond the q-th percentile'."""
    ref = {m: np.sort(dev[m].to_numpy()) for m in BAD_SIGN}

    def score(df):
        ranks = []
        for m, s in BAD_SIGN.items():
            r = np.searchsorted(ref[m], df[m].to_numpy(), side="right") / len(ref[m])
            ranks.append(r if s > 0 else 1 - r)          # lower-is-worse metrics: small value -> high score
        return np.max(ranks, axis=0)
    return score


def z_scorer(prefix: str):
    """Robust z-score detector (no fitting needed: z-scores are already standardised)."""
    return lambda dev: (lambda df: bad_z(df, prefix))


def iforest_features(df: pd.DataFrame) -> np.ndarray:
    """15 features: bad-direction cohort z-scores (instant, 3h, 24h) for the 5 metrics, floored at 0 so that
    'better than normal' counts as normal. Isolation Forest is direction-blind, so we remove the direction ourselves."""
    return np.column_stack([np.clip(s * df[f"{p}_{m}"].to_numpy(), 0, None)
                            for p in ("zc", "zc3h", "zc24h") for m, s in BAD_SIGN.items()])


def iforest_scorer(dev: pd.DataFrame):
    """Unsupervised multivariate detector: points that are easy to isolate with random splits get high scores."""
    model = IsolationForest(n_estimators=200, max_samples=256, random_state=42, n_jobs=-1)
    model.fit(iforest_features(dev))                      # fitted WITHOUT labels, on DEV rows only
    return lambda df: -model.score_samples(iforest_features(df))


# name -> (scorer factory, threshold grid builder). Grid builders receive the DEV scores.
quantile_grid = lambda sc: [(f"q{q}", float(np.quantile(sc, q))) for q in Q_GRID]
fixed_grid = lambda vals: (lambda sc: [(f"{v}", float(v)) for v in vals])
SPECS = {
    "static_threshold": (ecdf_scorer, lambda sc: [(f"q{q}", q) for q in Q_GRID]),
    "seasonal_z": (z_scorer("zs"), fixed_grid([3, 4, 5, 6, 8])),
    "cohort_z": (z_scorer("zc"), fixed_grid([3, 4, 5, 6, 8])),
    "cohort_z_3h": (z_scorer("zc3h"), fixed_grid([1, 1.5, 2, 3, 4])),
    "cohort_z_24h": (z_scorer("zc24h"), fixed_grid([0.5, 1, 1.5, 2, 3])),
    "isolation_forest": (iforest_scorer, quantile_grid),
}


def main() -> None:
    df = pd.read_parquet(DATA / "features.parquet")
    truth = load_truth()
    dev = df[df["cohort"].isin(DEV_COHORTS)].reset_index(drop=True)
    test = df[df["cohort"].isin(TEST_COHORTS)].reset_index(drop=True)
    truth_dev, truth_test = truth[truth["cohort"].isin(DEV_COHORTS)], truth[truth["cohort"].isin(TEST_COHORTS)]
    base_cols = ["service_id", "cohort", "ts", "incident_id"]
    d_dev, d_test = dev[base_cols].copy(), test[base_cols].copy()

    grid_rows, chosen, dev_sum, test_sum, dev_per, test_per = [], {}, [], [], [], []
    for name, (factory, grid_fn) in SPECS.items():
        scorer = factory(dev)
        s_dev, s_test = scorer(dev), scorer(test)
        cands = []
        for label, thr in grid_fn(s_dev):
            d_dev["flag"] = s_dev > thr
            summ, _ = evaluate(name, d_dev, truth_dev)
            cands.append((label, thr, summ))
            grid_rows.append(dict(detector=name, threshold=label, **{k: v for k, v in summ.items() if k != "detector"}))
        label, thr, _ = min(cands, key=lambda c: (c[2]["false_alarms"], -c[2]["row_f1"]))   # selection rule
        chosen[name] = label
        d_dev["flag"] = s_dev > thr
        d_test["flag"] = s_test > thr
        a, b = evaluate(name, d_dev, truth_dev)
        dev_sum.append(a); dev_per += b
        a, b = evaluate(name, d_test, truth_test)
        test_sum.append(a); test_per += b
        print(f"done: {name} (chosen threshold {label})", flush=True)

    pd.set_option("display.width", 220)
    dev_df = pd.DataFrame(dev_sum).set_index("detector")
    test_df = pd.DataFrame(test_sum).set_index("detector")
    dev_df.insert(0, "threshold", pd.Series(chosen))
    test_df.insert(0, "threshold", pd.Series(chosen))
    print("\n=== DEV cohorts (B, C): thresholds were chosen here ===")
    print(dev_df.round(3).to_string())
    print("\n=== TEST cohorts (A, D): unseen, honest numbers ===")
    print(test_df.round(3).to_string())
    for title, per in (("DEV", dev_per), ("TEST", test_per)):
        print(f"\n=== Hours to detect, {title} incidents (NaN = missed) ===")
        print(pd.DataFrame(per).pivot(index="incident_id", columns="detector", values="hours_to_detect").round(1).to_string())

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(grid_rows).round(4).to_csv(RESULTS / "tuning_grid.csv", index=False)
    dev_df.round(4).to_csv(RESULTS / "tuned_dev_summary.csv")
    test_df.round(4).to_csv(RESULTS / "tuned_test_summary.csv")
    pd.DataFrame(test_per).round(4).to_csv(RESULTS / "tuned_test_per_incident.csv", index=False)


if __name__ == "__main__":
    main()