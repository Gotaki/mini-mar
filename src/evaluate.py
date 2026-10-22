"""Evaluation framework shared by every detector.

A detector only has to produce a boolean `flag` per (service_id, ts) row. This module scores it two ways:

  Row level   : every service-interval is a prediction (positive = inside a labelled incident).
  Event level : flags are rolled up into cohort alarms (>= MIN_SERVICES flagged services in the same
                cohort at the same time, runs merged if <= 1h apart) and matched to the incident log.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
MIN_SERVICES = 5                    # flagged services needed to raise a cohort alarm / detect an incident
MERGE_GAP = 4                       # intervals (1h): alarm runs closer than this are merged into one event
GRACE = pd.Timedelta("2h")          # alarms up to 2h after an incident ends still count as detecting it


def load_truth() -> pd.DataFrame:
    inc = pd.read_csv(DATA / "incidents.csv", parse_dates=["start", "end"])
    inc["services"] = inc["services"].str.split()
    return inc


def _prf(tp: float, fp: float, fn: float):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _runs(active: pd.Series, max_gap: int):
    """Contiguous True runs of a time-indexed boolean series; runs <= max_gap intervals apart are merged."""
    idx = np.flatnonzero(active.to_numpy())
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > max_gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(active.index[a], active.index[b]) for a, b in runs]


def evaluate(name: str, df: pd.DataFrame, truth: pd.DataFrame):
    """df needs columns: service_id, cohort, ts, incident_id, flag (bool). Returns (summary dict, per-incident rows)."""
    # ---- row level ----
    y = df["incident_id"].to_numpy() > 0
    p = df["flag"].to_numpy().astype(bool)
    row_p, row_r, row_f = _prf((y & p).sum(), (~y & p).sum(), (y & ~p).sum())

    # ---- per incident: detected? how fast? ----
    per = []
    for _, inc in truth.iterrows():
        sub = df[df["service_id"].isin(inc["services"]) & (df["ts"] >= inc["start"]) & (df["ts"] <= inc["end"] + GRACE)]
        flagged = sub[sub["flag"]]
        per_ts = flagged.groupby("ts")["service_id"].nunique()
        hit = per_ts[per_ts >= MIN_SERVICES].index
        rows = df["incident_id"] == inc["incident_id"]
        per.append(dict(detector=name, incident_id=inc["incident_id"], type=inc["type"], cohort=inc["cohort"],
                        detected=len(hit) > 0,
                        hours_to_detect=(hit.min() - inc["start"]).total_seconds() / 3600 if len(hit) else np.nan,
                        row_recall=(df.loc[rows, "flag"].mean() if rows.any() else np.nan)))

    # ---- event level: cohort alarms ----
    all_ts = pd.DatetimeIndex(sorted(df["ts"].unique()))
    cohorts = sorted(df["cohort"].unique())
    counts = (df[df["flag"]].groupby(["cohort", "ts"])["service_id"].nunique().unstack("cohort")
              .reindex(index=all_ts, columns=cohorts).fillna(0))
    n_alarms = tp_alarms = 0
    for c in cohorts:
        for s, e in _runs(counts[c] >= MIN_SERVICES, MERGE_GAP):
            n_alarms += 1
            hit = ((truth["cohort"] == c) & (truth["start"] <= e) & (truth["end"] + GRACE >= s)).any()
            tp_alarms += int(hit)
    detected = sum(r["detected"] for r in per)
    ev_p = tp_alarms / n_alarms if n_alarms else 0.0
    ev_r = detected / len(truth)
    ev_f = 2 * ev_p * ev_r / (ev_p + ev_r) if ev_p + ev_r else 0.0

    summary = dict(detector=name, row_precision=row_p, row_recall=row_r, row_f1=row_f,
                   alarms=n_alarms, false_alarms=n_alarms - tp_alarms,
                   event_precision=ev_p, event_recall=ev_r, event_f1=ev_f,
                   mean_hours_to_detect=np.nanmean([r["hours_to_detect"] for r in per]) if detected else np.nan)
    return summary, per