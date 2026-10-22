"""Evaluation framework shared by every detector.

A detector only has to produce a boolean `flag` per (service_id, ts) row. This module scores it two ways:

  Row level   : every service-interval is a prediction (positive = inside a labelled incident).
  Event level : flags are rolled up into cohort alarms (>= MIN_SERVICES flagged services in the same
                cohort at the same time, runs merged if <= 1h apart) and matched to the incident log.

KEY TERMS (used everywhere below)
  TP (true positive)  : detector flagged it AND a real incident was happening   -> a correct catch
  FP (false positive) : detector flagged it BUT nothing was wrong               -> a false alarm
  FN (false negative) : a real incident was happening BUT the detector stayed quiet -> a miss
  (TN, true negative, is "correctly stayed quiet"; there are so many that we don't use it)

  Precision (P) = TP / (TP + FP)
      "Of everything the detector flagged, how much was real?"
      Low precision = lots of false alarms = alert fatigue, engineers stop trusting the alarms.
  Recall (R)    = TP / (TP + FN)
      "Of all the real incidents, how much did the detector catch?"
      Low recall = incidents slip through unnoticed (the 'slow burn' problem).
  F1            = 2 * P * R / (P + R)
      The harmonic mean of P and R: one number that is only high when BOTH are high.
      Example: P=1.0, R=0.1 -> F1=0.18 (a detector that almost never fires looks 'precise' but is useless).
  Precision and recall trade off: a lower threshold raises recall but lowers precision, and vice versa.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
MIN_SERVICES = 5                    # flagged services needed to raise a cohort alarm / detect an incident
                                    # (one noisy service is not an incident; several in the same cohort is)
MERGE_GAP = 4                       # intervals (1h): alarm runs closer than this are merged into one event
GRACE = pd.Timedelta("2h")          # alarms up to 2h after an incident ends still count as detecting it


def load_truth() -> pd.DataFrame:
    """Load the ground-truth incident log written by simulate.py (which services, which window)."""
    inc = pd.read_csv(DATA / "incidents.csv", parse_dates=["start", "end"])
    inc["services"] = inc["services"].str.split()   # "S001 S002 ..." -> ["S001", "S002", ...]
    return inc


def _prf(tp: float, fp: float, fn: float):
    """Turn raw counts into (precision, recall, F1). The 'if' guards avoid dividing by zero
    (e.g. a detector that never fires has tp + fp = 0, so we report precision 0 rather than crash)."""
    p = tp / (tp + fp) if tp + fp else 0.0  #Recall
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _runs(active: pd.Series, max_gap: int):
    """Contiguous True runs of a time-indexed boolean series; runs <= max_gap intervals apart are merged.
    Used to turn 'cohort A had >= 5 flagged services at 10:00, 10:15, 10:30...' into ONE alarm event
    with a start and end time, instead of one alarm per 15-minute interval."""
    idx = np.flatnonzero(active.to_numpy())          # positions where the alarm condition is True
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > max_gap:                   # gap too large -> close the current run, start a new one
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(active.index[a], active.index[b]) for a, b in runs]   # convert positions back to timestamps


def evaluate(name: str, df: pd.DataFrame, truth: pd.DataFrame):
    """df needs columns: service_id, cohort, ts, incident_id, flag (bool). Returns (summary dict, per-incident rows)."""
    # ---- 1. ROW LEVEL: score every single (service, 15-min) row ----
    y = df["incident_id"].to_numpy() > 0            # truth: True if this row is inside a real incident
    p = df["flag"].to_numpy().astype(bool)          # prediction: True if the detector flagged this row
    # tp = flagged & real | fp = flagged & not real | fn = real & not flagged
    row_p, row_r, row_f = _prf((y & p).sum(), (~y & p).sum(), (y & ~p).sum())
    # NOTE: row-level recall is harsh on slow burns: the first hours of the ramp are
    # statistically indistinguishable from noise, so those rows are 'missed' by any sane detector.

    # ---- 2. PER INCIDENT: was each incident detected, and how quickly? ----
    per = []
    for _, inc in truth.iterrows():
        # Look only at the services hit by this incident, from its start to its end (+ grace period)
        sub = df[df["service_id"].isin(inc["services"]) & (df["ts"] >= inc["start"]) & (df["ts"] <= inc["end"] + GRACE)]
        flagged = sub[sub["flag"]]
        per_ts = flagged.groupby("ts")["service_id"].nunique()   # how many hit services were flagged at each time
        hit = per_ts[per_ts >= MIN_SERVICES].index               # times when enough of them were flagged together
        rows = df["incident_id"] == inc["incident_id"]           # all rows belonging to this incident
        per.append(dict(detector=name, incident_id=inc["incident_id"], type=inc["type"], cohort=inc["cohort"],
                        detected=len(hit) > 0,
                        # time-to-detect: the 'Mean Time To Identify (MTTI)' idea. Earlier = better.
                        hours_to_detect=(hit.min() - inc["start"]).total_seconds() / 3600 if len(hit) else np.nan,
                        # fraction of this incident's rows that were flagged (row-level recall for one incident)
                        row_recall=(df.loc[rows, "flag"].mean() if rows.any() else np.nan)))

    # ---- 3. EVENT LEVEL: roll flags up into cohort alarms, like a real alerting system would ----
    all_ts = pd.DatetimeIndex(sorted(df["ts"].unique()))
    cohorts = sorted(df["cohort"].unique())
    # Table: rows = timestamps, columns = cohorts, value = number of services flagged in that cohort then
    counts = (df[df["flag"]].groupby(["cohort", "ts"])["service_id"].nunique().unstack("cohort")
              .reindex(index=all_ts, columns=cohorts).fillna(0))
    n_alarms = tp_alarms = 0
    for c in cohorts:
        for s, e in _runs(counts[c] >= MIN_SERVICES, MERGE_GAP):    # each run of 'enough services flagged' = 1 alarm
            n_alarms += 1
            # An alarm is a TRUE alarm if it overlaps (in time, same cohort) with a real incident; otherwise FALSE alarm
            hit = ((truth["cohort"] == c) & (truth["start"] <= e) & (truth["end"] + GRACE >= s)).any()
            tp_alarms += int(hit)
    detected = sum(r["detected"] for r in per)
    ev_p = tp_alarms / n_alarms if n_alarms else 0.0     # event precision: true alarms / all alarms raised
    ev_r = detected / len(truth)                         # event recall: incidents detected / all incidents
    ev_f = 2 * ev_p * ev_r / (ev_p + ev_r) if ev_p + ev_r else 0.0

    summary = dict(detector=name, row_precision=row_p, row_recall=row_r, row_f1=row_f,
                   alarms=n_alarms, false_alarms=n_alarms - tp_alarms,
                   event_precision=ev_p, event_recall=ev_r, event_f1=ev_f,
                   mean_hours_to_detect=np.nanmean([r["hours_to_detect"] for r in per]) if detected else np.nan)
    return summary, per