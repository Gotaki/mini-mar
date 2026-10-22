# mini-mar: anomaly detection on simulated mobile-network telemetry

A small end-to-end project that detects customer-impacting incidents in mobile-network service metrics.
It simulates telemetry with known, labelled faults, builds features in **PySpark**, compares several
detectors, and scores them with **Precision / Recall / F1** and **time-to-detect**.

The aim is to show the full loop on a problem with real operational constraints: seasonality, cohorts of
similar services, subtle "slow burn" faults, alert fatigue and data-quality gaps.

> **Status:** simulator, PySpark features, evaluation framework and baseline detectors are complete.
> Isolation Forest, a validation split for threshold tuning, alarm correlation and a data-quality check are planned (see Roadmap).

## The problem

Network incidents are detected late when alarms only fire on large, obvious failures. Two cases are hard:

- **Slow burns**: a metric degrades gradually over days, so no single reading looks alarming.
- **Localised faults**: a handful of services are affected, hidden inside normal variation.

A good detector must catch these early without drowning engineers in false alarms.

## Data (simulated)

`src/simulate.py` generates 300 services in 4 cohorts (similar device/area groups) with six metrics every
15 minutes for 14 days (402,300 rows). Metrics follow a daily and weekly load cycle with service- and
cohort-specific baselines and multiplicative noise. Five incidents are injected, plus one data gap:

| ID | Type | Cohort | Services hit | Duration | Shape |
|---|---|---|---|---|---|
| 1 | Outage | B | 20 | 2h | Step: drop rate x6, throughput x0.3 |
| 2 | Slow burn | C | 30 | 96h | Ramp: drop rate up to x2.5, throughput down to x0.75 |
| 3 | Localised | D | 8 | 3h | Step: throughput x0.4, setup failures x3 |
| 4 | Outage | A | 25 | 2h | Step: drop and attach failures up, throughput x0.35 |
| 5 | Slow burn | A | 25 | 72h | Ramp: setup failures up to x3, MOS down 0.6 |
| gap | Missing data | A | all | 3h | Rows absent (a platform issue, **not** a network fault) |

Metrics: `drop_rate`, `setup_fail_rate`, `attach_fail_rate`, `throughput_mbps`, `volume_mb`, `mos`.
The seed is fixed, so results are reproducible.

## Pipeline

```
simulate.py -> telemetry.csv -> features.py (PySpark) -> features.parquet -> baselines.py -> results/*.csv
                                                                                 |
                                                                           evaluate.py (scoring)
```

### Features (`src/features.py`)

| Step | Operation | Reason |
|---|---|---|
| 1 | Log-transform rates, throughput, volume | Noise is multiplicative; makes spikes and dips symmetric |
| 2 | Subtract the cohort median at the same timestamp | Peers share the load cycle, so seasonality cancels |
| 3 | Robust z-score (median / MAD) per service: `zc_*` (vs cohort), `zs_*` (vs own hour-of-day history) | Median/MAD stop incidents from distorting their own baseline |
| 4 | Trailing 3h and 24h means of each z-score | Lets small persistent shifts (slow burns) accumulate; time-based windows tolerate missing rows |

### Evaluation (`src/evaluate.py`)

- **Precision** = TP / (TP + FP): of what was flagged, how much was real (low = alert fatigue).
- **Recall** = TP / (TP + FN): of real incidents, how many were caught (low = missed incidents).
- **F1** = 2PR / (P + R): high only when both are high.
- **Row level**: every service-interval scored against its label.
- **Event level**: flags roll up into cohort alarms (>= 5 flagged services in a cohort, runs within 1h merged); an alarm is true if it overlaps a real incident in the same cohort.
- **Time-to-detect**: hours from incident start until >= 5 affected services are flagged together.

## Results (baselines)

| Detector | Row P | Row R | Row F1 | Alarms | False alarms | Event P | Event R | Event F1 | Mean hrs to detect |
|---|---|---|---|---|---|---|---|---|---|
| static_threshold | 0.63 | 0.04 | 0.08 | 7 | 1 | 0.86 | 1.0 | 0.92 | 24.8 |
| seasonal_z>4 | 0.48 | 0.17 | 0.25 | 51 | 35 | 0.31 | 1.0 | 0.48 | 18.4 |
| cohort_z>4 | 0.97 | 0.14 | 0.24 | 7 | 0 | 1.00 | 1.0 | 1.00 | 23.3 |
| cohort_z_3h>2 | 0.91 | 0.42 | 0.57 | 9 | 5 | 0.44 | 1.0 | 0.62 | 17.7 |
| cohort_z_24h>1 | 0.62 | 0.51 | 0.56 | 7 | 4 | 0.43 | 0.8 | 0.56 | 19.8 |

Hours to detect per incident (blank = missed):

| Incident | static | seasonal_z>4 | cohort_z>4 | cohort_z_3h>2 | cohort_z_24h>1 |
|---|---|---|---|---|---|
| 1 outage | 0 | 0 | 0 | 0.2 | 1.0 |
| 2 slow burn (96h) | 85.0 | 72.0 | 77.2 | 56.0 | 47.5 |
| 3 localised | 0 | 0 | 0 | 0.8 | missed |
| 4 outage | 0 | 0 | 0 | 0 | 0 |
| 5 slow burn (72h) | 39.0 | 19.8 | 39.0 | 31.2 | 30.8 |

## Findings

- **Outages are easy** for every method (detected at hour 0), so they don't separate the detectors.
- **Slow burns are a speed problem, not a detection problem.** A 99.9th-percentile threshold does catch incident 2, but 85 hours into a 96-hour fault. The 24-hour cohort window cuts that to 47.5 hours.
- **Cohort-relative scoring beats own-history scoring.** `cohort_z>4` raised 0 false alarms; `seasonal_z>4` raised 35, because each service has only ~10 weekdays of history per hour-of-day slot.
- **Windows trade false alarms for speed.** The 24h window is fastest on slow burns but misses the 3-hour localised incident (diluted by the average) and keeps alarming after incidents end.
- **Row-level recall stays below 0.51 for all detectors.** The early part of a slow-burn ramp is statistically indistinguishable from noise.

## Limitations

- Data is simulated with clean, independent noise; real telemetry is messier and incidents overlap.
- Injected outages are much stronger than typical real faults, so they are too easy.
- Only 5 incidents, so event-level F1 moves in large steps; read it alongside the row-level numbers.
- Event-level matching is generous: any alarm inside a multi-day slow-burn window counts as a hit.
- Detector thresholds (4, 2, 1) are hand-picked, not tuned.
- Baselines are computed over the whole period including incidents; production would use a trailing or held-out window.
- No network topology (cells, sectors, sites); cohorts are a crude proxy for spatial structure.

## Roadmap

- [ ] Isolation Forest on cohort-relative features, compared against the baselines
- [ ] Time-based train/validation split to tune thresholds without leakage
- [ ] Alarm correlation: merge simultaneous anomalies into one alarm and report the alert reduction
- [ ] Data-quality check so missing-data gaps are not reported as network faults
- [ ] Stretch: LLM-generated plain-English alarm summaries

## Reproduce

Requires Python 3.10+ and Java 17+ (for PySpark).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/simulate.py     # -> data/telemetry.csv, incidents.csv, dq_gaps.csv
python src/features.py     # -> data/features.parquet (about 2 minutes)
python src/baselines.py    # -> results/baseline_summary.csv, baseline_per_incident.csv
```

Run all scripts from the repository root.

## Layout

```
src/simulate.py    telemetry + labelled incident simulator
src/features.py    PySpark feature pipeline
src/evaluate.py    row-level and event-level scoring, time-to-detect
src/baselines.py   static-threshold and z-score detectors
results/           saved metrics
data/              generated data (git-ignored)
```
