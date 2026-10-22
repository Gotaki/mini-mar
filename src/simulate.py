"""Simulate 15-minute mobile-network service telemetry with labelled incidents.

Outputs (in data/):
  telemetry.csv  one row per service per 15-min interval (rows missing during data-quality gaps)
  incidents.csv  ground-truth incident log
  dq_gaps.csv    ground-truth data-quality gaps (NOT network incidents)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
COHORTS = ["A", "B", "C", "D"]
N_PER_COHORT = 75
DAYS = 14
START = "2026-10-01"  # a Thursday
INTERVALS_PER_DAY = 96
OUT = Path("data")

COHORT_FACTOR = {"A": 1.0, "B": 1.15, "C": 0.9, "D": 1.3}

# metric: (base, load_exponent, noise_sd)  -> multiplicative log-normal noise
RATE_METRICS = {
    "drop_rate": (0.8, 0.5, 0.25),
    "setup_fail_rate": (0.5, 0.5, 0.30),
    "attach_fail_rate": (0.3, 0.3, 0.30),
}


def slot(day: int, hour: int, minute: int = 0) -> int:
    return day * INTERVALS_PER_DAY + hour * 4 + minute // 15


# (type, cohort, n_services, start_slot, end_slot, ramp, effects)
# effects: metric -> ("mul", factor) or ("add", delta) at full intensity
INCIDENTS = [
    ("outage", "B", 20, slot(3, 10), slot(3, 12), "step",
     {"drop_rate": ("mul", 6), "setup_fail_rate": ("mul", 5),
      "throughput_mbps": ("mul", 0.3), "mos": ("add", -1.5)}),
    ("slow_burn", "C", 30, slot(5, 0), slot(9, 0), "ramp",
     {"drop_rate": ("mul", 2.5), "throughput_mbps": ("mul", 0.75), "mos": ("add", -0.5)}),
    ("localised", "D", 8, slot(8, 18), slot(8, 21), "step",
     {"throughput_mbps": ("mul", 0.4), "setup_fail_rate": ("mul", 3)}),
    ("outage", "A", 25, slot(11, 3), slot(11, 5), "step",
     {"drop_rate": ("mul", 5), "attach_fail_rate": ("mul", 6),
      "throughput_mbps": ("mul", 0.35), "mos": ("add", -1.2)}),
    ("slow_burn", "A", 25, slot(9, 12), slot(12, 12), "ramp",
     {"setup_fail_rate": ("mul", 3), "mos": ("add", -0.6)}),
]

# Data-quality gaps: (cohort, start_slot, end_slot) -> rows simply missing (platform issue)
DQ_GAPS = [("A", slot(6, 6), slot(6, 9))]


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)

    n_services = len(COHORTS) * N_PER_COHORT
    T = DAYS * INTERVALS_PER_DAY
    ts = pd.date_range(START, periods=T, freq="15min")
    hours = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    load = 1 + 0.45 * np.sin(2 * np.pi * (hours - 9) / 24)  # peaks ~15:00
    load = np.where(ts.dayofweek.to_numpy() >= 5, load * 0.85, load)  # quieter weekends

    cohort_of = np.repeat(COHORTS, N_PER_COHORT)
    cf = np.array([COHORT_FACTOR[c] for c in cohort_of])[:, None]
    svc_scale = rng.lognormal(0, 0.15, (n_services, 1))

    def noise(sd):
        return np.exp(rng.normal(0, sd, (n_services, T)))

    M = {}
    for name, (base, exp, sd) in RATE_METRICS.items():
        M[name] = base * cf * svc_scale * (load[None, :] ** exp) * noise(sd)
    M["throughput_mbps"] = 40 / cf / svc_scale * (load[None, :] ** -0.5) * noise(0.15)
    M["volume_mb"] = 120 * svc_scale * (load[None, :] ** 1.0) * noise(0.30)
    M["mos"] = 4.1 - 0.3 * (load[None, :] - 1) - 0.1 * (cf - 1) + rng.normal(0, 0.08, (n_services, T))

    incident_id = np.zeros((n_services, T), dtype=int)
    log = []
    for i, (kind, cohort, n, a, b, ramp, effects) in enumerate(INCIDENTS, start=1):
        pool = np.where(cohort_of == cohort)[0]
        svc = rng.choice(pool, size=n, replace=False)
        r = np.ones(b - a) if ramp == "step" else np.linspace(0, 1, b - a)
        for metric, (op, val) in effects.items():
            if op == "mul":
                M[metric][svc, a:b] *= 1 + (val - 1) * r
            else:
                M[metric][svc, a:b] += val * r
        incident_id[svc, a:b] = i
        log.append(dict(incident_id=i, type=kind, cohort=cohort, n_services=n,
                        start=ts[a], end=ts[b - 1], services=" ".join(f"S{s:03d}" for s in sorted(svc))))

    M["mos"] = np.clip(M["mos"], 1.0, 4.5)

    df = pd.DataFrame({
        "service_id": np.repeat([f"S{s:03d}" for s in range(n_services)], T),
        "cohort": np.repeat(cohort_of, T),
        "ts": np.tile(ts, n_services),
        **{k: v.ravel() for k, v in M.items()},
        "incident_id": incident_id.ravel(),
    })

    gaps = []
    for cohort, a, b in DQ_GAPS:
        mask = (df["cohort"] == cohort) & (df["ts"] >= ts[a]) & (df["ts"] <= ts[b - 1])
        df = df[~mask]
        gaps.append(dict(cohort=cohort, start=ts[a], end=ts[b - 1]))

    df.to_csv(OUT / "telemetry.csv", index=False, float_format="%.4f")
    pd.DataFrame(log).to_csv(OUT / "incidents.csv", index=False)
    pd.DataFrame(gaps).to_csv(OUT / "dq_gaps.csv", index=False)

    print(f"telemetry rows: {len(df):,}  services: {n_services}  intervals: {T}")
    print(f"incident-labelled rows: {(df['incident_id'] > 0).sum():,} ({(df['incident_id'] > 0).mean():.2%})")
    print(pd.DataFrame(log).drop(columns="services").to_string(index=False))


if __name__ == "__main__":
    main()