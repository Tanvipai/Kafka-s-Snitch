import numpy as np
import pandas as pd

from anomaly.detectors.statistical import StatisticalDetector, FEATURES, _mad
from anomaly.detectors.ensemble import EnsembleScorer


def flat_frame(days=60, workload="wl-a", **overrides):
    base = {
        "change_rate": 0.10,
        "bytes_total": 1_000_000_000.0,
        "avg_file_size": 100_000.0,
        "duration_seconds": 300.0,
        "files_scanned": 10_000.0,
        "file_size_stddev": 50_000.0,
        "start_hour": 2.0,
    }
    rows = []
    for day in range(days):
        row = {
            "workload": workload,
            "day_index": day,
            "job_id": f"{workload}-{day}",
            "job_date": str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)),
        }
        row.update(base)
        rows.append(row)
    df = pd.DataFrame(rows)
    for feature, values in overrides.items():
        for day, value in values.items():
            df.loc[df.day_index == day, feature] = value
    return df


def jitter(df, seed=7, scale=0.02):
    rng = np.random.default_rng(seed)
    for feature in FEATURES:
        df[feature] = df[feature] * (1 + rng.normal(0, scale, len(df)))
    return df


def test_today_does_not_seed_its_own_baseline():
    detector = StatisticalDetector(window=30, clean_days=30)
    low = jitter(flat_frame(days=45), seed=1)
    high = low.copy()
    low.loc[low.day_index == 40, "change_rate"] = 0.20
    high.loc[high.day_index == 40, "change_rate"] = 0.90

    floor = 1e-9
    z_low = detector._rolling_z(low.change_rate, floor).iloc[40]
    z_high = detector._rolling_z(high.change_rate, floor).iloc[40]

    implied_median = (0.90 * z_low - 0.20 * z_high) / (z_low - z_high)
    implied_mad = 0.6745 * (0.20 - implied_median) / z_low

    excludes_today = low.change_rate.iloc[10:40]
    includes_today = low.change_rate.iloc[11:41]

    assert abs(implied_median - excludes_today.median()) < 1e-9
    assert abs(implied_mad - _mad(excludes_today.values)) < 1e-9
    assert abs(implied_mad - _mad(includes_today.values)) > 1e-6


def test_cusum_resets_after_alarm():
    detector = StatisticalDetector(k=0.5, h=5.0)
    drifting = pd.Series([2.0] * 20)
    active = [True] * 20
    sums = detector._cusum(drifting, mu=0.0, sd=1.0, direction="high", active=active)

    alarms = [i for i, value in enumerate(sums) if value > 5.0]
    assert alarms
    first = alarms[0]
    assert sums.iloc[first + 1] < sums.iloc[first]
    assert sums.iloc[first + 1] <= 1.6


def test_cold_start_abstains_instead_of_scoring_zero():
    detector = StatisticalDetector(window=30, clean_days=30)
    df = jitter(flat_frame(days=50), seed=3)
    scored = detector.fit(df).score(df)

    early = scored[scored.day_index < 30]
    late = scored[scored.day_index >= 30]

    assert early.suspicion.isna().all()
    assert not late.suspicion.isna().any()
    assert (early.suspicion == 0).sum() == 0


def test_ensemble_requires_agreement_or_extreme_statistic():
    stat = pd.DataFrame({
        "job_id": ["a", "b", "c", "d"],
        "workload": ["wl"] * 4,
        "job_date": ["2026-01-01"] * 4,
        "day_index": [200, 201, 202, 203],
        "label": ["normal"] * 4,
        "is_anomaly": [False] * 4,
        "suspicion": [95.0, 95.0, 99.5, 40.0],
        "scored": [True] * 4,
        "top_feature": ["change_rate"] * 4,
        "cusum_feature": ["change_rate"] * 4,
        "cusum_alarm": [False] * 4,
    })
    forest = pd.DataFrame({
        "job_id": ["a", "b", "c", "d"],
        "if_score": [99.0, 20.0, 20.0, 99.0],
        "if_flag": [True, False, False, True],
    })

    result = EnsembleScorer().combine(stat, forest).set_index("job_id")

    assert result.loc["a", "confidence"] == "high"
    assert result.loc["b", "confidence"] == "none"
    assert result.loc["c", "confidence"] == "medium"
    assert result.loc["d", "confidence"] == "none"
    assert bool(result.loc["a", "ensemble_flag"])
    assert bool(result.loc["c", "ensemble_flag"])
    assert not bool(result.loc["b", "ensemble_flag"])
    assert not bool(result.loc["d", "ensemble_flag"])