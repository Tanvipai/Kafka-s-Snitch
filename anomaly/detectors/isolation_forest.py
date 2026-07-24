from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CLEAN_DAYS = 120
CONTAMINATION = 0.02
N_ESTIMATORS = 200
SEED = 42

FEATURES = [
    "change_rate",
    "size_cv",
    "duration_per_gb",
    "start_hour",
    "avg_file_size",
    "files_ratio",
]

DATA_PATH = Path(__file__).parents[1] / "data" / "backup_jobs.csv"
OUTPUT_PATH = Path(__file__).parents[1] / "data" / "iforest_scores.csv"


def engineer(df):
    df = df.copy()
    df["size_cv"] = df.file_size_stddev / df.avg_file_size.clip(lower=1)
    df["duration_per_gb"] = df.duration_seconds / (df.bytes_total / 1e9).clip(lower=1e-6)
    df = df.sort_values(["workload", "day_index"])
    baseline = df.groupby("workload").files_scanned.transform(
        lambda s: s.rolling(30, min_periods=5).median().shift(1).bfill()
    )
    df["files_ratio"] = df.files_scanned / baseline.clip(lower=1)
    return df.sort_index()


class IsolationForestDetector:

    def __init__(self, contamination=CONTAMINATION, n_estimators=N_ESTIMATORS, seed=SEED):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.seed = seed
        self.scalers = {}
        self.model = None

    def _fit_scalers(self, clean):
        for workload, group in clean.groupby("workload"):
            stats = {}
            for feature in FEATURES:
                med = float(group[feature].median())
                q1, q3 = group[feature].quantile([0.25, 0.75])
                iqr = float(q3 - q1)
                stats[feature] = (med, max(iqr, abs(med) * 1e-3, 1e-9))
            self.scalers[workload] = stats

    def _transform(self, df):
        out = np.empty((len(df), len(FEATURES)), dtype=float)
        for i, (_, row) in enumerate(df.iterrows()):
            stats = self.scalers[row.workload]
            for j, feature in enumerate(FEATURES):
                med, iqr = stats[feature]
                out[i, j] = (row[feature] - med) / iqr
        return out

    def fit(self, df):
        df = engineer(df)
        clean = df[df.day_index < CLEAN_DAYS]
        self._fit_scalers(clean)
        X = self._transform(clean)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.seed,
        ).fit(X)
        self._clean_scores = np.sort(-self.model.score_samples(X))
        return self

    def score(self, df):
        df = engineer(df)
        X = self._transform(df)
        anomaly = -self.model.score_samples(X)
        df["if_anomaly_raw"] = anomaly
        df["if_flag"] = self.model.predict(X) == -1
        ranks = np.searchsorted(self._clean_scores, anomaly, side="right")
        df["if_score"] = (100 * ranks / len(self._clean_scores)).clip(0, 100).round(1)
        return df


def evaluate(df):
    scored = df.copy()
    print(f"\nfit on {int((df.day_index < CLEAN_DAYS).sum())} clean jobs, "
          f"scored all {len(df)}")
    print(f"contamination = {CONTAMINATION}\n")
    print(f"{'label':<16} {'jobs':>5} {'flagged':>8} {'rate':>7}   {'median if_score':>15}")
    print("-" * 58)
    for label, group in scored.groupby("label"):
        print(f"{label:<16} {len(group):>5} {int(group.if_flag.sum()):>8} "
              f"{group.if_flag.mean():>6.1%}   {group.if_score.median():>15.1f}")

    truth, pred = scored.is_anomaly, scored.if_flag
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"\ntp={tp}  fp={fp}  fn={fn}")
    print(f"precision {prec:.3f}   recall {rec:.3f}")


def hypothesis_check(df):
    print("\n--- benign spike vs ransomware (both look identical on change_rate) ---")
    for label in ["benign_spike", "ransomware"]:
        g = df[df.label == label]
        print(f"  {label:<14} change_rate {g.change_rate.mean():.3f} | "
              f"size_cv {g.size_cv.mean():.3f} | if_score {g.if_score.mean():.1f} | "
              f"flagged {int(g.if_flag.sum())}/{len(g)}")


def drift_check(df):
    normal = df[(df.label == "normal") & (df.day_index >= CLEAN_DAYS)]
    early = normal[normal.day_index < 240]
    late = normal[normal.day_index >= 240]
    print("\n--- growth cost: false-positive rate on normal jobs, early vs late ---")
    print(f"  days 120-240: {early.if_flag.mean():.1%}")
    print(f"  days 240-365: {late.if_flag.mean():.1%}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    detector = IsolationForestDetector().fit(df)
    scored = detector.score(df)
    scored.to_csv(OUTPUT_PATH, index=False)

    evaluate(scored)
    hypothesis_check(scored)
    drift_check(scored)
    print(f"\nwritten to {OUTPUT_PATH}")