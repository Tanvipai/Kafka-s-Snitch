from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 30
CLEAN_DAYS = 120
CUSUM_K = 0.5         
CUSUM_H = 5.0          
Z_WARN = 3.0
Z_CRIT = 12.0
SUSPICION_THRESHOLD = 90  
MAD_FLOOR_FRAC = 0.5   

FEATURES = {
    "change_rate": "high",
    "bytes_total": "high",
    "avg_file_size": "high",
    "duration_seconds": "high",
    "files_scanned": "low",
    "file_size_stddev": "low",
    "start_hour": "abs",
}

CUSUM_FEATURES = ["change_rate", "avg_file_size", "start_hour", "file_size_stddev"]

DATA_PATH = Path(__file__).parents[1] / "data" / "backup_jobs.csv"
OUTPUT_PATH = Path(__file__).parents[1] / "data" / "statistical_scores.csv"


def _mad(values):
    med = np.median(values)
    return np.median(np.abs(values - med))


class StatisticalDetector:

    def __init__(self, window=WINDOW, clean_days=CLEAN_DAYS, k=CUSUM_K, h=CUSUM_H):
        self.window = window
        self.clean_days = clean_days
        self.k = k
        self.h = h
        self.frozen = {}
        self.mad_floor = {}

    def fit(self, df):
        clean = df[df.day_index < self.clean_days]
        if clean.empty:
            raise ValueError("no clean period in this data — cannot fit CUSUM baselines")

        for workload, group in clean.groupby("workload"):
            for feature in FEATURES:
                mu = float(group[feature].mean())
                sd = float(group[feature].std())
                self.frozen[(workload, feature)] = (mu, max(sd, abs(mu) * 1e-4, 1e-9))
                self.mad_floor[(workload, feature)] = max(_mad(group[feature].values) * MAD_FLOOR_FRAC, 1e-9)
        return self

    def _rolling_z(self, series, floor):
        med = series.rolling(self.window, min_periods=self.window).median().shift(1)
        mad = series.rolling(self.window, min_periods=self.window).apply(_mad, raw=True).shift(1)
        return 0.6745 * (series - med) / mad.clip(lower=floor)

    def _cusum(self, series, mu, sd, direction, active):
        s_hi = s_lo = 0.0
        out = []
        for value, live in zip(series, active):
            if not live:
                out.append(0.0)
                continue
            z = (value - mu) / sd
            s_hi = max(0.0, s_hi + z - self.k)
            s_lo = max(0.0, s_lo - z - self.k)

            if direction == "high":
                s = s_hi
            elif direction == "low":
                s = s_lo
            else:
                s = max(s_hi, s_lo)

            out.append(s)
            if s > self.h:
                s_hi = s_lo = 0.0
        return pd.Series(out, index=series.index)

    def score(self, df):
        df = df.sort_values(["workload", "day_index"]).reset_index(drop=True)
        pieces = []

        for workload, group in df.groupby("workload", sort=False):
            group = group.copy()
            z_active = group.day_index >= self.window
            c_active = group.day_index >= self.clean_days

            directional = {}
            cusums = {}

            for feature, direction in FEATURES.items():
                z = self._rolling_z(group[feature], self.mad_floor[(workload, feature)])
                if direction == "high":
                    z = z
                elif direction == "low":
                    z = -z
                else:
                    z = z.abs()
                directional[feature] = z.where(z_active)

                if feature in CUSUM_FEATURES:
                    mu, sd = self.frozen[(workload, feature)]
                    cusums[feature] = self._cusum(group[feature], mu, sd, direction, c_active)

            z_frame = pd.DataFrame(directional)
            c_frame = pd.DataFrame(cusums)

            group["max_z"] = z_frame.max(axis=1)
            group["top_feature"] = z_frame.fillna(-np.inf).idxmax(axis=1).where(z_frame.notna().any(axis=1))
            group["n_features_over"] = (z_frame > Z_WARN).sum(axis=1)
            group["max_cusum"] = c_frame.max(axis=1)
            group["cusum_feature"] = c_frame.idxmax(axis=1)
            group["cusum_alarm"] = group.max_cusum > self.h
            group["scored"] = z_active | c_active

            pieces.append(group)

        out = pd.concat(pieces).sort_values(["job_date", "workload"]).reset_index(drop=True)
        return self._suspicion(out)

    def _suspicion(self, df):
    
        z_part = ((df.max_z - Z_WARN) / (Z_CRIT - Z_WARN)).clip(0, 1).fillna(0)
        c_part = ((df.max_cusum - 0.6 * self.h) / (0.4 * self.h)).clip(0, 1).fillna(0)
        breadth = (df.n_features_over.fillna(0) / 3).clip(0, 1)

        raw = np.maximum(z_part, c_part) + 0.15 * breadth
        df["suspicion"] = (100 * raw.clip(0, 1)).round(1)
        df.loc[~df.scored, "suspicion"] = np.nan
        return df


def evaluate(df, threshold=SUSPICION_THRESHOLD):
    scored = df[df.scored].copy()
    scored["flagged"] = scored.suspicion >= threshold

    print(f"\nscored {len(scored)} of {len(df)} jobs "
          f"({len(df) - len(scored)} withheld — no baseline yet)")
    print(f"\nthreshold = {threshold}\n")
    print(f"{'label':<16} {'jobs':>5} {'flagged':>8} {'rate':>7}   {'median susp':>11}")
    print("-" * 54)

    for label, group in scored.groupby("label"):
        rate = group.flagged.mean()
        print(f"{label:<16} {len(group):>5} {int(group.flagged.sum()):>8} "
              f"{rate:>6.1%}   {group.suspicion.median():>11.1f}")

    truth = scored.is_anomaly
    pred = scored.flagged
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print(f"\ntp={tp}  fp={fp}  fn={fn}")
    print(f"precision {precision:.3f}   recall {recall:.3f}")


def lead_time(df):
    print("\nfirst detection inside each attack window:")
    attacks = df[df.is_anomaly & df.scored]
    for (workload, label), group in attacks.groupby(["workload", "label"]):
        group = group.sort_values("day_index")
        hits = group[group.suspicion >= SUSPICION_THRESHOLD]
        if hits.empty:
            print(f"  {workload:<20} {label:<14} MISSED ({len(group)} nights)")
            continue
        night = int(hits.iloc[0].day_index - group.iloc[0].day_index) + 1
        driver = hits.iloc[0].cusum_feature if hits.iloc[0].cusum_alarm else hits.iloc[0].top_feature
        how = "cusum" if hits.iloc[0].cusum_alarm else "z-score"
        print(f"  {workload:<20} {label:<14} night {night} of {len(group)}  "
              f"via {how} on {driver}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    detector = StatisticalDetector().fit(df)
    scored = detector.score(df)
    scored.to_csv(OUTPUT_PATH, index=False)

    evaluate(scored)
    lead_time(scored)
    print(f"\nwritten to {OUTPUT_PATH}")