from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = date(2025, 7, 1)
TOTAL_DAYS = 365
CLEAN_DAYS = 120

OUTPUT_PATH = Path(__file__).parent / "backup_jobs.csv"

WORKLOADS = [
    {"name": "finance-fileshare",  "files": 12000, "size": 180_000,   "cv": 0.9, "change": 0.06, "mbps": 220, "hour": 1.0,  "growth": 0.35},
    {"name": "hr-sharepoint",      "files": 4200,  "size": 340_000,   "cv": 1.1, "change": 0.04, "mbps": 180, "hour": 1.5,  "growth": 0.20},
    {"name": "eng-gitlab",         "files": 38000, "size": 22_000,    "cv": 1.6, "change": 0.14, "mbps": 300, "hour": 2.0,  "growth": 0.55},
    {"name": "sales-crm-db",       "files": 900,   "size": 4_800_000, "cv": 0.4, "change": 0.22, "mbps": 420, "hour": 0.5,  "growth": 0.30},
    {"name": "legal-archive",      "files": 26000, "size": 95_000,    "cv": 1.2, "change": 0.01, "mbps": 150, "hour": 3.0,  "growth": 0.10},
    {"name": "media-assets",       "files": 1600,  "size": 9_200_000, "cv": 0.7, "change": 0.08, "mbps": 500, "hour": 2.5,  "growth": 0.45},
    {"name": "email-exchange",     "files": 41000, "size": 65_000,    "cv": 1.4, "change": 0.11, "mbps": 260, "hour": 1.25, "growth": 0.25},
    {"name": "clinical-records",   "files": 7300,  "size": 520_000,   "cv": 0.8, "change": 0.05, "mbps": 190, "hour": 3.5,  "growth": 0.18},
    {"name": "devops-artifacts",   "files": 5100,  "size": 2_100_000, "cv": 1.0, "change": 0.31, "mbps": 460, "hour": 0.25, "growth": 0.60},
    {"name": "customer-analytics", "files": 2400,  "size": 1_450_000, "cv": 0.6, "change": 0.09, "mbps": 380, "hour": 4.0,  "growth": 0.40},
]

ANOMALY_WINDOWS = [
    ("finance-fileshare",  "ransomware",    204, 2),
    ("clinical-records",   "ransomware",    311, 3),
    ("legal-archive",      "insider_exfil", 158, 18),
    ("customer-analytics", "insider_exfil", 268, 21),
    ("media-assets",       "mass_deletion", 226, 1),
    ("eng-gitlab",         "benign_spike",  183, 1),
    ("hr-sharepoint",      "benign_spike",  247, 1),
]

TRUE_ANOMALIES = {"ransomware", "insider_exfil", "mass_deletion"}


class BackupMetadataGenerator:

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.windows = self._index_windows()

    def _index_windows(self):
        out = {}
        for name, label, start, length in ANOMALY_WINDOWS:
            out.setdefault(name, []).append((label, start, length))
        return out

    def _active_window(self, workload, day):
        for label, start, length in self.windows.get(workload, []):
            if start <= day < start + length:
                return label, day - start, length
        return None, 0, 0

    def _noise(self, sigma):
        return float(self.rng.lognormal(mean=0.0, sigma=sigma))

    def _one_job(self, wl, day):
        job_date = START_DATE + timedelta(days=day)
        label, offset, length = self._active_window(wl["name"], day)

        growth = 1.0 + wl["growth"] * (day / TOTAL_DAYS)

        weekend = job_date.weekday() >= 5
        seasonal = 0.55 if weekend else 1.0

        files_scanned = wl["files"] * growth * self._noise(0.05)
        change_rate = wl["change"] * seasonal * self._noise(0.28)
        avg_size = wl["size"] * self._noise(0.12)
        size_sd = avg_size * wl["cv"] * self._noise(0.10)
        hour = wl["hour"] + float(self.rng.normal(0, 0.06))
        status = "success"

        if label == "ransomware":
            change_rate = float(self.rng.uniform(0.92, 0.99))
            avg_size *= 1.05
            size_sd = avg_size * 0.02
            status = "warning"

        elif label == "insider_exfil":
            ramp = 1.0 + 0.05 * offset
            avg_size *= ramp
            hour += 1.6 * (offset / max(length - 1, 1))

        elif label == "mass_deletion":
            files_scanned *= 0.58
            change_rate = min(change_rate * 3.2, 0.85)
            avg_size *= 0.7
            status = "warning"

        elif label == "benign_spike":
            change_rate = float(self.rng.uniform(0.78, 0.88))

        elif self.rng.random() < 0.02:
            status = "warning"

        files_scanned = int(max(files_scanned, 1))
        change_rate = float(np.clip(change_rate, 0.0005, 1.0))
        files_changed = int(max(round(files_scanned * change_rate), 1))
        bytes_total = int(files_changed * avg_size)

        throughput = wl["mbps"] * 1_000_000 / 8.0
        duration = bytes_total / throughput * self._noise(0.09) + 45

        if label == "ransomware":
            duration *= 3.1
        elif label == "mass_deletion":
            duration *= 0.55

        return {
            "job_id": f"{wl['name']}-{job_date.isoformat()}",
            "workload": wl["name"],
            "job_date": job_date.isoformat(),
            "day_index": day,
            "start_hour": round(hour, 3),
            "files_scanned": files_scanned,
            "files_changed": files_changed,
            "bytes_total": bytes_total,
            "duration_seconds": round(duration, 1),
            "change_rate": round(files_changed / files_scanned, 5),
            "avg_file_size": round(bytes_total / files_changed, 1),
            "file_size_stddev": round(size_sd, 1),
            "status": status,
            "label": label or "normal",
            "is_anomaly": label in TRUE_ANOMALIES,
        }

    def generate(self):
        rows = []
        for wl in WORKLOADS:
            for day in range(TOTAL_DAYS):
                rows.append(self._one_job(wl, day))
        df = pd.DataFrame(rows)
        return df.sort_values(["job_date", "workload"]).reset_index(drop=True)


def summarise(df):
    print(f"\n{len(df)} jobs | {df['workload'].nunique()} workloads | "
          f"{df['job_date'].min()} to {df['job_date'].max()}")

    print("\nlabel breakdown:")
    for label, n in df["label"].value_counts().items():
        flag = "anomalous" if label in TRUE_ANOMALIES else "not anomalous"
        print(f"  {label:<16} {n:>5}   ({flag})")

    rate = df["is_anomaly"].mean()
    print(f"\ntrue contamination: {rate:.4f}  <- this is the Isolation Forest parameter on day 8")

    dirty = df[df["day_index"] < CLEAN_DAYS]["label"].unique()
    print(f"clean period (first {CLEAN_DAYS} days) labels: {list(dirty)}")

    print("\nchange_rate by label:")
    print(df.groupby("label")["change_rate"].agg(["mean", "max"]).round(3).to_string())


if __name__ == "__main__":
    gen = BackupMetadataGenerator()
    df = gen.generate()
    df.to_csv(OUTPUT_PATH, index=False)
    summarise(df)
    print(f"\nwritten to {OUTPUT_PATH}")