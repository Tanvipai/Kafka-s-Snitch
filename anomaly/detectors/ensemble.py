from pathlib import Path

import numpy as np
import pandas as pd

STAT_PATH = Path(__file__).parents[1] / "data" / "statistical_scores.csv"
IF_PATH = Path(__file__).parents[1] / "data" / "iforest_scores.csv"
OUTPUT_PATH = Path(__file__).parents[1] / "data" / "ensemble_scores.csv"

STAT_THRESHOLD = 90
ESCALATE_THRESHOLD = 99


class EnsembleScorer:

    def __init__(self, stat_threshold=STAT_THRESHOLD, escalate=ESCALATE_THRESHOLD):
        self.stat_threshold = stat_threshold
        self.escalate = escalate

    def combine(self, stat_df, if_df):
        stat = stat_df[["job_id", "workload", "job_date", "day_index", "label",
                        "is_anomaly", "suspicion", "scored", "top_feature",
                        "cusum_feature", "cusum_alarm"]]
        forest = if_df[["job_id", "if_score", "if_flag"]]
        df = stat.merge(forest, on="job_id", how="inner")

        stat_part = df.suspicion.fillna(0)
        both = (stat_part >= self.stat_threshold) & df.if_flag

        df["ensemble_flag"] = both | (stat_part >= self.escalate)
        df["confidence"] = np.where(
            both, "high",
            np.where(stat_part >= self.escalate, "medium", "none"),
        )
        df["ensemble_score"] = np.where(
            both, 100,
            np.where(stat_part >= self.escalate, 75, (0.5 * stat_part).round(1)),
        )
        df["agreement"] = np.where(
            both, "both",
            np.where(stat_part >= self.stat_threshold, "statistical",
                     np.where(df.if_flag, "isolation_forest", "neither")),
        )
        df.loc[~df.scored, ["ensemble_score", "ensemble_flag"]] = [np.nan, False]
        return df


def metrics(df, pred):
    truth = df.is_anomaly
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def compare(df):
    scored = df[df.scored]
    print("\ndetector comparison")
    print(f"{'detector':<22}{'tp':>5}{'fp':>6}{'fn':>5}{'precision':>11}{'recall':>9}{'f1':>7}")
    print("-" * 65)
    rows = {
        "statistical only": scored.suspicion >= 90,
        "isolation forest only": scored.if_flag,
        "ensemble": scored.ensemble_flag,
    }
    for name, pred in rows.items():
        tp, fp, fn, p, r, f1 = metrics(scored, pred)
        print(f"{name:<22}{tp:>5}{fp:>6}{fn:>5}{p:>11.3f}{r:>9.3f}{f1:>7.3f}")


def rule_sweep(df):
    scored = df[df.scored]
    stat = scored.suspicion.fillna(0)
    rules = {
        "statistical alone": stat >= 90,
        "agreement only": (stat >= 90) & scored.if_flag,
        "agreement or stat 99": ((stat >= 90) & scored.if_flag) | (stat >= 99),
        "either detector": (stat >= 90) | scored.if_flag,
    }
    print("\ncombination rules")
    print(f"{'rule':<24}{'tp':>5}{'fp':>6}{'fn':>5}{'precision':>11}{'recall':>9}{'f1':>7}")
    print("-" * 67)
    for name, pred in rules.items():
        tp, fp, fn, p, r, f1 = metrics(scored, pred)
        print(f"{name:<24}{tp:>5}{fp:>6}{fn:>5}{p:>11.3f}{r:>9.3f}{f1:>7.3f}")


def lead_time(df):
    print("\nnights until first detection, per attack window")
    print(f"{'workload':<22}{'attack':<16}{'stat':>6}{'IF':>5}{'ensemble':>10}{'window':>8}")
    print("-" * 68)
    attacks = df[df.is_anomaly & df.scored]
    for (workload, label), group in attacks.groupby(["workload", "label"]):
        group = group.sort_values("day_index").reset_index(drop=True)
        start = group.day_index.iloc[0]

        def first(mask):
            hits = group[mask]
            return int(hits.day_index.iloc[0] - start) + 1 if len(hits) else None

        s = first(group.suspicion >= 90)
        i = first(group.if_flag)
        e = first(group.ensemble_flag)
        fmt = lambda v: str(v) if v is not None else "-"
        print(f"{workload:<22}{label:<16}{fmt(s):>6}{fmt(i):>5}{fmt(e):>10}{len(group):>8}")


def agreement_table(df):
    scored = df[df.scored]
    print("\nwhat each detector contributes")
    print(f"{'agreement':<20}{'jobs':>7}{'true anomalies':>16}{'precision':>11}")
    print("-" * 54)
    for kind, group in scored.groupby("agreement"):
        n = len(group)
        t = int(group.is_anomaly.sum())
        print(f"{kind:<20}{n:>7}{t:>16}{t / n if n else 0:>11.3f}")


if __name__ == "__main__":
    stat_df = pd.read_csv(STAT_PATH)
    if_df = pd.read_csv(IF_PATH)

    scorer = EnsembleScorer()
    df = scorer.combine(stat_df, if_df)
    df.to_csv(OUTPUT_PATH, index=False)

    compare(df)
    agreement_table(df)
    rule_sweep(df)
    lead_time(df)

    print(f"\nwritten to {OUTPUT_PATH}")