import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from isolation_forest import (
    DATA_PATH,
    FEATURES,
    IsolationForestDetector,
    engineer,
)

PLOT_DIR = Path(__file__).parents[1] / "data" / "shap_plots"
OUTPUT_PATH = Path(__file__).parents[1] / "data" / "shap_explanations.json"
MAX_PLOTS = 12
TOP_N = 3


class SHAPExplainer:

    def __init__(self, detector, top_n=TOP_N):
        self.detector = detector
        self.top_n = top_n
        self.explainer = shap.TreeExplainer(detector.model)
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    def _contributions(self, X):
        raw = np.array(self.explainer.shap_values(X))
        return -raw

    def explain(self, df):
        df = engineer(df)
        X = self.detector._transform(df)
        contrib = self._contributions(X)

        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            pairs = sorted(
                zip(FEATURES, contrib[i], X[i]),
                key=lambda p: p[1],
                reverse=True,
            )
            top = [
                {
                    "feature": name,
                    "contribution": round(float(value), 4),
                    "scaled_value": round(float(scaled), 3),
                }
                for name, value, scaled in pairs[: self.top_n]
            ]
            records.append(
                {
                    "job_id": row.job_id,
                    "workload": row.workload,
                    "job_date": row.job_date,
                    "label": row.label,
                    "total_contribution": round(float(contrib[i].sum()), 4),
                    "top_features": top,
                }
            )

        self._contrib_matrix = contrib
        self._X = X
        return records

    def narrative(self, record, flagged=True):
        if not flagged or record["total_contribution"] <= 0:
            return "not flagged — no feature pushed this job toward anomalous"
        parts = []
        for item in record["top_features"]:
            if item["contribution"] <= 0:
                continue
            direction = "above" if item["scaled_value"] > 0 else "below"
            parts.append(
                f"{item['feature']} {direction} baseline "
                f"(+{item['contribution']:.2f})"
            )
        if not parts:
            return "no feature pushed this job toward anomalous"
        return "flagged mainly by " + ", ".join(parts)

    def waterfall(self, index, job_id, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        explanation = shap.Explanation(
            values=self._contrib_matrix[index],
            base_values=-self.base_value,
            data=self._X[index],
            feature_names=FEATURES,
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
        path = out_dir / f"{job_id}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close("all")
        return path


def feature_importance(explainer, df, records):
    contrib = explainer._contrib_matrix
    print("\nmean contribution by feature and label")
    print(f"{'label':<16} " + " ".join(f"{f[:13]:>14}" for f in FEATURES))
    print("-" * (16 + 15 * len(FEATURES)))
    for label in ["normal", "ransomware", "insider_exfil", "mass_deletion", "benign_spike"]:
        idx = df.index[df.label == label]
        if len(idx) == 0:
            continue
        means = contrib[idx].mean(axis=0)
        print(f"{label:<16} " + " ".join(f"{m:>14.3f}" for m in means))


def sample_narratives(explainer, df, records):
    print("\nexplanations for one job per attack type")
    for label in ["ransomware", "insider_exfil", "mass_deletion", "benign_spike"]:
        idx = df.index[df.label == label]
        if len(idx) == 0:
            continue
        flagged_idx = [i for i in idx if df.loc[i, "if_flag"]]
        pick = int(flagged_idx[len(flagged_idx) // 2]) if flagged_idx else int(idx[len(idx) // 2])
        record = records[pick]
        print(f"\n  {label} — {record['job_id']}")
        print(f"    {explainer.narrative(record, bool(df.loc[pick, 'if_flag']))}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    detector = IsolationForestDetector().fit(df)
    scored = detector.score(df)

    explainer = SHAPExplainer(detector)
    records = explainer.explain(df)

    OUTPUT_PATH.write_text(json.dumps(records, indent=2))

    feature_importance(explainer, scored, records)
    sample_narratives(explainer, scored, records)

    flagged = scored[scored.if_flag].sort_values("if_score", ascending=False)
    plotted = 0
    for position in flagged.index[:MAX_PLOTS]:
        explainer.waterfall(int(position), scored.loc[position, "job_id"], PLOT_DIR)
        plotted += 1

    print(f"\n{len(records)} explanations written to {OUTPUT_PATH}")
    print(f"{plotted} waterfall plots written to {PLOT_DIR}")