"""Train OTG-feature classifier to predict best ILS variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    LogisticRegression = None


FEATURE_COLS = [
    "otg_compression",
    "otg_mean_terminal_rank",
    "otg_top5_reach",
    "otg_dag_depth",
    "otg_has_cycles",
    "otg_cycle_fraction",
    "mean_orc",
    "std_orc",
]


def _best_algo_label(row: pd.Series, algo_cols: list[str]) -> str:
    best = -1.0
    name = algo_cols[0]
    for c in algo_cols:
        v = row.get(c, 0)
        if v > best:
            best = v
            name = c.replace("success_", "").replace("_pct", "")
    return name


def train_and_evaluate(
    features_path: Path,
    ils_path: Path,
    output_dir: Path,
) -> dict:
    if LogisticRegression is None:
        raise ImportError("pip install scikit-learn pandas")

    feat = pd.read_csv(features_path)
    ils = pd.read_csv(ils_path)
    merge_keys = [c for c in ["type", "seed", "n", "k", "nu", "instance"] if c in feat.columns and c in ils.columns]
    df = feat.merge(ils, on=merge_keys, suffixes=("_f", "_i"))

    algo_cols = [c for c in df.columns if c.startswith("success_") and c.endswith("_pct")]
    if not algo_cols:
        raise ValueError("No success_* columns in ILS results")

    df["best_algo"] = df.apply(lambda r: _best_algo_label(r, algo_cols), axis=1)

    X = df[FEATURE_COLS].fillna(0).values
    y = df["best_algo"].values

    if len(df) < 5:
        return {"accuracy": 0.0, "n_samples": len(df), "note": "too few samples"}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
    ])

    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])[0]
        if pred == y[test_idx][0]:
            correct += 1
    acc = correct / len(df)

    pipe.fit(X, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "loo_accuracy": acc,
        "n_samples": len(df),
        "classes": list(np.unique(y)),
        "feature_cols": FEATURE_COLS,
    }
    with (output_dir / "selector_results.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(out, f, indent=2)
    df.to_csv(output_dir / "labeled_instances.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train_and_evaluate(args.features, args.ils, args.output)
    print(f"LOO accuracy: {result['loo_accuracy']:.1%} ({result['n_samples']} instances)")


if __name__ == "__main__":
    main()
