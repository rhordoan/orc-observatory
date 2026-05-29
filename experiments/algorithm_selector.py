"""Algorithm selection: OTG features vs classical FLA features.

Evaluates using standard AS metrics: VBS/SBS gap closure,
10-fold CV accuracy, confusion matrix, and feature importance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


OTG_FEATURES = [
    "otg_compression",
    "otg_mean_terminal_rank",
    "otg_top5_reach",
    "otg_dag_depth",
    "otg_has_cycles",
    "otg_cycle_fraction",
    "mean_orc",
    "std_orc",
]

FLA_FEATURES = [
    "fdc",
    "autocorrelation_length",
    "information_content_h",
    "partial_info_content_m",
    "neutrality_ratio",
]


def _best_algo_per_instance(perf: pd.DataFrame, algo_cols: list[str]) -> pd.Series:
    """For each row, return the algorithm name with highest mean performance."""
    return perf[algo_cols].idxmax(axis=1).str.replace("perf_", "")


def _compute_selection_metrics(
    df: pd.DataFrame,
    feature_cols: list[str],
    algo_cols: list[str],
    label: str,
) -> dict:
    """Train selector, compute VBS/SBS gap closure and CV metrics."""
    X = df[feature_cols].fillna(0).values
    y_best = df["best_algo"].values
    perf_matrix = df[algo_cols].values

    n_folds = min(10, len(df))
    if n_folds < 3 or len(np.unique(y_best)) < 2:
        return {"label": label, "error": "insufficient data"}

    vbs = perf_matrix.max(axis=1).mean()
    sbs_scores = perf_matrix.mean(axis=0)
    sbs_idx = sbs_scores.argmax()
    sbs = sbs_scores[sbs_idx]
    sbs_name = algo_cols[sbs_idx].replace("perf_", "")

    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
        )),
    ])
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    pred_rf = cross_val_predict(pipe_rf, X, y_best, cv=cv)
    pred_lr = cross_val_predict(pipe_lr, X, y_best, cv=cv)

    def _selector_perf(preds):
        algo_to_idx = {c.replace("perf_", ""): i for i, c in enumerate(algo_cols)}
        selected_perf = [perf_matrix[i, algo_to_idx[p]] for i, p in enumerate(preds)]
        return np.mean(selected_perf)

    sel_rf = _selector_perf(pred_rf)
    sel_lr = _selector_perf(pred_lr)

    vbs_sbs_gap = vbs - sbs
    gap_closure_rf = (sel_rf - sbs) / max(vbs_sbs_gap, 1e-9) if vbs_sbs_gap > 1e-9 else 1.0
    gap_closure_lr = (sel_lr - sbs) / max(vbs_sbs_gap, 1e-9) if vbs_sbs_gap > 1e-9 else 1.0

    acc_rf = (pred_rf == y_best).mean()
    acc_lr = (pred_lr == y_best).mean()

    pipe_rf.fit(X, y_best)
    importances = dict(zip(feature_cols, pipe_rf.named_steps["clf"].feature_importances_))

    classes = sorted(np.unique(y_best))
    cm_rf = confusion_matrix(y_best, pred_rf, labels=classes).tolist()

    return {
        "label": label,
        "n_instances": len(df),
        "n_classes": len(classes),
        "classes": classes,
        "vbs_mean_perf": float(vbs),
        "sbs_mean_perf": float(sbs),
        "sbs_algorithm": sbs_name,
        "selector_rf_mean_perf": float(sel_rf),
        "selector_lr_mean_perf": float(sel_lr),
        "gap_closure_rf_pct": float(gap_closure_rf * 100),
        "gap_closure_lr_pct": float(gap_closure_lr * 100),
        "cv_accuracy_rf_pct": float(acc_rf * 100),
        "cv_accuracy_lr_pct": float(acc_lr * 100),
        "feature_importance_rf": {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])},
        "confusion_matrix_rf": cm_rf,
    }


def train_and_evaluate(
    features_path: Path,
    fla_path: Path | None,
    ils_path: Path,
    output_dir: Path,
) -> dict:
    feat = pd.read_csv(features_path)
    ils = pd.read_csv(ils_path)

    merge_keys = [c for c in ["type", "seed", "n", "k", "nu", "instance"]
                  if c in feat.columns and c in ils.columns]
    df = feat.merge(ils, on=merge_keys, suffixes=("_f", "_i"))

    algo_cols = [c for c in df.columns if c.startswith("perf_")]
    if not algo_cols:
        success_cols = [c for c in df.columns if c.startswith("success_") and c.endswith("_pct")]
        for c in success_cols:
            new_name = "perf_" + c.replace("success_", "").replace("_pct", "")
            df[new_name] = df[c]
            algo_cols.append(new_name)

    df["best_algo"] = _best_algo_per_instance(df, algo_cols)

    if fla_path and fla_path.exists():
        fla = pd.read_csv(fla_path)
        fla_merge = [c for c in merge_keys if c in fla.columns]
        df = df.merge(fla, on=fla_merge, suffixes=("", "_fla"))

    results = {}

    otg_available = [c for c in OTG_FEATURES if c in df.columns]
    if otg_available:
        results["otg_only"] = _compute_selection_metrics(df, otg_available, algo_cols, "OTG features only")

    fla_available = [c for c in FLA_FEATURES if c in df.columns]
    if fla_available:
        results["fla_only"] = _compute_selection_metrics(df, fla_available, algo_cols, "Classical FLA only")

    combined = otg_available + fla_available
    if combined and fla_available:
        results["combined"] = _compute_selection_metrics(df, combined, algo_cols, "OTG + FLA combined")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selector_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    df.to_csv(output_dir / "labeled_instances.csv", index=False)

    for key, res in results.items():
        if "error" not in res:
            print(f"  [{key}] CV-acc RF={res['cv_accuracy_rf_pct']:.1f}%  "
                  f"Gap-closure RF={res['gap_closure_rf_pct']:.1f}%  "
                  f"VBS={res['vbs_mean_perf']:.2f} SBS={res['sbs_mean_perf']:.2f} ({res['sbs_algorithm']})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fla", type=Path, default=None)
    parser.add_argument("--ils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_and_evaluate(args.features, args.fla, args.ils, args.output)


if __name__ == "__main__":
    main()
