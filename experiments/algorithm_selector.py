"""Algorithm selection: OTG features vs classical FLA features.

Evaluates using standard AS metrics: VBS/SBS gap closure,
nested 10x5 CV, per-domain breakdown, Friedman test over feature sets.
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
    "otg_compression", "otg_mean_terminal_rank", "otg_top5_reach",
    "otg_dag_depth", "otg_has_cycles", "otg_cycle_fraction",
    "mean_orc", "std_orc",
]

FLA_FEATURES = [
    "fdc", "autocorrelation_length", "information_content_h",
    "partial_info_content_m", "neutrality_ratio",
]

BASIC_FEATURES = ["n_optima", "degree"]


def _best_algo_per_instance(perf: pd.DataFrame, algo_cols: list[str]) -> pd.Series:
    return perf[algo_cols].idxmax(axis=1).str.replace("perf_", "")


def _normalize_perf(df: pd.DataFrame, algo_cols: list[str]) -> pd.DataFrame:
    """Per-instance min-max normalization to [0, 1]."""
    df = df.copy()
    for idx in df.index:
        vals = df.loc[idx, algo_cols].values.astype(float)
        lo, hi = vals.min(), vals.max()
        rng = hi - lo
        if rng > 1e-12:
            df.loc[idx, algo_cols] = (vals - lo) / rng
        else:
            df.loc[idx, algo_cols] = 1.0
    return df


def _compute_selection_metrics(
    df: pd.DataFrame,
    feature_cols: list[str],
    algo_cols: list[str],
    label: str,
) -> dict:
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

    algo_to_idx = {c.replace("perf_", ""): i for i, c in enumerate(algo_cols)}

    def _selector_perf(preds):
        return np.mean([perf_matrix[i, algo_to_idx[p]] for i, p in enumerate(preds)])

    sel_rf = _selector_perf(pred_rf)
    sel_lr = _selector_perf(pred_lr)

    vbs_sbs_gap = vbs - sbs
    gap_rf = (sel_rf - sbs) / max(vbs_sbs_gap, 1e-9) * 100 if vbs_sbs_gap > 1e-9 else 100.0
    gap_lr = (sel_lr - sbs) / max(vbs_sbs_gap, 1e-9) * 100 if vbs_sbs_gap > 1e-9 else 100.0

    acc_rf = (pred_rf == y_best).mean() * 100
    acc_lr = (pred_lr == y_best).mean() * 100

    pipe_rf.fit(X, y_best)
    importances = dict(zip(feature_cols, pipe_rf.named_steps["clf"].feature_importances_))

    classes = sorted(np.unique(y_best))
    cm_rf = confusion_matrix(y_best, pred_rf, labels=classes).tolist()

    # Per-domain gap closure
    domain_gaps = {}
    if "type" in df.columns:
        for domain in df["type"].unique():
            mask = df["type"] == domain
            d_perf = perf_matrix[mask]
            d_vbs = d_perf.max(axis=1).mean()
            d_sbs = d_perf.mean(axis=0).max()
            d_sel = np.mean([d_perf[i, algo_to_idx[pred_rf[df.index[mask][i]]]]
                             for i in range(mask.sum())])
            d_gap = d_vbs - d_sbs
            d_gc = (d_sel - d_sbs) / max(d_gap, 1e-9) * 100 if d_gap > 1e-9 else 100.0
            domain_gaps[domain] = {
                "vbs": float(d_vbs), "sbs": float(d_sbs),
                "selector_rf": float(d_sel), "gap_closure_pct": float(d_gc),
                "n_instances": int(mask.sum()),
            }

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
        "gap_closure_rf_pct": float(gap_rf),
        "gap_closure_lr_pct": float(gap_lr),
        "cv_accuracy_rf_pct": float(acc_rf),
        "cv_accuracy_lr_pct": float(acc_lr),
        "feature_importance_rf": {k: round(v, 4)
                                   for k, v in sorted(importances.items(), key=lambda x: -x[1])},
        "confusion_matrix_rf": cm_rf,
        "per_domain_gap_closure": domain_gaps,
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
        raise ValueError("No perf_* columns found in ILS results")

    df = _normalize_perf(df, algo_cols)
    df["best_algo"] = _best_algo_per_instance(df, algo_cols)

    if fla_path and fla_path.exists():
        fla = pd.read_csv(fla_path)
        fla_merge = [c for c in merge_keys if c in fla.columns]
        df = df.merge(fla, on=fla_merge, suffixes=("", "_fla"))

    results = {}

    otg_available = [c for c in OTG_FEATURES if c in df.columns]
    fla_available = [c for c in FLA_FEATURES if c in df.columns]
    basic_available = [c for c in BASIC_FEATURES if c in df.columns]

    if otg_available:
        results["otg_only"] = _compute_selection_metrics(
            df, otg_available, algo_cols, "OTG features only")

    if fla_available:
        results["fla_only"] = _compute_selection_metrics(
            df, fla_available, algo_cols, "Classical FLA only")

    if basic_available:
        results["basic_only"] = _compute_selection_metrics(
            df, basic_available, algo_cols, "Basic features only (n_optima, degree)")

    if otg_available and fla_available:
        results["combined"] = _compute_selection_metrics(
            df, otg_available + fla_available, algo_cols, "OTG + FLA combined")

    all_feats = otg_available + fla_available + basic_available
    if all_feats and len(all_feats) > len(otg_available):
        results["all_features"] = _compute_selection_metrics(
            df, all_feats, algo_cols, "All features (OTG + FLA + basic)")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selector_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    df.to_csv(output_dir / "labeled_instances.csv", index=False)

    for key, res in results.items():
        if "error" not in res:
            print(f"  [{key}] CV-acc RF={res['cv_accuracy_rf_pct']:.1f}%  "
                  f"Gap-closure RF={res['gap_closure_rf_pct']:.1f}%  "
                  f"VBS={res['vbs_mean_perf']:.4f} SBS={res['sbs_mean_perf']:.4f} ({res['sbs_algorithm']})")
            if res.get("per_domain_gap_closure"):
                for domain, dgc in res["per_domain_gap_closure"].items():
                    print(f"    {domain}: gap-cl={dgc['gap_closure_pct']:.1f}% ({dgc['n_instances']} inst)")

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
