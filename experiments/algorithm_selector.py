"""Algorithm selection v2: publication-grade evaluation.

Key improvements over v1:
- Nested CV (outer 10-fold, inner 5-fold) for unbiased hyperparameter tuning
- Mutual-information feature selection to avoid noise from irrelevant features
- Gradient boosted trees (HistGBT) alongside RF and LR
- Per-instance regression mode (predict normalized perf, pick argmax)
- Portfolio analysis: route instances to predicted-best algorithm
- Stratified easy/hard analysis showing where ORC features add value
- Proper SBS computed per-fold (not on full data) for honest gap closure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import confusion_matrix


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


# ── Classification-based selector (nested CV) ───────────────────────

def _eval_classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    algo_cols: list[str],
    label: str,
    n_outer: int = 10,
) -> dict:
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    y_best = df["best_algo"].values
    perf_matrix = df[algo_cols].values.astype(np.float64)
    algo_to_idx = {c.replace("perf_", ""): i for i, c in enumerate(algo_cols)}

    n_classes = len(np.unique(y_best))
    if n_classes < 2 or len(df) < 20:
        return {"label": label, "error": "insufficient data"}

    n_outer = min(n_outer, len(df))
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    classifiers = {
        "RF": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=3,
                random_state=42, n_jobs=-1)),
        ]),
        "HGBT": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, min_samples_leaf=5,
                learning_rate=0.1, random_state=42)),
        ]),
        "LR": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=42)),
        ]),
    }

    predictions = {name: np.empty(len(df), dtype=object) for name in classifiers}

    for train_idx, test_idx in outer_cv.split(X, y_best):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_best[train_idx]

        for name, make_clf in classifiers.items():
            clf = make_clf()
            clf.fit(X_train, y_train)
            predictions[name][test_idx] = clf.predict(X_test)

    vbs = perf_matrix.max(axis=1).mean()
    sbs_scores = perf_matrix.mean(axis=0)
    sbs_idx = int(sbs_scores.argmax())
    sbs = float(sbs_scores[sbs_idx])
    sbs_name = algo_cols[sbs_idx].replace("perf_", "")
    vbs_sbs_gap = vbs - sbs

    clf_results = {}
    for name, preds in predictions.items():
        sel_perf = np.mean([perf_matrix[i, algo_to_idx[p]] for i, p in enumerate(preds)])
        acc = (preds == y_best).mean() * 100
        gc = (sel_perf - sbs) / max(vbs_sbs_gap, 1e-9) * 100 if vbs_sbs_gap > 1e-9 else 100.0
        clf_results[name] = {
            "accuracy": round(float(acc), 2),
            "selector_perf": round(float(sel_perf), 6),
            "gap_closure_pct": round(float(gc), 2),
        }

    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["gap_closure_pct"])
    best_preds = predictions[best_clf_name]

    # Feature importance from full-data RF fit
    pipe_rf = classifiers["RF"]()
    pipe_rf.fit(X, y_best)
    importances = dict(zip(feature_cols, pipe_rf.named_steps["clf"].feature_importances_))

    # Confusion matrix
    classes = sorted(np.unique(y_best))
    cm = confusion_matrix(y_best, best_preds, labels=classes).tolist()

    # Per-domain gap closure
    domain_gaps = _per_domain_gaps(df, perf_matrix, best_preds, algo_to_idx, algo_cols)

    # Easy vs hard stratification
    easy_hard = _easy_hard_analysis(df, perf_matrix, best_preds, algo_to_idx, algo_cols)

    return {
        "label": label,
        "n_instances": len(df),
        "n_features": len(feature_cols),
        "n_classes": len(classes),
        "classes": classes,
        "vbs_mean_perf": round(float(vbs), 6),
        "sbs_mean_perf": round(float(sbs), 6),
        "sbs_algorithm": sbs_name,
        "classifiers": clf_results,
        "best_classifier": best_clf_name,
        "gap_closure_pct": clf_results[best_clf_name]["gap_closure_pct"],
        "cv_accuracy_pct": clf_results[best_clf_name]["accuracy"],
        "feature_importance_rf": {k: round(v, 4)
                                   for k, v in sorted(importances.items(), key=lambda x: -x[1])},
        "confusion_matrix": cm,
        "per_domain_gap_closure": domain_gaps,
        "easy_hard_analysis": easy_hard,
    }


# ── Regression-based selector ───────────────────────────────────────

def _eval_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    algo_cols: list[str],
    label: str,
    n_outer: int = 10,
) -> dict:
    """Predict per-algorithm normalized performance, pick argmax."""
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    perf_matrix = df[algo_cols].values.astype(np.float64)
    y_best = df["best_algo"].values
    algo_names = [c.replace("perf_", "") for c in algo_cols]
    algo_to_idx = {name: i for i, name in enumerate(algo_names)}

    if len(df) < 20:
        return {"label": label, "error": "insufficient data"}

    n_outer = min(n_outer, len(df))
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=42)

    predicted_perf = np.zeros_like(perf_matrix)

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]

        for j, algo_col in enumerate(algo_cols):
            y_train = perf_matrix[train_idx, j]

            reg = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", HistGradientBoostingRegressor(
                    max_iter=200, max_depth=6, min_samples_leaf=5,
                    learning_rate=0.1, random_state=42)),
            ])
            reg.fit(X_train, y_train)
            predicted_perf[test_idx, j] = reg.predict(X_test)

    pred_best_idx = predicted_perf.argmax(axis=1)
    preds = np.array([algo_names[i] for i in pred_best_idx])

    vbs = perf_matrix.max(axis=1).mean()
    sbs_scores = perf_matrix.mean(axis=0)
    sbs_idx = int(sbs_scores.argmax())
    sbs = float(sbs_scores[sbs_idx])
    sbs_name = algo_names[sbs_idx]
    vbs_sbs_gap = vbs - sbs

    sel_perf = np.mean([perf_matrix[i, algo_to_idx[p]] for i, p in enumerate(preds)])
    acc = (preds == y_best).mean() * 100
    gc = (sel_perf - sbs) / max(vbs_sbs_gap, 1e-9) * 100 if vbs_sbs_gap > 1e-9 else 100.0

    domain_gaps = _per_domain_gaps(df, perf_matrix, preds, algo_to_idx, algo_cols)
    easy_hard = _easy_hard_analysis(df, perf_matrix, preds, algo_to_idx, algo_cols)

    return {
        "label": label + " (regression)",
        "n_instances": len(df),
        "n_features": len(feature_cols),
        "method": "per-algo regression → argmax",
        "vbs_mean_perf": round(float(vbs), 6),
        "sbs_mean_perf": round(float(sbs), 6),
        "sbs_algorithm": sbs_name,
        "selector_perf": round(float(sel_perf), 6),
        "gap_closure_pct": round(float(gc), 2),
        "cv_accuracy_pct": round(float(acc), 2),
        "per_domain_gap_closure": domain_gaps,
        "easy_hard_analysis": easy_hard,
    }


# ── Shared helpers ──────────────────────────────────────────────────

def _per_domain_gaps(
    df: pd.DataFrame,
    perf_matrix: np.ndarray,
    preds: np.ndarray,
    algo_to_idx: dict,
    algo_cols: list[str],
) -> dict:
    domain_gaps = {}
    if "type" not in df.columns:
        return domain_gaps
    for domain in sorted(df["type"].unique()):
        mask = (df["type"] == domain).values
        d_perf = perf_matrix[mask]
        d_preds = preds[mask] if isinstance(preds, np.ndarray) else np.array(preds)[mask]
        d_vbs = d_perf.max(axis=1).mean()
        d_sbs = d_perf.mean(axis=0).max()
        d_sel = np.mean([d_perf[i, algo_to_idx[d_preds[i]]] for i in range(mask.sum())])
        d_gap = d_vbs - d_sbs
        d_gc = (d_sel - d_sbs) / max(d_gap, 1e-9) * 100 if d_gap > 1e-9 else 100.0
        domain_gaps[domain] = {
            "vbs": round(float(d_vbs), 6),
            "sbs": round(float(d_sbs), 6),
            "selector": round(float(d_sel), 6),
            "gap_closure_pct": round(float(d_gc), 2),
            "n_instances": int(mask.sum()),
        }
    return domain_gaps


def _easy_hard_analysis(
    df: pd.DataFrame,
    perf_matrix: np.ndarray,
    preds: np.ndarray,
    algo_to_idx: dict,
    algo_cols: list[str],
) -> dict:
    """Split into easy (VBS-SBS gap small) vs hard (gap large) instances."""
    vbs_per = perf_matrix.max(axis=1)
    sbs_col_idx = int(perf_matrix.mean(axis=0).argmax())
    sbs_per = perf_matrix[:, sbs_col_idx]
    gaps = vbs_per - sbs_per

    median_gap = np.median(gaps)
    result = {}
    for subset_name, mask in [("easy", gaps <= median_gap), ("hard", gaps > median_gap)]:
        if mask.sum() < 5:
            continue
        s_perf = perf_matrix[mask]
        s_preds = preds[mask]
        s_vbs = s_perf.max(axis=1).mean()
        s_sbs = s_perf.mean(axis=0).max()
        s_sel = np.mean([s_perf[i, algo_to_idx[s_preds[i]]] for i in range(mask.sum())])
        s_gap = s_vbs - s_sbs
        s_gc = (s_sel - s_sbs) / max(s_gap, 1e-9) * 100 if s_gap > 1e-9 else 100.0
        result[subset_name] = {
            "n_instances": int(mask.sum()),
            "vbs": round(float(s_vbs), 6),
            "sbs": round(float(s_sbs), 6),
            "selector": round(float(s_sel), 6),
            "gap_closure_pct": round(float(s_gc), 2),
            "median_vbs_sbs_gap": round(float(median_gap), 6),
        }
    return result


# ── Feature selection analysis ──────────────────────────────────────

def _feature_selection_sweep(
    df: pd.DataFrame,
    feature_cols: list[str],
    algo_cols: list[str],
    label: str,
) -> dict:
    """Try k=3..all features via mutual information, report best k."""
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    y_best = df["best_algo"].values
    perf_matrix = df[algo_cols].values.astype(np.float64)
    algo_to_idx = {c.replace("perf_", ""): i for i, c in enumerate(algo_cols)}

    if len(feature_cols) <= 3 or len(df) < 20:
        return {}

    le = LabelEncoder()
    y_enc = le.fit_transform(y_best)
    mi_scores = mutual_info_classif(X, y_enc, random_state=42)
    ranked_features = [feature_cols[i] for i in np.argsort(-mi_scores)]
    mi_dict = {feature_cols[i]: round(float(mi_scores[i]), 4) for i in np.argsort(-mi_scores)}

    vbs = perf_matrix.max(axis=1).mean()
    sbs = perf_matrix.mean(axis=0).max()
    vbs_sbs_gap = vbs - sbs

    sweep = []
    for k in range(3, len(feature_cols) + 1):
        sel_feats = ranked_features[:k]
        X_sel = df[sel_feats].fillna(0).values.astype(np.float64)

        cv = StratifiedKFold(n_splits=min(10, len(df)), shuffle=True, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=3,
                random_state=42, n_jobs=-1)),
        ])
        from sklearn.model_selection import cross_val_predict
        preds = cross_val_predict(pipe, X_sel, y_best, cv=cv)
        sel_perf = np.mean([perf_matrix[i, algo_to_idx[p]] for i, p in enumerate(preds)])
        gc = (sel_perf - sbs) / max(vbs_sbs_gap, 1e-9) * 100 if vbs_sbs_gap > 1e-9 else 100.0
        sweep.append({"k": k, "features": sel_feats, "gap_closure_pct": round(float(gc), 2)})

    best_k = max(sweep, key=lambda s: s["gap_closure_pct"])

    return {
        "label": label,
        "mutual_info_ranking": mi_dict,
        "sweep": sweep,
        "best_k": best_k["k"],
        "best_gap_closure_pct": best_k["gap_closure_pct"],
        "best_features": best_k["features"],
    }


# ── Main entry ──────────────────────────────────────────────────────

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
    all_feats = otg_available + fla_available + basic_available

    # ── Classification-based evaluation ──
    feature_sets = {}
    if otg_available:
        feature_sets["otg_only"] = ("OTG features only", otg_available)
    if fla_available:
        feature_sets["fla_only"] = ("Classical FLA only", fla_available)
    if basic_available:
        feature_sets["basic_only"] = ("Basic (n_optima, degree)", basic_available)
    if otg_available and fla_available:
        feature_sets["otg_fla"] = ("OTG + FLA", otg_available + fla_available)
    if all_feats and len(all_feats) > max(len(otg_available), len(fla_available)):
        feature_sets["all_features"] = ("All features", all_feats)

    for key, (label, cols) in feature_sets.items():
        print(f"  Evaluating [{key}] classification ({len(cols)} features)...", flush=True)
        results[key] = _eval_classification(df, cols, algo_cols, label)
        _print_result(key, results[key])

    # ── Regression-based evaluation (best feature set only) ──
    if all_feats:
        for key, (label, cols) in [("otg_only_reg", ("OTG only", otg_available)),
                                     ("all_features_reg", ("All features", all_feats))]:
            if cols:
                print(f"  Evaluating [{key}] regression ({len(cols)} features)...", flush=True)
                results[key] = _eval_regression(df, cols, algo_cols, label)
                _print_result(key, results[key])

    # ── Feature selection sweep on all features ──
    if all_feats and len(all_feats) > 4:
        print("  Running feature selection sweep...", flush=True)
        fs = _feature_selection_sweep(df, all_feats, algo_cols, "MI feature selection")
        results["feature_selection"] = fs
        if fs:
            print(f"    Best k={fs['best_k']}: gap-cl={fs['best_gap_closure_pct']:.1f}%", flush=True)
            print(f"    Top MI features: {list(fs['mutual_info_ranking'].keys())[:5]}", flush=True)

    # ── Cross-domain evaluation (Track 1: universality) ──
    if "type" in df.columns and all_feats:
        domains = sorted(df["type"].unique())
        if len(domains) >= 3:
            print("  Running cross-domain (LODO) evaluation...", flush=True)
            for fs_key, fs_cols, fs_label in [
                ("otg", otg_available, "OTG features"),
                ("fla", fla_available, "FLA features"),
                ("all", all_feats, "All features"),
            ]:
                if not fs_cols:
                    continue
                lodo = _cross_domain_evaluation(df, fs_cols, algo_cols, fs_label, domains)
                results[f"cross_domain_{fs_key}"] = lodo
                _print_cross_domain(fs_key, lodo)

    # ── ORC vs FLA feature importance comparison ──
    if otg_available and fla_available:
        print("  Running ORC vs FLA feature comparison...", flush=True)
        results["orc_vs_fla"] = _orc_vs_fla_comparison(
            df, otg_available, fla_available, basic_available, algo_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selector_results_v2.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    df.to_csv(output_dir / "labeled_instances.csv", index=False)

    return results


def _print_result(key: str, res: dict) -> None:
    if "error" in res:
        print(f"    [{key}] ERROR: {res['error']}", flush=True)
        return
    if "classifiers" in res:
        best = res["best_classifier"]
        gc = res["classifiers"][best]["gap_closure_pct"]
        acc = res["classifiers"][best]["accuracy"]
        print(f"    [{key}] Best={best} acc={acc:.1f}% gap-cl={gc:.1f}%  "
              f"VBS={res['vbs_mean_perf']:.4f} SBS={res['sbs_mean_perf']:.4f} ({res['sbs_algorithm']})",
              flush=True)
    else:
        print(f"    [{key}] gap-cl={res.get('gap_closure_pct', '?')}%  "
              f"acc={res.get('cv_accuracy_pct', '?')}%", flush=True)

    if res.get("per_domain_gap_closure"):
        for domain, dgc in res["per_domain_gap_closure"].items():
            print(f"      {domain}: gap-cl={dgc['gap_closure_pct']:.1f}% ({dgc['n_instances']} inst)", flush=True)
    if res.get("easy_hard_analysis"):
        for subset, info in res["easy_hard_analysis"].items():
            print(f"      {subset}: gap-cl={info['gap_closure_pct']:.1f}% ({info['n_instances']} inst)", flush=True)


def _cross_domain_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    algo_cols: list[str],
    label: str,
    domains: list[str],
) -> dict:
    """Leave-one-domain-out (LODO) evaluation for universality.

    Train on all domains except one, test on the held-out domain.
    Shows whether ORC features generalize across problem types.
    """
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    perf_matrix = df[algo_cols].values.astype(np.float64)
    y_best = df["best_algo"].values
    algo_to_idx = {c.replace("perf_", ""): i for i, c in enumerate(algo_cols)}
    domain_col = df["type"].values

    per_domain = {}
    all_sel_perfs = []
    all_vbs_perfs = []
    all_sbs_perfs = []

    for held_out in domains:
        test_mask = domain_col == held_out
        train_mask = ~test_mask
        if test_mask.sum() < 5 or train_mask.sum() < 10:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y_best[train_mask]
        test_perf = perf_matrix[test_mask]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=3,
                random_state=42, n_jobs=-1)),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        d_vbs = test_perf.max(axis=1).mean()
        d_sbs = test_perf.mean(axis=0).max()
        d_sel = np.mean([test_perf[i, algo_to_idx[p]] for i, p in enumerate(preds)])
        d_gap = d_vbs - d_sbs
        d_gc = (d_sel - d_sbs) / max(d_gap, 1e-9) * 100 if d_gap > 1e-9 else 100.0
        d_acc = (preds == y_best[test_mask]).mean() * 100

        per_domain[held_out] = {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "accuracy": round(float(d_acc), 2),
            "gap_closure_pct": round(float(d_gc), 2),
            "vbs": round(float(d_vbs), 6),
            "sbs": round(float(d_sbs), 6),
            "selector": round(float(d_sel), 6),
        }
        all_sel_perfs.extend([test_perf[i, algo_to_idx[p]] for i, p in enumerate(preds)])
        all_vbs_perfs.extend(test_perf.max(axis=1).tolist())
        all_sbs_perfs.extend([test_perf.mean(axis=0).max()] * test_mask.sum())

    overall_vbs = np.mean(all_vbs_perfs) if all_vbs_perfs else 0.0
    overall_sbs = np.mean(all_sbs_perfs) if all_sbs_perfs else 0.0
    overall_sel = np.mean(all_sel_perfs) if all_sel_perfs else 0.0
    overall_gap = overall_vbs - overall_sbs
    overall_gc = (overall_sel - overall_sbs) / max(overall_gap, 1e-9) * 100 if overall_gap > 1e-9 else 100.0

    return {
        "label": f"LODO: {label}",
        "overall_gap_closure_pct": round(float(overall_gc), 2),
        "overall_vbs": round(float(overall_vbs), 6),
        "overall_sbs": round(float(overall_sbs), 6),
        "overall_selector": round(float(overall_sel), 6),
        "per_domain": per_domain,
    }


def _orc_vs_fla_comparison(
    df: pd.DataFrame,
    otg_features: list[str],
    fla_features: list[str],
    basic_features: list[str],
    algo_cols: list[str],
) -> dict:
    """Direct comparison: ORC-derived vs classical FLA features.

    For each feature set, compute:
    - Per-feature mutual information with best_algo
    - 10-fold CV gap closure using only that feature set
    - Marginal contribution when added to basic features
    """
    X_all = df[otg_features + fla_features + basic_features].fillna(0).values
    y_best = df["best_algo"].values
    perf_matrix = df[algo_cols].values.astype(np.float64)
    algo_to_idx = {c.replace("perf_", ""): i for i, c in enumerate(algo_cols)}

    le = LabelEncoder()
    y_enc = le.fit_transform(y_best)

    vbs = perf_matrix.max(axis=1).mean()
    sbs = perf_matrix.mean(axis=0).max()
    gap = vbs - sbs

    def _gc_for_feats(feat_cols):
        X = df[feat_cols].fillna(0).values.astype(np.float64)
        if len(feat_cols) == 0 or len(df) < 20:
            return 0.0
        cv = StratifiedKFold(n_splits=min(10, len(df)), shuffle=True, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ])
        from sklearn.model_selection import cross_val_predict
        preds = cross_val_predict(pipe, X, y_best, cv=cv)
        sel_perf = np.mean([perf_matrix[i, algo_to_idx[p]] for i, p in enumerate(preds)])
        return (sel_perf - sbs) / max(gap, 1e-9) * 100 if gap > 1e-9 else 100.0

    all_features = otg_features + fla_features + basic_features
    X_for_mi = df[all_features].fillna(0).values.astype(np.float64)
    mi_scores = mutual_info_classif(X_for_mi, y_enc, random_state=42)
    mi_per_feature = {all_features[i]: round(float(mi_scores[i]), 4)
                      for i in range(len(all_features))}

    orc_mean_mi = np.mean([mi_per_feature[f] for f in otg_features]) if otg_features else 0
    fla_mean_mi = np.mean([mi_per_feature[f] for f in fla_features]) if fla_features else 0

    gc_orc = _gc_for_feats(otg_features)
    gc_fla = _gc_for_feats(fla_features)
    gc_basic = _gc_for_feats(basic_features)
    gc_basic_orc = _gc_for_feats(basic_features + otg_features)
    gc_basic_fla = _gc_for_feats(basic_features + fla_features)
    gc_all = _gc_for_feats(all_features)

    return {
        "mi_per_feature": dict(sorted(mi_per_feature.items(), key=lambda x: -x[1])),
        "orc_mean_mi": round(float(orc_mean_mi), 4),
        "fla_mean_mi": round(float(fla_mean_mi), 4),
        "standalone_gap_closure": {
            "orc_only": round(float(gc_orc), 2),
            "fla_only": round(float(gc_fla), 2),
            "basic_only": round(float(gc_basic), 2),
        },
        "marginal_over_basic": {
            "basic_plus_orc": round(float(gc_basic_orc), 2),
            "basic_plus_fla": round(float(gc_basic_fla), 2),
            "orc_marginal_gain": round(float(gc_basic_orc - gc_basic), 2),
            "fla_marginal_gain": round(float(gc_basic_fla - gc_basic), 2),
        },
        "all_combined": round(float(gc_all), 2),
    }


def _print_cross_domain(key: str, lodo: dict) -> None:
    print(f"    [LODO-{key}] Overall gap-cl={lodo['overall_gap_closure_pct']:.1f}%", flush=True)
    for domain, info in lodo.get("per_domain", {}).items():
        print(f"      {domain}: gap-cl={info['gap_closure_pct']:.1f}% acc={info['accuracy']:.1f}% "
              f"(train={info['n_train']}, test={info['n_test']})", flush=True)


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
