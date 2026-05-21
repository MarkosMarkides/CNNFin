from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from cnnfin.config import ExperimentConfig
from cnnfin.utils import ensure_dir, write_json


LABELS = [0, 1, 2]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    target_names = [class_names[i] for i in LABELS]
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if y_proba is not None:
        out["mean_max_probability"] = float(np.max(y_proba, axis=1).mean())
    return out


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        values.append(f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0))
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"low": float(low), "high": float(high)}


def bootstrap_macro_f1_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        score_a = f1_score(y_true[idx], y_pred_a[idx], labels=LABELS, average="macro", zero_division=0)
        score_b = f1_score(y_true[idx], y_pred_b[idx], labels=LABELS, average="macro", zero_division=0)
        values.append(score_a - score_b)
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"low": float(low), "high": float(high)}


def save_model_outputs(
    config: ExperimentConfig,
    model_name: str,
    split: str,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    out_dir = ensure_dir(config.artifact_path("results", model_name))
    predictions.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    write_json(metrics, out_dir / f"{split}_metrics.json")
    pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(out_dir / f"{split}_confusion_matrix.csv")


def collect_test_results(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    per_class_rows = []
    results_dir = config.artifact_path("results")
    for metrics_path in results_dir.glob("*/test_metrics.json"):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        model_name = metrics_path.parent.name
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_f1_ci_low": metrics.get("macro_f1_ci", {}).get("low"),
                "macro_f1_ci_high": metrics.get("macro_f1_ci", {}).get("high"),
                "status": metrics.get("status", "ok"),
            }
        )
        report = metrics.get("classification_report", {})
        for class_name in config.class_names.values():
            if class_name in report:
                per_class_rows.append(
                    {
                        "model": model_name,
                        "class": class_name,
                        "precision": report[class_name].get("precision"),
                        "recall": report[class_name].get("recall"),
                        "f1_score": report[class_name].get("f1-score"),
                        "support": report[class_name].get("support"),
                    }
                )
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, na_position="last")
    ensure_dir(config.artifact_path("results"))
    summary.to_csv(config.artifact_path("results", "model_summary.csv"), index=False)
    pd.DataFrame(per_class_rows).to_csv(config.artifact_path("results", "per_class_test_metrics.csv"), index=False)
    return summary


def write_cnn_vs_best_baseline(config: ExperimentConfig) -> dict[str, Any] | None:
    summary = collect_test_results(config)
    if summary.empty or "efficientnet_b0" not in set(summary["model"]):
        return None
    baseline_rows = summary[(summary["model"] != "efficientnet_b0") & (summary["status"] == "ok")]
    if baseline_rows.empty:
        return None
    best_baseline = baseline_rows.iloc[0]["model"]
    cnn_pred = pd.read_csv(config.artifact_path("results", "efficientnet_b0", "test_predictions.csv"))
    base_pred = pd.read_csv(config.artifact_path("results", str(best_baseline), "test_predictions.csv"))
    merged = cnn_pred[["sample_id", "y_true", "y_pred"]].merge(
        base_pred[["sample_id", "y_pred"]],
        on="sample_id",
        suffixes=("_cnn", "_baseline"),
    )
    ci = bootstrap_macro_f1_diff_ci(
        merged["y_true"].to_numpy(),
        merged["y_pred_cnn"].to_numpy(),
        merged["y_pred_baseline"].to_numpy(),
        iterations=config.bootstrap_iterations,
        seed=config.seeds[0],
    )
    out = {
        "cnn_model": "efficientnet_b0",
        "baseline_model": str(best_baseline),
        "macro_f1_difference": float(
            f1_score(merged["y_true"], merged["y_pred_cnn"], labels=LABELS, average="macro", zero_division=0)
            - f1_score(merged["y_true"], merged["y_pred_baseline"], labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_f1_difference_ci": ci,
    }
    write_json(out, config.artifact_path("results", "cnn_vs_best_baseline.json"))
    return out
