from __future__ import annotations

import numpy as np
import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.utils import ensure_dir, write_json


SHORT = 0
NO_TRADE = 1
LONG = 2


def _atr(df: pd.DataFrame, window: int, *, high: str = "High", low: str = "Low", close: str = "Close") -> pd.Series:
    prev_close = df[close].shift(1)
    tr = pd.concat(
        [
            df[high] - df[low],
            (df[high] - prev_close).abs(),
            (df[low] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def triple_barrier_3class(
    df: pd.DataFrame,
    *,
    horizon: int,
    atr_window: int,
    barrier_multiple: float,
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
) -> pd.DataFrame:
    """Causal 3-class triple-barrier labels for long/short/no-trade decisions."""

    out = df.copy()
    atr_col = f"ATR_{atr_window}"
    if atr_col not in out.columns:
        out[atr_col] = _atr(out, atr_window, high=high_col, low=low_col, close=close_col)

    labels = np.full(len(out), np.nan)
    hit_steps = np.full(len(out), np.nan)
    hit_sides: list[str | None] = [None] * len(out)
    statuses: list[str] = ["unlabeled"] * len(out)
    upper_values = np.full(len(out), np.nan)
    lower_values = np.full(len(out), np.nan)

    highs = out[high_col].to_numpy(dtype=float)
    lows = out[low_col].to_numpy(dtype=float)
    closes = out[close_col].to_numpy(dtype=float)
    atr_values = out[atr_col].to_numpy(dtype=float)

    n = len(out)
    for idx in range(n):
        atr_value = atr_values[idx]
        if not np.isfinite(atr_value):
            statuses[idx] = "missing_atr"
            continue
        if idx + horizon >= n:
            statuses[idx] = "insufficient_horizon"
            continue

        upper = closes[idx] + barrier_multiple * atr_value
        lower = closes[idx] - barrier_multiple * atr_value
        upper_values[idx] = upper
        lower_values[idx] = lower

        label = NO_TRADE
        status = "no_hit"
        side: str | None = "none"
        step_value = horizon

        for step in range(1, horizon + 1):
            future_idx = idx + step
            upper_hit = highs[future_idx] >= upper
            lower_hit = lows[future_idx] <= lower
            if upper_hit and lower_hit:
                label = np.nan
                status = "ambiguous_same_bar"
                side = "ambiguous"
                step_value = step
                break
            if upper_hit:
                label = LONG
                status = "upper_hit"
                side = "upper"
                step_value = step
                break
            if lower_hit:
                label = SHORT
                status = "lower_hit"
                side = "lower"
                step_value = step
                break

        labels[idx] = label
        hit_steps[idx] = step_value
        hit_sides[idx] = side
        statuses[idx] = status

    out["label"] = labels
    out["barrier_upper"] = upper_values
    out["barrier_lower"] = lower_values
    out["barrier_hit_step"] = hit_steps
    out["barrier_hit_side"] = hit_sides
    out["label_status"] = statuses
    return out


def build_label_table(config: ExperimentConfig) -> pd.DataFrame:
    from cnnfin.features import load_feature_table

    features = load_feature_table(config)
    labels = triple_barrier_3class(
        features,
        horizon=config.horizon,
        atr_window=config.atr_window,
        barrier_multiple=config.barrier_multiple,
    )
    ensure_dir(config.artifact_path("processed"))
    labels.to_pickle(config.artifact_path("processed", "labels_5m.pkl"))

    status_counts = labels["label_status"].value_counts(dropna=False).to_dict()
    class_counts = labels["label"].dropna().astype(int).value_counts().sort_index().to_dict()
    write_json(
        {
            "rows": int(len(labels)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "ambiguous_excluded": int(status_counts.get("ambiguous_same_bar", 0)),
            "insufficient_horizon_excluded": int(status_counts.get("insufficient_horizon", 0)),
        },
        config.artifact_path("reports", "label_report.json"),
    )
    return labels


def load_label_table(config: ExperimentConfig) -> pd.DataFrame:
    path = config.artifact_path("processed", "labels_5m.pkl")
    if not path.exists():
        return build_label_table(config)
    return pd.read_pickle(path)
