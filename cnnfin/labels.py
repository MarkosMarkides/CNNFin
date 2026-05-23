from __future__ import annotations

import numpy as np
import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.utils import ensure_dir, write_json


DOWN = 0
NEUTRAL = 1
UP = 2

# Backward-compatible aliases for the V1 triple-barrier implementation.
SHORT = DOWN
NO_TRADE = NEUTRAL
LONG = UP


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


def average_future_return_3class(
    df: pd.DataFrame,
    *,
    horizon: int,
    train_mask: pd.Series | np.ndarray,
    close_col: str = "Close",
    theta_down: float | None = None,
    theta_up: float | None = None,
) -> pd.DataFrame:
    """Three-class labels from train-fitted average future close returns.

    The future target for row t is log(mean(Close[t+1:t+H]) / Close[t]).
    Thresholds are fitted only from the supplied train mask when not provided.
    """

    out = df.copy()
    close = out[close_col].astype(float)

    future_sum = sum(close.shift(-step) for step in range(1, horizon + 1))
    future_avg = future_sum / float(horizon)
    future_return = np.log(future_avg / close)

    out["future_avg_close"] = future_avg
    out["future_avg_log_return"] = future_return

    finite_returns = out["future_avg_log_return"].replace([np.inf, -np.inf], np.nan)
    train_mask = pd.Series(train_mask, index=out.index).fillna(False).astype(bool)
    fit_values = finite_returns.loc[train_mask & finite_returns.notna()]

    if theta_down is None or theta_up is None:
        if fit_values.empty:
            raise ValueError("Cannot fit average-return thresholds: no finite train returns in train_mask.")
        fitted = np.quantile(fit_values.to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0])
        theta_down = float(fitted[0]) if theta_down is None else float(theta_down)
        theta_up = float(fitted[1]) if theta_up is None else float(theta_up)
    else:
        theta_down = float(theta_down)
        theta_up = float(theta_up)

    if theta_down > theta_up:
        raise ValueError(f"theta_down must be <= theta_up, got {theta_down} > {theta_up}")

    labels = np.full(len(out), np.nan)
    statuses: list[str] = ["unlabeled"] * len(out)

    finite_mask = finite_returns.notna().to_numpy(dtype=bool)
    labels[finite_mask] = NEUTRAL
    labels[(finite_returns < theta_down).to_numpy(dtype=bool)] = DOWN
    labels[(finite_returns > theta_up).to_numpy(dtype=bool)] = UP

    for idx, is_finite in enumerate(finite_mask):
        if is_finite:
            statuses[idx] = "labeled"
        else:
            statuses[idx] = "insufficient_horizon"

    out["label"] = labels
    out["theta_down"] = theta_down
    out["theta_up"] = theta_up
    out["label_status"] = statuses
    return out


def build_label_table(config: ExperimentConfig) -> pd.DataFrame:
    from cnnfin.features import load_feature_table

    features = load_feature_table(config)
    train_mask = features["split"].eq("train") & features.get("candidate_valid_sample", True)
    labels = average_future_return_3class(features, horizon=config.horizon, train_mask=train_mask)
    ensure_dir(config.artifact_path("processed"))
    labels.to_pickle(config.artifact_path("processed", f"labels_{config.interval}.pkl"))

    status_counts = labels["label_status"].value_counts(dropna=False).to_dict()
    class_counts = labels["label"].dropna().astype(int).value_counts().sort_index().to_dict()
    write_json(
        {
            "rows": int(len(labels)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "label_method": "average_future_return_3class",
            "price_basis": "Close",
            "horizon": int(config.horizon),
            "theta_down": float(labels["theta_down"].dropna().iloc[0]),
            "theta_up": float(labels["theta_up"].dropna().iloc[0]),
            "insufficient_horizon_excluded": int(status_counts.get("insufficient_horizon", 0)),
        },
        config.artifact_path("reports", "label_report.json"),
    )
    return labels


def load_label_table(config: ExperimentConfig) -> pd.DataFrame:
    path = config.artifact_path("processed", f"labels_{config.interval}.pkl")
    if not path.exists():
        return build_label_table(config)
    labels = pd.read_pickle(path)
    required_v2_cols = {"future_avg_close", "future_avg_log_return", "theta_down", "theta_up"}
    if not required_v2_cols.issubset(labels.columns):
        return build_label_table(config)
    return labels
