from __future__ import annotations

import numpy as np
import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.data import load_aligned_market_data
from cnnfin.labels import triple_barrier_3class
from cnnfin.utils import ensure_dir, write_json
from feature_engineering.indicators import IndicatorFactory


def atr(df: pd.DataFrame, window: int, *, high: str = "High", low: str = "Low", close: str = "Close") -> pd.Series:
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


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out.fillna(50.0)


def stochastic_k(df: pd.DataFrame, window: int = 14) -> pd.Series:
    low_min = df["Low"].rolling(window, min_periods=window).min()
    high_max = df["High"].rolling(window, min_periods=window).max()
    return 100.0 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def add_core_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["return_1"] = close.pct_change(1)
    out["log_return_1"] = np.log(close).diff(1)
    for w in [3, 6, 12, 24, 48, 96]:
        out[f"return_{w}"] = close.pct_change(w)
        out[f"log_return_{w}"] = np.log(close).diff(w)

    out[f"ATR_{config.atr_window}"] = atr(out, config.atr_window)
    out["hl_range_pct"] = (out["High"] - out["Low"]) / close.replace(0, np.nan)
    out["oc_return"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)
    out["volume_return_1"] = out["Volume"].pct_change(1)

    out["RSI_14"] = rsi(close, 14)

    bb_ma = close.rolling(20, min_periods=20).mean()
    bb_sd = close.rolling(20, min_periods=20).std()
    bb_up = bb_ma + 2.0 * bb_sd
    bb_low = bb_ma - 2.0 * bb_sd
    out["BB_MA_20"] = bb_ma
    out["BB_width_20"] = (bb_up - bb_low) / bb_ma.replace(0, np.nan)
    out["BB_pct_20"] = (close - bb_low) / (bb_up - bb_low).replace(0, np.nan)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    pv = typical * out["Volume"]
    out["VWAP_24"] = pv.rolling(24, min_periods=24).sum() / out["Volume"].rolling(24, min_periods=24).sum()
    out["Volatility_24"] = out["return_1"].rolling(24, min_periods=24).std()
    out["Stoch_K_14"] = stochastic_k(out, 14)
    raw_obv = obv(close, out["Volume"])
    out["OBV_norm"] = raw_obv / raw_obv.abs().rolling(96, min_periods=24).max().replace(0, np.nan)

    for symbol in config.alt_symbols:
        close_col = f"{symbol}_Close"
        vol_col = f"{symbol}_Volume"
        out[f"{symbol}_return_1"] = out[close_col].pct_change(1)
        out[f"{symbol}_return_12"] = out[close_col].pct_change(12)
        out[f"{symbol}_volume_return_1"] = out[vol_col].pct_change(1)
        out[f"{symbol}_divergence_1"] = out[f"{symbol}_return_1"] - out["return_1"]

    alt_volume_cols = [f"{s}_Volume" for s in config.alt_symbols]
    out["TotalAltVolume"] = out[alt_volume_cols].sum(axis=1)
    out["TotalAltVolume_return_1"] = out["TotalAltVolume"].pct_change(1)

    return out.replace([np.inf, -np.inf], np.nan)


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    config: ExperimentConfig,
    base_feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    generated: dict[str, pd.Series] = {}
    new_cols: list[str] = []
    for col in base_feature_cols:
        for lag in config.lag_periods:
            name = f"{col}_lag_{lag}"
            generated[name] = df[col].shift(lag)
            new_cols.append(name)
        for window in config.rolling_windows:
            roll = df[col].rolling(window, min_periods=window)
            for stat_name, values in {
                "mean": roll.mean(),
                "std": roll.std(),
                "min": roll.min(),
                "max": roll.max(),
            }.items():
                name = f"{col}_roll_{window}_{stat_name}"
                generated[name] = values
                new_cols.append(name)
    generated_frame = pd.DataFrame(generated, index=df.index).astype("float32")
    out = pd.concat([df, generated_frame], axis=1)
    return out.replace([np.inf, -np.inf], np.nan), new_cols


def select_lag_rolling_base_columns(df: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    preferred = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Number of trades",
        "return_1",
        "return_12",
        "log_return_1",
        "log_return_12",
        f"ATR_{config.atr_window}",
        "hl_range_pct",
        "oc_return",
        "volume_return_1",
        "RSI_14",
        "BB_width_20",
        "BB_pct_20",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "VWAP_24",
        "Volatility_24",
        "Stoch_K_14",
        "OBV_norm",
        "TotalAltVolume",
        "TotalAltVolume_return_1",
    ]
    for symbol in config.alt_symbols:
        preferred.extend(
            [
                f"{symbol}_Close",
                f"{symbol}_Volume",
                f"{symbol}_return_1",
                f"{symbol}_return_12",
                f"{symbol}_divergence_1",
            ]
        )
    return [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def build_feature_table(config: ExperimentConfig) -> pd.DataFrame:
    return build_merged_df(config)


def _sample_id_from_timestamp(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%d%H%M")


def _split_for_year(config: ExperimentConfig, year: int) -> str | None:
    if year in config.train_years:
        return "train"
    if year in config.val_years:
        return "val"
    if year in config.test_years:
        return "test"
    return None


def _add_sample_validity_flags(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    idx = np.arange(n)
    expected_delta = pd.Timedelta(minutes=config.interval_minutes)
    ts = pd.to_datetime(out["Open time"], utc=True)
    prev_ok = ts.diff().eq(expected_delta).fillna(False).to_numpy(dtype=bool)
    prev_ok_int = prev_ok.astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(prev_ok_int)])

    if config.lookback <= 1:
        lookback_continuous = np.ones(n, dtype=bool)
    else:
        lookback_start = idx - config.lookback + 2
        lookback_continuous = np.zeros(n, dtype=bool)
        eligible = idx >= config.lookback - 1
        sums = prefix[idx[eligible] + 1] - prefix[lookback_start[eligible]]
        lookback_continuous[eligible] = sums == (config.lookback - 1)

    horizon_continuous = np.zeros(n, dtype=bool)
    horizon_eligible = idx + config.horizon < n
    sums = prefix[idx[horizon_eligible] + config.horizon + 1] - prefix[idx[horizon_eligible] + 1]
    horizon_continuous[horizon_eligible] = sums == config.horizon

    split = out["split"]
    lookback_same_split = split.notna() & split.shift(config.lookback - 1).eq(split)
    if config.lookback <= 1:
        lookback_same_split = split.notna()
    horizon_same_split = split.notna() & split.shift(-config.horizon).eq(split)

    out["prev_candle_continuous"] = prev_ok
    out["lookback_continuous"] = lookback_continuous
    out["horizon_continuous"] = horizon_continuous
    out["lookback_same_split"] = lookback_same_split.to_numpy(dtype=bool)
    out["horizon_same_split"] = horizon_same_split.to_numpy(dtype=bool)
    required_feature_cols = _model_feature_columns(out, config)
    row_features_present = out[required_feature_cols].notna().all(axis=1).to_numpy(dtype=bool)
    row_features_present_int = row_features_present.astype(np.int64)
    feature_prefix = np.concatenate([[0], np.cumsum(row_features_present_int)])
    model_lookback = config.image_lookback or config.lookback
    required_features_present = np.zeros(n, dtype=bool)
    required_eligible = idx >= model_lookback - 1
    required_start = idx - model_lookback + 1
    required_sums = feature_prefix[idx[required_eligible] + 1] - feature_prefix[required_start[required_eligible]]
    required_features_present[required_eligible] = required_sums == model_lookback
    out["required_features_present"] = required_features_present
    out["valid_sample"] = (
        out["label"].notna()
        & out["split"].notna()
        & (out["label_status"] != "ambiguous_same_bar")
        & out["required_features_present"]
        & out["lookback_continuous"]
        & out["horizon_continuous"]
        & out["lookback_same_split"]
        & out["horizon_same_split"]
    )
    return out


def _model_feature_columns(df: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    """Return the numeric source columns used to render each image window.

    This keeps the tabular and sequence baselines on the same information set
    as the CNN: BTC OHLC for the candlestick/GAF panels, altcoin closes for
    the divergence panel, and the configured indicator rows for the heatmap.
    """
    preferred = [
        "Open",
        "High",
        "Low",
        "Close",
        *[f"{symbol}_Close" for symbol in config.alt_symbols],
        *config.image_indicators,
    ]
    cols: list[str] = []
    seen: set[str] = set()
    for col in preferred:
        if col in seen:
            continue
        seen.add(col)
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _add_target_symbol_aliases(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[f"{config.target_symbol}_{col}"] = out[col]
    return out


def build_merged_df(config: ExperimentConfig) -> pd.DataFrame:
    aligned = load_aligned_market_data(config).copy().reset_index(drop=True)
    aligned["Open time"] = pd.to_datetime(aligned["Open time"], utc=True)
    aligned = _add_target_symbol_aliases(aligned, config)

    print("Applying IndicatorFactory.apply_all to aligned 5m data")
    indicator_output = IndicatorFactory.apply_all(aligned).copy()
    indicator_output["Open time"] = pd.to_datetime(indicator_output["Open time"], utc=True)
    indicator_cols = [c for c in indicator_output.columns if c not in aligned.columns]
    merged = aligned.merge(
        indicator_output[["Open time", *indicator_cols]],
        on="Open time",
        how="left",
        validate="one_to_one",
    )
    merged = merged.sort_values("Open time").reset_index(drop=True)

    print("Building 3-class triple-barrier labels")
    merged = triple_barrier_3class(
        merged,
        horizon=config.horizon,
        atr_window=config.atr_window,
        barrier_multiple=config.barrier_multiple,
    )
    merged["row_idx"] = np.arange(len(merged), dtype=np.int64)
    merged["sample_id"] = merged["Open time"].map(_sample_id_from_timestamp)
    merged["year"] = merged["Open time"].dt.year.astype(int)
    merged["split"] = merged["year"].map(lambda y: _split_for_year(config, int(y)))
    merged = _add_sample_validity_flags(merged, config)

    numeric_cols = [
        c
        for c in merged.columns
        if c not in {"Open time"} and pd.api.types.is_numeric_dtype(merged[c]) and merged[c].dtype == "float64"
    ]
    merged[numeric_cols] = merged[numeric_cols].astype("float32")

    missing_image_cols = [c for c in config.image_indicators if c not in merged.columns]
    if missing_image_cols:
        raise ValueError(f"Configured image indicators are missing after apply_all: {missing_image_cols}")

    model_cols = _model_feature_columns(merged, config)
    processed_dir = ensure_dir(config.artifact_path("processed"))
    merged.to_pickle(processed_dir / "merged_df.pkl")
    merged.to_pickle(processed_dir / "features_5m.pkl")
    stale_csv = processed_dir / "merged_df.csv"
    if stale_csv.exists():
        stale_csv.unlink()

    write_json(
        {
            "sequence_feature_cols": model_cols,
            "tabular_feature_cols": model_cols,
            "image_source_feature_cols": model_cols,
            "model_window_lookback": int(config.image_lookback or config.lookback),
            "image_indicators": config.image_indicators,
        },
        processed_dir / "feature_columns.json",
    )

    status_counts = merged["label_status"].value_counts(dropna=False).to_dict()
    class_counts = merged["label"].dropna().astype(int).value_counts().sort_index().to_dict()
    valid_samples = merged[merged["valid_sample"]].copy()
    write_json(
        {
            "rows": int(len(merged)),
            "columns": int(len(merged.columns)),
            "model_feature_count": int(len(model_cols)),
            "image_indicators": config.image_indicators,
            "missing_image_indicators": missing_image_cols,
            "label_status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "valid_sample_rows": int(len(valid_samples)),
            "valid_sample_split_counts": {
                str(k): int(v) for k, v in valid_samples["split"].value_counts().to_dict().items()
            },
            "valid_sample_class_counts": {
                split: {str(k): int(v) for k, v in part["label"].astype(int).value_counts().sort_index().to_dict().items()}
                for split, part in valid_samples.groupby("split")
            },
            "gap_crossing_excluded": int((~merged["lookback_continuous"] | ~merged["horizon_continuous"]).sum()),
            "split_boundary_excluded": int((~merged["lookback_same_split"] | ~merged["horizon_same_split"]).sum()),
            "feature_warmup_or_missing_excluded": int((~merged["required_features_present"]).sum()),
        },
        config.artifact_path("reports", "merged_df_report.json"),
    )
    write_json(
        {
            "rows": int(len(merged)),
            "columns": int(len(merged.columns)),
            "sequence_feature_count": int(len(model_cols)),
            "tabular_feature_count": int(len(model_cols)),
        },
        config.artifact_path("reports", "feature_report.json"),
    )
    return merged


def load_feature_table(config: ExperimentConfig) -> pd.DataFrame:
    path = config.artifact_path("processed", "features_5m.pkl")
    if not path.exists():
        return build_feature_table(config)
    return pd.read_pickle(path)
