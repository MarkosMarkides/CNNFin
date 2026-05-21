from __future__ import annotations

import time
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.utils import ensure_dir, write_json


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_BULK_MONTHLY_URL = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_COLUMNS = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]


def parse_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def interval_to_pandas_freq(interval: str) -> str:
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    if interval.endswith("d"):
        return f"{int(interval[:-1])}D"
    raise ValueError(f"Unsupported interval: {interval}")


def _request_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    url = f"{BINANCE_KLINES_URL}?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read().decode("utf-8")
    import json

    data = json.loads(payload)
    if isinstance(data, dict) and "code" in data:
        raise RuntimeError(f"Binance API error for {symbol}: {data}")
    return data


def _clean_klines(raw: list[list]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=BINANCE_COLUMNS)
    df = pd.DataFrame(raw, columns=BINANCE_COLUMNS)
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms", utc=True)
    float_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
    ]
    df[float_cols] = df[float_cols].astype(float)
    df["Number of trades"] = df["Number of trades"].astype(int)
    return df.drop(columns=["Ignore"])


def _clean_bulk_klines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = BINANCE_COLUMNS
    open_unit = "us" if pd.to_numeric(df["Open time"], errors="coerce").max() > 10**15 else "ms"
    close_unit = "us" if pd.to_numeric(df["Close time"], errors="coerce").max() > 10**15 else "ms"
    df["Open time"] = pd.to_datetime(df["Open time"], unit=open_unit, utc=True)
    df["Close time"] = pd.to_datetime(df["Close time"], unit=close_unit, utc=True)
    float_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
    ]
    df[float_cols] = df[float_cols].astype(float)
    df["Number of trades"] = df["Number of trades"].astype(int)
    return df.drop(columns=["Ignore"])


def _month_starts(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = parse_utc(start_date).tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    end_exclusive = parse_utc(end_date)
    end_month = end_exclusive.tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    months = []
    cur = start
    while cur <= end_month:
        if cur < end_exclusive:
            months.append(cur)
        cur = cur + pd.DateOffset(months=1)
    return months


def download_symbol_candles_bulk(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    *,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download official Binance monthly ZIP kline files for [start_date, end_date)."""

    cache_path = ensure_dir(cache_dir) if cache_dir is not None else None
    frames: list[pd.DataFrame] = []
    for month in _month_starts(start_date, end_date):
        ym = month.strftime("%Y-%m")
        filename = f"{symbol}-{interval}-{ym}.zip"
        url = f"{BINANCE_BULK_MONTHLY_URL}/{symbol}/{interval}/{filename}"
        if cache_path is not None:
            zip_path = cache_path / filename
            if zip_path.exists():
                data = zip_path.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                zip_path.write_bytes(data)
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()

        with zipfile.ZipFile(BytesIO(data)) as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not csv_names:
                raise RuntimeError(f"No CSV found inside {url}")
            with zf.open(csv_names[0]) as f:
                month_df = pd.read_csv(f, header=None)
        frames.append(_clean_bulk_klines(month_df))

    if not frames:
        return pd.DataFrame(columns=BINANCE_COLUMNS[:-1])
    start = parse_utc(start_date)
    end = parse_utc(end_date)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Open time"]).sort_values("Open time").reset_index(drop=True)
    return out[(out["Open time"] >= start) & (out["Open time"] < end)].reset_index(drop=True)


def download_symbol_candles(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    *,
    limit: int = 1000,
    sleep_seconds: float = 0.12,
) -> pd.DataFrame:
    """Download public Binance spot klines for [start_date, end_date)."""

    start = parse_utc(start_date)
    end = parse_utc(end_date)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) - 1

    chunks: list[pd.DataFrame] = []
    cursor = start_ms
    while cursor <= end_ms:
        raw = _request_klines(symbol, interval, cursor, end_ms, limit)
        if not raw:
            break
        chunk = _clean_klines(raw)
        chunks.append(chunk)
        last_open_ms = int(chunk["Open time"].iloc[-1].timestamp() * 1000)
        next_cursor = last_open_ms + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(sleep_seconds)

    if not chunks:
        return pd.DataFrame(columns=BINANCE_COLUMNS[:-1])
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["Open time"]).sort_values("Open time").reset_index(drop=True)
    return out[(out["Open time"] >= start) & (out["Open time"] < end)].reset_index(drop=True)


def download_raw_candles(config: ExperimentConfig, *, force: bool = False, use_bulk: bool = True) -> dict[str, Path]:
    raw_dir = ensure_dir(config.artifact_path("raw_candles"))
    paths: dict[str, Path] = {}
    report: dict[str, dict[str, object]] = {}
    for symbol in config.all_symbols:
        path = raw_dir / f"{symbol}_{config.interval}.pkl"
        if path.exists() and not force:
            print(f"{symbol}: using existing file {path}")
            df = pd.read_pickle(path)
        else:
            print(f"{symbol}: downloading {config.interval} candles")
            if use_bulk:
                try:
                    df = download_symbol_candles_bulk(
                        symbol,
                        config.interval,
                        config.start_date,
                        config.end_date,
                        cache_dir=config.artifact_path("raw_zips", symbol),
                    )
                except Exception as exc:
                    print(f"{symbol}: bulk download failed ({exc}); falling back to REST")
                    df = download_symbol_candles(symbol, config.interval, config.start_date, config.end_date)
            else:
                df = download_symbol_candles(symbol, config.interval, config.start_date, config.end_date)
            df.to_pickle(path)
            print(f"{symbol}: saved {len(df):,} rows to {path}")
        paths[symbol] = path
        report[symbol] = {
            "rows": int(len(df)),
            "start": str(df["Open time"].min()) if len(df) else None,
            "end": str(df["Open time"].max()) if len(df) else None,
            "path": str(path),
        }
    write_json(report, config.artifact_path("reports", "download_report.json"))
    return paths


def validate_cadence(df: pd.DataFrame, *, time_col: str, freq: str) -> dict[str, object]:
    ts = pd.to_datetime(df[time_col], utc=True)
    duplicates = int(ts.duplicated().sum())
    if ts.empty:
        return {"rows": 0, "duplicates": duplicates, "missing_count": 0, "missing_examples": []}
    expected = pd.date_range(ts.min(), ts.max(), freq=freq, tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(ts))
    return {
        "rows": int(len(df)),
        "duplicates": duplicates,
        "start": str(ts.min()),
        "end": str(ts.max()),
        "expected_rows": int(len(expected)),
        "missing_count": int(len(missing)),
        "missing_examples": [str(x) for x in missing[:20]],
    }


def _load_raw_symbol(config: ExperimentConfig, symbol: str) -> pd.DataFrame:
    path = config.artifact_path("raw_candles", f"{symbol}_{config.interval}.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Missing raw candles for {symbol}: {path}. Run stage 'download' first.")
    return pd.read_pickle(path)


def align_market_data(config: ExperimentConfig) -> pd.DataFrame:
    """Align target and altcoin candles on target timestamps."""

    freq = interval_to_pandas_freq(config.interval)
    target = _load_raw_symbol(config, config.target_symbol).copy()
    target = target.sort_values("Open time").drop_duplicates("Open time")
    report: dict[str, object] = {config.target_symbol: validate_cadence(target, time_col="Open time", freq=freq)}

    keep_target = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
    ]
    aligned = target[keep_target].set_index("Open time")

    for symbol in config.alt_symbols:
        alt = _load_raw_symbol(config, symbol).copy()
        alt = alt.sort_values("Open time").drop_duplicates("Open time")
        report[symbol] = validate_cadence(alt, time_col="Open time", freq=freq)
        prefix = symbol
        alt = alt[["Open time", "Close", "Volume"]].rename(
            columns={"Close": f"{prefix}_Close", "Volume": f"{prefix}_Volume"}
        )
        aligned = aligned.join(alt.set_index("Open time"), how="left")

    alt_cols = [c for c in aligned.columns if c.endswith("_Close") or c.endswith("_Volume")]
    missing_before = aligned[alt_cols].isna().sum().to_dict()
    aligned[alt_cols] = aligned[alt_cols].ffill(limit=3)
    missing_after = aligned[alt_cols].isna().sum().to_dict()

    target_cols = ["Open", "High", "Low", "Close", "Volume"]
    aligned = aligned.dropna(subset=target_cols + alt_cols)
    aligned = aligned.reset_index().rename(columns={"index": "Open time"})
    aligned = aligned[(aligned["Open time"] >= parse_utc(config.start_date)) & (aligned["Open time"] < parse_utc(config.end_date))]
    aligned = aligned.reset_index(drop=True)

    report["alignment"] = {
        "rows": int(len(aligned)),
        "alt_missing_before_ffill": {k: int(v) for k, v in missing_before.items()},
        "alt_missing_after_ffill": {k: int(v) for k, v in missing_after.items()},
        "dropped_rows_after_alignment": int(len(target) - len(aligned)),
    }
    write_json(report, config.artifact_path("reports", "data_integrity_report.json"))
    ensure_dir(config.artifact_path("processed"))
    aligned.to_pickle(config.artifact_path("processed", f"aligned_{config.interval}.pkl"))
    return aligned


def load_aligned_market_data(config: ExperimentConfig) -> pd.DataFrame:
    path = config.artifact_path("processed", f"aligned_{config.interval}.pkl")
    if not path.exists():
        return align_market_data(config)
    return pd.read_pickle(path)


def chronological_sample(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, max_rows).round().astype(int)
    return df.iloc[np.unique(idx)].reset_index(drop=True)
