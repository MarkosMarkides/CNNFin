from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Single source of truth for the CNNFin 5m experiment."""

    experiment_name: str = "cnnfin_5m"
    artifact_dir: str = "artifacts/cnnfin_5m"

    target_symbol: str = "BTCUSDT"
    alt_symbols: list[str] = field(
        default_factory=lambda: [
            "ADAUSDT",
            "BNBUSDT",
            "ETHUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "SOLUSDT",
            "TRXUSDT",
            "XLMUSDT",
            "XRPUSDT",
        ]
    )
    interval: str = "5m"
    interval_minutes: int = 5
    start_date: str = "2021-01-01T00:00:00Z"
    end_date: str = "2026-01-01T00:00:00Z"

    lookback: int = 96
    horizon: int = 96
    atr_window: int = 14
    barrier_multiple: float = 1.5
    class_names: dict[int, str] = field(default_factory=lambda: {0: "short", 1: "no_trade", 2: "long"})

    train_years: list[int] = field(default_factory=lambda: [2021, 2022, 2023])
    val_years: list[int] = field(default_factory=lambda: [2024])
    test_years: list[int] = field(default_factory=lambda: [2025])

    lag_periods: list[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 48, 96])
    rolling_windows: list[int] = field(default_factory=lambda: [12, 24, 48, 96])
    image_indicators: list[str] = field(
        default_factory=lambda: [
            "return_1",
            "return_10",
            "RSI_14",
            "ATR_14",
            "CCI_20",
            "MACD26_12",
            "SIGNAL26_12",
            "HIST26_12",
            "VWAP_14",
            "Volatility_12",
            "%K_14",
            "%D_14",
            "OBV",
        ]
    )

    models: list[str] = field(
        default_factory=lambda: ["logistic_regression", "random_forest", "xgboost", "mlp", "numeric_lstm", "efficientnet_b0"]
    )
    seeds: list[int] = field(default_factory=lambda: [42])
    bootstrap_iterations: int = 1000

    image_lookback: int | None = None
    image_size_each: int = 112
    cnn_input_size: int = 224
    batch_size: int = 64
    num_epochs: int = 20
    early_stopping_patience: int = 5
    learning_rate: float = 1e-4
    freeze_cnn_backbone: bool = True
    num_workers: int = 0

    max_samples_per_split: int | None = None

    def artifact_path(self, *parts: str) -> Path:
        return Path(self.artifact_dir).joinpath(*parts)

    @property
    def all_symbols(self) -> list[str]:
        return [self.target_symbol, *self.alt_symbols]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["class_names"] = {int(k): v for k, v in self.class_names.items()}
        return data


def _coerce_class_names(value: Any) -> dict[int, str]:
    if value is None:
        return {0: "short", 1: "no_trade", 2: "long"}
    return {int(k): str(v) for k, v in dict(value).items()}


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a YAML mapping.")
        data.update(loaded)
    data.update({k: v for k, v in overrides.items() if v is not None})
    if "class_names" in data:
        data["class_names"] = _coerce_class_names(data["class_names"])
    return ExperimentConfig(**data)


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    return path
