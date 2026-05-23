import tempfile
import unittest
from pathlib import Path

import json
import numpy as np
import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.data import align_market_data
from cnnfin.features import build_feature_table
from cnnfin.images import generate_images
from cnnfin.samples import build_samples


def synthetic_candles(symbol: str, periods_per_year: int = 360) -> pd.DataFrame:
    frames = []
    for year, offset in [(2021, 0), (2024, 500), (2025, 1000)]:
        ts = pd.date_range(f"{year}-01-01", periods=periods_per_year, freq="1h", tz="UTC")
        x = np.arange(periods_per_year, dtype=float)
        base = 100.0 + offset + 0.05 * x + 2.0 * np.sin(x / 6.0)
        if symbol != "BTCUSDT":
            base = base * (0.1 + (abs(hash(symbol)) % 7) / 100.0)
        close = base
        open_ = close - 0.02
        high = close + 1.0
        low = close - 1.0
        volume = 1000.0 + x
        frames.append(
            pd.DataFrame(
                {
                    "Open time": ts,
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "Close time": ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
                    "Quote asset volume": close * volume,
                    "Number of trades": x.astype(int) + 1,
                    "Taker buy base asset volume": volume * 0.5,
                    "Taker buy quote asset volume": close * volume * 0.5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class PipelineSmokeTest(unittest.TestCase):
    def test_dataset_and_images_build_from_synthetic_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                artifact_dir=str(Path(tmp) / "artifacts"),
                interval="1h",
                interval_minutes=60,
                lookback=8,
                image_lookback=5,
                horizon=4,
                atr_window=14,
                image_size_each=24,
                max_samples_per_split=30,
                start_date="2021-01-01T00:00:00Z",
                end_date="2026-01-01T00:00:00Z",
            )
            raw_dir = Path(config.artifact_dir) / "raw_candles"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for symbol in config.all_symbols:
                synthetic_candles(symbol).to_pickle(raw_dir / f"{symbol}_{config.interval}.pkl")

            aligned = align_market_data(config)
            merged = build_feature_table(config)
            samples = build_samples(config)
            manifest = generate_images(config)
            feature_columns = json.loads(Path(config.artifact_dir, "processed", "feature_columns.json").read_text())

            self.assertFalse(aligned.empty)
            self.assertFalse(merged.empty)
            for col in [
                "BTCUSDT_Open",
                "BTCUSDT_High",
                "BTCUSDT_Low",
                "BTCUSDT_Close",
                "BTCUSDT_Volume",
                "ADAUSDT_Close",
                "RSI_14",
                "ATR_14",
                "label",
                "future_avg_close",
                "future_avg_log_return",
                "theta_down",
                "theta_up",
                "sample_id",
                "split",
                "candidate_valid_sample",
                "required_features_present",
                "valid_sample",
            ]:
                self.assertIn(col, merged.columns)
            for col in config.image_indicators:
                self.assertIn(col, merged.columns)
            self.assertEqual(set(samples["split"]), {"train", "val", "test"})
            self.assertTrue(samples["lookback_continuous"].all())
            self.assertTrue(samples["horizon_continuous"].all())
            self.assertTrue(samples["lookback_same_split"].all())
            self.assertTrue(samples["horizon_same_split"].all())
            self.assertTrue(samples["required_features_present"].all())
            self.assertEqual(set(merged.loc[merged["valid_sample"] & merged["split"].eq("train"), "label"].astype(int)), {0, 1, 2})
            train_counts = (
                merged.loc[merged["valid_sample"] & merged["split"].eq("train"), "label"]
                .astype(int)
                .value_counts()
                .sort_index()
            )
            self.assertLessEqual(int(train_counts.max() - train_counts.min()), 1)
            self.assertEqual(set(samples["label"].astype(int)), {0, 1, 2})
            expected_source_cols = [
                "Open",
                "High",
                "Low",
                "Close",
                *[f"{symbol}_Close" for symbol in config.alt_symbols],
                *config.image_indicators,
            ]
            expected_source_cols = list(dict.fromkeys(expected_source_cols))
            self.assertEqual(feature_columns["image_source_feature_cols"], expected_source_cols)
            self.assertEqual(feature_columns["tabular_feature_cols"], expected_source_cols)
            self.assertEqual(feature_columns["sequence_feature_cols"], expected_source_cols)
            self.assertEqual(feature_columns["model_window_lookback"], config.image_lookback)
            self.assertEqual(set(manifest["split"]), {"train", "val", "test"})
            self.assertEqual(set(manifest["label"].astype(int)), {0, 1, 2})
            self.assertTrue((manifest["image_lookback"] == 5).all())
            for image_path in manifest["image_path"]:
                self.assertTrue(Path(image_path).exists())


if __name__ == "__main__":
    unittest.main()
