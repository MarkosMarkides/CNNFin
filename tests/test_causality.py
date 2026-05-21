import unittest

import numpy as np
import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.features import add_core_features, add_lag_and_rolling_features


class CausalityTest(unittest.TestCase):
    def test_feature_row_does_not_change_when_future_changes(self):
        config = ExperimentConfig()
        rows = 140
        base = pd.DataFrame(
            {
                "Open time": pd.date_range("2021-01-01", periods=rows, freq="5min", tz="UTC"),
                "Open": np.linspace(100, 120, rows),
                "High": np.linspace(101, 121, rows),
                "Low": np.linspace(99, 119, rows),
                "Close": np.linspace(100, 120, rows),
                "Volume": np.linspace(1000, 2000, rows),
                "Quote asset volume": np.linspace(100000, 200000, rows),
                "Number of trades": np.arange(rows),
                "Taker buy base asset volume": np.linspace(500, 900, rows),
                "Taker buy quote asset volume": np.linspace(50000, 90000, rows),
            }
        )
        for symbol in config.alt_symbols:
            base[f"{symbol}_Close"] = np.linspace(10, 20, rows)
            base[f"{symbol}_Volume"] = np.linspace(100, 200, rows)

        changed = base.copy()
        changed.loc[80:, "Close"] = changed.loc[80:, "Close"] * 10.0
        changed.loc[80:, "High"] = changed.loc[80:, "High"] * 10.0
        changed.loc[80:, "Low"] = changed.loc[80:, "Low"] * 10.0

        a = add_core_features(base, config)
        b = add_core_features(changed, config)
        feature_cols = [c for c in a.columns if c != "Open time" and pd.api.types.is_numeric_dtype(a[c])]
        a, new_cols = add_lag_and_rolling_features(a, config, feature_cols)
        b, _ = add_lag_and_rolling_features(b, config, feature_cols)

        row = 79
        cols = feature_cols + new_cols
        diff = (a.loc[row, cols].astype(float) - b.loc[row, cols].astype(float)).abs().fillna(0.0)
        self.assertLess(float(diff.max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
