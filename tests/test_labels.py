import unittest

import numpy as np
import pandas as pd

from cnnfin.labels import LONG, NO_TRADE, SHORT, triple_barrier_3class


def frame(highs, lows):
    n = len(highs)
    return pd.DataFrame(
        {
            "Open time": pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
            "Open": [100.0] * n,
            "High": highs,
            "Low": lows,
            "Close": [100.0] * n,
            "Volume": [1.0] * n,
            "ATR_14": [10.0] * n,
        }
    )


class TripleBarrier3ClassTest(unittest.TestCase):
    def test_upper_hit_first_is_long(self):
        df = frame([101, 116, 101, 101], [99, 95, 99, 99])
        out = triple_barrier_3class(df, horizon=3, atr_window=14, barrier_multiple=1.5)
        self.assertEqual(int(out.loc[0, "label"]), LONG)
        self.assertEqual(out.loc[0, "label_status"], "upper_hit")
        self.assertEqual(int(out.loc[0, "barrier_hit_step"]), 1)

    def test_lower_hit_first_is_short(self):
        df = frame([101, 104, 101, 101], [99, 84, 99, 99])
        out = triple_barrier_3class(df, horizon=3, atr_window=14, barrier_multiple=1.5)
        self.assertEqual(int(out.loc[0, "label"]), SHORT)
        self.assertEqual(out.loc[0, "label_status"], "lower_hit")

    def test_neither_hit_is_no_trade(self):
        df = frame([101, 104, 108, 110], [99, 96, 94, 91])
        out = triple_barrier_3class(df, horizon=3, atr_window=14, barrier_multiple=1.5)
        self.assertEqual(int(out.loc[0, "label"]), NO_TRADE)
        self.assertEqual(out.loc[0, "label_status"], "no_hit")

    def test_same_bar_double_hit_is_ambiguous(self):
        df = frame([101, 116, 101, 101], [99, 84, 99, 99])
        out = triple_barrier_3class(df, horizon=3, atr_window=14, barrier_multiple=1.5)
        self.assertTrue(np.isnan(out.loc[0, "label"]))
        self.assertEqual(out.loc[0, "label_status"], "ambiguous_same_bar")

    def test_insufficient_horizon_is_unlabeled(self):
        df = frame([101, 101, 101, 101], [99, 99, 99, 99])
        out = triple_barrier_3class(df, horizon=3, atr_window=14, barrier_multiple=1.5)
        self.assertTrue(np.isnan(out.loc[1, "label"]))
        self.assertEqual(out.loc[1, "label_status"], "insufficient_horizon")


if __name__ == "__main__":
    unittest.main()
