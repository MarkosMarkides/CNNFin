import unittest

import numpy as np
import pandas as pd

from cnnfin.labels import DOWN, NEUTRAL, UP, average_future_return_3class


def frame(closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "Open time": pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
            "Open": closes,
            "High": [value + 1.0 for value in closes],
            "Low": [value - 1.0 for value in closes],
            "Close": closes,
            "Volume": [1.0] * n,
        }
    )


class AverageFutureReturn3ClassTest(unittest.TestCase):
    def test_positive_future_average_is_up(self):
        df = frame([100.0, 102.0, 103.0, 104.0])
        out = average_future_return_3class(
            df,
            horizon=3,
            train_mask=[True, False, False, False],
            theta_down=-0.01,
            theta_up=0.01,
        )
        self.assertEqual(int(out.loc[0, "label"]), UP)
        self.assertEqual(out.loc[0, "label_status"], "labeled")
        self.assertAlmostEqual(float(out.loc[0, "future_avg_close"]), 103.0)

    def test_negative_future_average_is_down(self):
        df = frame([100.0, 98.0, 97.0, 96.0])
        out = average_future_return_3class(
            df,
            horizon=3,
            train_mask=[True, False, False, False],
            theta_down=-0.01,
            theta_up=0.01,
        )
        self.assertEqual(int(out.loc[0, "label"]), DOWN)

    def test_middle_future_average_is_neutral(self):
        df = frame([100.0, 100.1, 99.9, 100.0])
        out = average_future_return_3class(
            df,
            horizon=3,
            train_mask=[True, False, False, False],
            theta_down=-0.01,
            theta_up=0.01,
        )
        self.assertEqual(int(out.loc[0, "label"]), NEUTRAL)

    def test_insufficient_horizon_is_unlabeled(self):
        df = frame([100.0, 101.0, 102.0, 103.0])
        out = average_future_return_3class(
            df,
            horizon=3,
            train_mask=[True, False, False, False],
            theta_down=-0.01,
            theta_up=0.01,
        )
        self.assertTrue(np.isnan(out.loc[1, "label"]))
        self.assertEqual(out.loc[1, "label_status"], "insufficient_horizon")

    def test_thresholds_are_fit_from_train_mask_only(self):
        df = frame([100.0, 90.0, 100.0, 110.0, 130.0, 100.0, 200.0, 300.0, 400.0])
        train_mask = [True, True, True, False, False, False, False, False, False]
        out = average_future_return_3class(df, horizon=1, train_mask=train_mask)

        expected_train_returns = np.log(np.array([90.0 / 100.0, 100.0 / 90.0, 110.0 / 100.0]))
        expected_down, expected_up = np.quantile(expected_train_returns, [1.0 / 3.0, 2.0 / 3.0])
        self.assertAlmostEqual(float(out["theta_down"].iloc[0]), float(expected_down))
        self.assertAlmostEqual(float(out["theta_up"].iloc[0]), float(expected_up))
        self.assertEqual(int(out.loc[0, "label"]), DOWN)
        self.assertEqual(int(out.loc[1, "label"]), UP)
        self.assertEqual(int(out.loc[2, "label"]), NEUTRAL)


if __name__ == "__main__":
    unittest.main()
