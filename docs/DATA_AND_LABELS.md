# Data And Labels

This document describes the tabular data construction and label policy.

## Raw Candles

Raw data comes from Binance 1-hour OHLCV candles.

Each symbol is saved as a pickle:

```text
artifacts/cnnfin_1h/raw_candles/{SYMBOL}_1h.pkl
```

The key time column is `Open time`.

Core candle columns include:

- `Open time`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Close time`
- `Quote asset volume`
- `Number of trades`
- `Taker buy base asset volume`
- `Taker buy quote asset volume`

## Alignment

The dataset builder aligns BTC and altcoins by `Open time`.

BTC is the label target, so BTC OHLC data is not fabricated. Altcoin close and volume columns can be forward-filled for short gaps to keep the context panel usable.

The aligned market table is saved as:

```text
artifacts/cnnfin_1h/processed/aligned_1h.pkl
```

## Canonical Merged DataFrame

The canonical full table is:

```text
artifacts/cnnfin_1h/processed/merged_df.pkl
```

It contains:

- BTC OHLCV columns
- target-symbol aliases such as `BTCUSDT_Close`
- altcoin close and volume columns
- indicators from `IndicatorFactory.apply_all`
- average-future-return labels and threshold metadata
- split and sample metadata
- validity flags

## Indicator Engine

The current dataset uses:

```text
feature_engineering/indicators.py::IndicatorFactory.apply_all
```

The image heatmap uses 13 configured indicator columns from this output. Those are documented in [Image Design](IMAGE_DESIGN.md).

## Label Policy

Each sample at time `t` uses the future 12 one-hour closes:

```text
[t+1, t+12]
```

The target return is:

```text
future_avg_close_t = mean(Close_{t+1}, ..., Close_{t+12})
R_t = log(future_avg_close_t / Close_t)
```

The lower and upper thresholds are fitted only on valid training candidates:

```text
theta_down = 33.33rd percentile of train R_t
theta_up   = 66.67th percentile of train R_t
```

Those fixed thresholds are then applied unchanged to train, validation, and test.

Classes:

- `0 = down`: `R_t < theta_down`
- `1 = neutral`: `theta_down <= R_t <= theta_up`
- `2 = up`: `R_t > theta_up`

This makes the training labels approximately balanced while keeping validation and test distributions honest.

## Sample Validity

A row is a valid sample only if:

- label exists
- split exists
- required source features are present for the full model window
- lookback window is continuous
- future horizon is continuous
- lookback window stays inside the same split
- future label horizon stays inside the same split

This prevents leakage across missing-candle gaps and split boundaries.

## Split Files

Processed sample files are saved as pickles:

```text
artifacts/cnnfin_1h/processed/samples.pkl
artifacts/cnnfin_1h/processed/train_samples.pkl
artifacts/cnnfin_1h/processed/val_samples.pkl
artifacts/cnnfin_1h/processed/test_samples.pkl
artifacts/cnnfin_1h/processed/train_val_samples.pkl
```

Model notebooks use `image_manifest.pkl` after images exist, because that is the exact sample set available to the CNN.
