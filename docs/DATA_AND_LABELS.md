# Data And Labels

This document describes the tabular data construction and label policy.

## Raw Candles

Raw data comes from Binance 5-minute OHLCV candles.

Each symbol is saved as a pickle:

```text
artifacts/cnnfin_5m/raw_candles/{SYMBOL}_5m.pkl
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
artifacts/cnnfin_5m/processed/aligned_5m.pkl
```

## Canonical Merged DataFrame

The canonical full table is:

```text
artifacts/cnnfin_5m/processed/merged_df.pkl
```

It contains:

- BTC OHLCV columns
- target-symbol aliases such as `BTCUSDT_Close`
- altcoin close and volume columns
- indicators from `IndicatorFactory.apply_all`
- triple-barrier labels and barrier metadata
- split and sample metadata
- validity flags

## Indicator Engine

The current dataset uses:

```text
feature_engineering/indicators.py::IndicatorFactory.apply_all
```

The image heatmap uses 13 configured indicator columns from this output. Those are documented in [Image Design](IMAGE_DESIGN.md).

## Label Policy

Each sample at time `t` uses the future 96 candles:

```text
[t+1, t+96]
```

Barriers are symmetric around `Close_t`:

```text
upper = Close_t + 1.5 * ATR_14_t
lower = Close_t - 1.5 * ATR_14_t
```

Classes:

- `0 = short`: lower barrier is hit before the upper barrier
- `1 = no_trade`: neither barrier is hit within the future horizon
- `2 = long`: upper barrier is hit before the lower barrier

If upper and lower are both touched in the same future candle before any earlier hit, the row is marked ambiguous and excluded from modeling.

## Sample Validity

A row is a valid sample only if:

- label exists
- label is not ambiguous
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
artifacts/cnnfin_5m/processed/samples.pkl
artifacts/cnnfin_5m/processed/train_samples.pkl
artifacts/cnnfin_5m/processed/val_samples.pkl
artifacts/cnnfin_5m/processed/test_samples.pkl
artifacts/cnnfin_5m/processed/train_val_samples.pkl
```

Model notebooks use `image_manifest.pkl` after images exist, because that is the exact sample set available to the CNN.

