# Project Overview

CNNFin is a reproducible experiment for comparing image-based market representations against numeric machine learning baselines.

## Research Question

The project asks whether representing recent market history as structured images can improve prediction quality compared with feeding the same information to standard numeric models.

The prediction task is not a trading backtest. It is a 3-class supervised classification problem evaluated primarily with test macro-F1.

## Contributions

1. Test whether image representations of financial time-series windows improve BTCUSDT directional predictability.
2. Test whether the chosen four-panel image design is useful as a market representation.
3. Compare the CNN against numeric baselines on the same sample IDs and the same source information.

## Market Universe

The target is `BTCUSDT`.

The altcoin context symbols are:

- `ADAUSDT`
- `BNBUSDT`
- `ETHUSDT`
- `LINKUSDT`
- `LTCUSDT`
- `SOLUSDT`
- `TRXUSDT`
- `XLMUSDT`
- `XRPUSDT`

The altcoin context is used in the image divergence panel and in the numeric model input surface.

## Time Setup

- Exchange/source: Binance historical candles
- Interval: 5 minutes
- Start: `2021-01-01T00:00:00Z`
- End: `2026-01-01T00:00:00Z`
- Test year: 2025

The configured end date is exclusive, so this covers candles through 2025.

## Splits

Splits are chronological:

- Train: 2021, 2022, 2023
- Validation: 2024
- Test: 2025

Model selection, early stopping, and hyperparameter decisions should use train and validation only. The test set should be evaluated once per final model.

## Main Metric

The primary metric is test macro-F1.

Macro-F1 is used because the task is multi-class and class imbalance matters, especially for the `no_trade` class.

## Current Modeling Target

The model predicts the triple-barrier label for a sample time `t`:

- `0 = short`
- `1 = no_trade`
- `2 = long`

The label uses the future 96 candles after `t`; model inputs use only data at or before `t`.

