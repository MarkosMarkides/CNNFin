# CNNFin

CNNFin is the implementation for an academic image-vs-numeric financial machine learning experiment.

The research question is:

> If market data is represented as structured images instead of only numerical features, can a CNN improve 3-class BTCUSDT directional prediction versus classical numeric ML baselines?

The experiment predicts 5-minute BTCUSDT triple-barrier outcomes using Binance candles from 2021 through 2025. The primary metric is test macro-F1 on the untouched 2025 test set.

## Current Experiment

- Target: `BTCUSDT`
- Context symbols: `ADAUSDT`, `BNBUSDT`, `ETHUSDT`, `LINKUSDT`, `LTCUSDT`, `SOLUSDT`, `TRXUSDT`, `XLMUSDT`, `XRPUSDT`
- Candle interval: 5 minutes
- Date range: `2021-01-01T00:00:00Z` through `2026-01-01T00:00:00Z`
- Modeling split:
  - Train: 2021-2023
  - Validation: 2024
  - Test: 2025
- Label classes:
  - `0 = short`
  - `1 = no_trade`
  - `2 = long`
- Label policy: 96-candle triple barrier with symmetric `1.5 * ATR_14` barriers.
- Image window: previous 30 candles.
- Model/sample lookback for numeric baselines: the same 30-candle image source window.

## Install

Create and activate an environment, then install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a GPU provider such as Jarvis Labs, use the PyTorch install command appropriate for the image if your base image requires a specific CUDA wheel. Then install `requirements.txt`.

## Canonical Notebook Run Order

Run the notebooks in this order:

1. `exploration/data_fetching.ipynb`
2. `exploration/image_builder.ipynb`
3. `exploration/ML_models.ipynb`
4. `exploration/cnn_builder.ipynb`

`image_builder.ipynb` has `RUN_FULL_IMAGE_BUILD = False` by default. Set it to `True` on Jarvis when you are ready to generate the full image dataset.

`ML_models.ipynb` and `cnn_builder.ipynb` have `DEBUG_MODE = False` by default. Set `DEBUG_MODE = True` only for quick smoke tests.

## What Each Stage Produces

- `data_fetching.ipynb`: downloads raw Binance candles into `artifacts/cnnfin_5m/raw_candles/`.
- `image_builder.ipynb`: builds `merged_df.pkl`, `samples.pkl`, split sample pickles, preview images, and optionally the full image dataset plus `image_manifest.pkl`.
- `ML_models.ipynb`: trains Logistic Regression, XGBoost, MLP, and LSTM on the exact numeric source windows used by the CNN images.
- `cnn_builder.ipynb`: trains EfficientNet-B0 on the generated four-panel PNG images.

Main outputs are written under `artifacts/cnnfin_5m/`.

## Fair Comparison Rule

All models use the same samples and the same source information.

- CNN: uses one PNG image generated from the 30-candle source window.
- Logistic Regression, XGBoost, MLP: use the flattened numeric version of that same 30-candle source window.
- LSTM: uses that same source window as a sequence.
- All model notebooks use sample IDs from `artifacts/cnnfin_5m/processed/image_manifest.pkl`.

This avoids giving numeric models extra information that was not available to the CNN image model.

## Documentation Map

- [Project Overview](docs/PROJECT_OVERVIEW.md)
- [Pipeline](docs/PIPELINE.md)
- [Data And Labels](docs/DATA_AND_LABELS.md)
- [Image Design](docs/IMAGE_DESIGN.md)
- [Models](docs/MODELS.md)
- [Notebooks](docs/NOTEBOOKS.md)
- [Artifacts](docs/ARTIFACTS.md)
- [Jarvis Runbook](docs/JARVIS_RUNBOOK.md)
- [Development](docs/DEVELOPMENT.md)

## Source Layout

- `cnnfin/`: importable pipeline modules.
- `feature_engineering/indicators.py`: technical indicators used by the dataset builder.
- `image_generation/image_generator.py`: four-panel image renderer.
- `configs/cnnfin_5m.yaml`: experiment configuration.
- `exploration/`: runnable notebooks.
- `tests/`: unit and smoke tests.

## Quick Validation

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

