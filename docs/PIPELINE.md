# Pipeline

CNNFin is organized as a staged pipeline. The notebooks are the Jarvis-friendly execution layer, and the Python modules contain the implementation logic.

## Stage Order

1. `exploration/data_fetching.ipynb`
2. `exploration/image_builder.ipynb`
3. `exploration/ML_models.ipynb`
4. `exploration/cnn_builder.ipynb`

## Data Flow

```text
Binance 5m candles
  -> raw candle pickles
  -> aligned multi-symbol market table
  -> IndicatorFactory.apply_all indicators
  -> triple-barrier labels
  -> valid samples
  -> four-panel PNG images
  -> ML baselines and CNN training
  -> metrics, predictions, models, result tables
```

## Download Stage

`data_fetching.ipynb` calls the download utilities in `cnnfin.data`.

It downloads all configured symbols and stores raw candle pickles in:

```text
artifacts/cnnfin_5m/raw_candles/
```

It also uses ZIP cache files under:

```text
artifacts/cnnfin_5m/raw_zips/
```

## Dataset Stage

`image_builder.ipynb` builds the canonical processed dataset:

- loads raw candles
- aligns all symbols by `Open time`
- forward-fills short altcoin gaps only
- keeps BTC OHLC data strict because labels depend on BTC candles
- applies `IndicatorFactory.apply_all`
- builds triple-barrier labels
- computes sample validity flags
- saves `merged_df.pkl` and split sample pickles

The main full table is:

```text
artifacts/cnnfin_5m/processed/merged_df.pkl
```

The clean modeling subset is:

```text
artifacts/cnnfin_5m/processed/samples.pkl
```

## Image Stage

`image_builder.ipynb` creates preview images by default.

Full image generation is disabled by default:

```python
RUN_FULL_IMAGE_BUILD = False
```

For a full Jarvis run, set:

```python
RUN_FULL_IMAGE_BUILD = True
```

Full images are saved under:

```text
artifacts/cnnfin_5m/images/{split}/{sample_id}.png
```

The full image manifest is:

```text
artifacts/cnnfin_5m/processed/image_manifest.pkl
```

## Baseline Model Stage

`ML_models.ipynb` trains:

- Logistic Regression
- XGBoost
- MLP
- LSTM

It uses `image_manifest.pkl` as the sample source to guarantee the same sample IDs as the CNN.

## CNN Stage

`cnn_builder.ipynb` trains EfficientNet-B0 on the full PNG image dataset.

The notebook uses staged fine-tuning:

1. train classifier head with the backbone frozen
2. unfreeze the top EfficientNet blocks and fine-tune by validation macro-F1

## Evaluation Stage

Each model writes predictions and metrics under:

```text
artifacts/cnnfin_5m/results/{model_name}/
```

Metrics include:

- macro-F1
- accuracy
- weighted F1
- per-class precision, recall, and F1
- confusion matrix
- bootstrap confidence interval for test macro-F1

