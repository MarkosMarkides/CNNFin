# Notebooks

The project is notebook-first for Jarvis execution. The notebooks call the project modules rather than duplicating all implementation details inline.

Run notebooks in this order:

1. `exploration/data_fetching.ipynb`
2. `exploration/image_builder.ipynb`
3. `exploration/ML_models_regularized_variants.ipynb`
4. `exploration/cnn_builder.ipynb`

## `data_fetching.ipynb`

Purpose:

- load `configs/cnnfin_1h.yaml`
- download all configured 1-hour Binance candles
- save raw candle pickles
- write download verification reports

Important toggle:

```python
FORCE_DOWNLOAD = False
```

Set `FORCE_DOWNLOAD = True` only when you intentionally want to redownload files that already exist.

## `image_builder.ipynb`

Purpose:

- load raw candles
- build aligned market data
- apply indicators
- build average-future-return labels
- build valid samples
- save `merged_df.pkl`, `samples.pkl`, and split sample files
- render preview images
- optionally generate the full image dataset

Important toggle:

```python
RUN_FULL_IMAGE_BUILD = False
```

When switching from V1 artifacts, set this once before rebuilding:

```python
CLEAN_DERIVED_ARTIFACTS = True
```

This preserves raw candles and ZIP downloads, but removes stale processed files, images, model outputs, and result tables.

For Jarvis full image generation, set:

```python
RUN_FULL_IMAGE_BUILD = True
```

Preview images are saved to:

```text
artifacts/cnnfin_1h/image_preview/
```

Full images are saved to:

```text
artifacts/cnnfin_1h/images/
```

## `ML_models_regularized_variants.ipynb`

Purpose:

- load `merged_df.pkl`
- load `feature_columns.json`
- load `image_manifest.pkl`
- build the exact same 30 one-hour candle numeric source windows used to create CNN images
- train Logistic Regression, XGBoost, MLP, and LSTM
- use raw numeric windows only, with no window normalization
- save metrics, predictions, confusion matrices, and model artifacts

Important toggle:

```python
DEBUG_MODE = False
```

Set `DEBUG_MODE = True` only for a small smoke run.

This notebook should be run after full image generation, because it uses `image_manifest.pkl` as the source of sample IDs.

## `cnn_builder.ipynb`

Purpose:

- load `image_manifest.pkl`
- load generated PNG images
- train EfficientNet-B0
- select checkpoint by validation macro-F1
- evaluate once on the 2025 test set
- save CNN metrics, predictions, confusion matrix, and checkpoint

Current CNN outputs are saved as `efficientnet_b0`.

Important toggle:

```python
DEBUG_MODE = False
```

Set `DEBUG_MODE = True` only for a small smoke run.

## Notebook Failure Checklist

If a notebook fails:

- verify `requirements.txt` has been installed
- verify the previous notebook stage has completed
- verify `artifact_dir` in `configs/cnnfin_1h.yaml`
- verify `image_builder.ipynb` generated `image_manifest.pkl` before running model notebooks
- verify paths are rooted in the repository, not the notebook directory
