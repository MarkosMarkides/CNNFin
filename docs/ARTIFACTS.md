# Artifacts

All generated experiment outputs live under:

```text
artifacts/cnnfin_5m/
```

The artifact directory is configured in:

```text
configs/cnnfin_5m.yaml
```

## Top-Level Artifact Map

```text
artifacts/cnnfin_5m/
  raw_candles/
  raw_zips/
  processed/
  image_preview/
  images/
  models/
  results/
  reports/
  experiment_config.yaml
```

## `raw_candles/`

Contains one pickle per symbol:

```text
{SYMBOL}_5m.pkl
```

These are downloaded by `data_fetching.ipynb`.

## `raw_zips/`

Contains cached Binance ZIP downloads used during data fetching.

This directory is useful for avoiding repeated network downloads.

## `processed/`

Important processed files:

```text
aligned_5m.pkl
merged_df.pkl
features_5m.pkl
samples.pkl
train_samples.pkl
val_samples.pkl
test_samples.pkl
train_val_samples.pkl
feature_columns.json
image_manifest.pkl
```

`merged_df.pkl` is the full auditable dataset.

`samples.pkl` is the valid modeling subset.

`image_manifest.pkl` is created after full image generation and is the sample source for the final ML and CNN notebooks.

## `image_preview/`

Contains balanced preview images for visual inspection.

The preview manifest is:

```text
image_preview/preview_manifest.pkl
```

These images are not the final training dataset.

## `images/`

Contains full CNN training/evaluation images:

```text
images/train/{sample_id}.png
images/val/{sample_id}.png
images/test/{sample_id}.png
```

This directory is created only when `RUN_FULL_IMAGE_BUILD = True`.

## `models/`

Contains trained model artifacts, for example:

```text
logistic_regression.joblib
xgboost.joblib
mlp.pt
numeric_lstm.pt
efficientnet_b0.pt
```

Preprocessors can also be stored here, such as:

```text
mlp_preprocess.joblib
numeric_lstm_preprocess.joblib
```

## `results/`

Each model gets a subdirectory:

```text
results/{model_name}/
```

Common files include:

```text
val_predictions.pkl
test_predictions.pkl
val_metrics.json
test_metrics.json
test_confusion_matrix.pkl
training_history.pkl
```

The ML baseline notebook also writes:

```text
results/ml_model_summary.pkl
```

## `reports/`

Reports are lightweight JSON files written by data and dataset stages.

Examples:

```text
download_report.json
data_integrity_report.json
merged_df_report.json
sample_report.json
image_report.json
```

Reports are intended for sanity checks and reproducibility notes.

