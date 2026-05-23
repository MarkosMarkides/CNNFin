# Jarvis Runbook

This is the intended full workflow for running CNNFin on Jarvis Labs or another GPU machine.

## 1. Clone Or Upload The Project

Start from the cleaned project directory containing:

```text
cnnfin/
configs/
exploration/
feature_engineering/
image_generation/
tests/
requirements.txt
```

Generated artifacts are not required at the start unless you are resuming a previous run.

## 2. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your Jarvis image has a specific CUDA setup, install the matching PyTorch/Torchvision wheels first, then install `requirements.txt`.

## 3. Validate Setup

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

All tests should pass before launching long jobs.

## 4. Download Data

Open and run:

```text
exploration/data_fetching.ipynb
```

Keep:

```python
FORCE_DOWNLOAD = False
```

unless you intentionally want to redownload existing files.

Expected output:

```text
artifacts/cnnfin_1h/raw_candles/
```

## 5. Build Dataset And Images

Open:

```text
exploration/image_builder.ipynb
```

First run with:

```python
RUN_FULL_IMAGE_BUILD = False
```

If this workspace has old V1 triple-barrier artifacts, also set this once before rebuilding:

```python
CLEAN_DERIVED_ARTIFACTS = True
```

This removes `processed/`, `reports/`, `image_preview/`, `images/`, `results/`, and `models/`, while keeping `raw_candles/` and `raw_zips/`.

Inspect the preview images.

When ready for full training, set:

```python
RUN_FULL_IMAGE_BUILD = True
```

Expected full output:

```text
artifacts/cnnfin_1h/processed/merged_df.pkl
artifacts/cnnfin_1h/processed/samples.pkl
artifacts/cnnfin_1h/processed/image_manifest.pkl
artifacts/cnnfin_1h/images/
```

## 6. Train Numeric Baselines

Open and run:

```text
exploration/ML_models_regularized_variants.ipynb
```

Keep:

```python
DEBUG_MODE = False
```

The notebook trains Logistic Regression, XGBoost, MLP, and LSTM on the same sample IDs and same raw source information used by the CNN image renderer. It does not run window-normalized variants.

Expected outputs:

```text
artifacts/cnnfin_1h/results/logistic_regression/
artifacts/cnnfin_1h/results/xgboost/
artifacts/cnnfin_1h/results/mlp/
artifacts/cnnfin_1h/results/numeric_lstm/
artifacts/cnnfin_1h/results/ml_model_raw_summary.pkl
```

## 7. Train CNN

Open and run:

```text
exploration/cnn_builder.ipynb
```

Keep:

```python
DEBUG_MODE = False
```

The notebook trains EfficientNet-B0 with staged fine-tuning and evaluates the test set once after model selection.

Expected outputs:

```text
artifacts/cnnfin_1h/models/efficientnet_b0_highres.pt
artifacts/cnnfin_1h/results/efficientnet_b0_highres/
```

## 8. Collect Results

Use the saved metrics files to build paper tables.

Primary result:

```text
test macro-F1
```

Useful files:

```text
artifacts/cnnfin_1h/results/*/test_metrics.json
artifacts/cnnfin_1h/results/*/test_predictions.pkl
artifacts/cnnfin_1h/results/*/test_confusion_matrix.pkl
```

## 9. Smoke Test Option

Before full runs, both model notebooks support:

```python
DEBUG_MODE = True
```

Use this only to check that the notebook executes. Do not use debug outputs for research conclusions.
