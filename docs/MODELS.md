# Models

CNNFin compares image and numeric models on the same prediction task.

## Fairness Rule

All models use the same samples and the same source information.

- CNN uses the PNG image built from a 30-candle source window.
- Logistic Regression, XGBoost, and MLP use the flattened numeric version of that same 30-candle source window.
- LSTM uses the same source window as a sequence.
- All models use sample IDs from `artifacts/cnnfin_5m/processed/image_manifest.pkl`.

This rule is important. Numeric models should not receive additional features that the CNN image did not have.

## Shared Source Window

The numeric source surface is defined in:

```text
artifacts/cnnfin_5m/processed/feature_columns.json
```

Current model window:

```text
30 candles x 26 source features
```

Tabular models receive:

```text
780 flattened values
```

The 26 source features are:

- `Open`
- `High`
- `Low`
- `Close`
- `ADAUSDT_Close`
- `BNBUSDT_Close`
- `ETHUSDT_Close`
- `LINKUSDT_Close`
- `LTCUSDT_Close`
- `SOLUSDT_Close`
- `TRXUSDT_Close`
- `XLMUSDT_Close`
- `XRPUSDT_Close`
- `return_1`
- `return_10`
- `RSI_14`
- `ATR_14`
- `CCI_20`
- `MACD26_12`
- `SIGNAL26_12`
- `HIST26_12`
- `VWAP_14`
- `Volatility_12`
- `%K_14`
- `%D_14`
- `OBV`

## ML Baselines

`exploration/ML_models.ipynb` trains:

- Logistic Regression
- XGBoost
- MLP
- numeric LSTM

Logistic Regression, XGBoost, and MLP use a flattened `30 x 26` numeric vector.

The LSTM uses a `30 x 26` sequence.

All preprocessing is fit on training data only. Validation is used for model selection where applicable. Test metrics are computed after training is complete.

## CNN Model

`exploration/cnn_builder.ipynb` trains EfficientNet-B0.

The CNN uses ImageNet-pretrained EfficientNet-B0 and staged fine-tuning:

1. freeze the backbone and train the classifier head
2. unfreeze the top EfficientNet feature blocks and fine-tune

Model selection uses validation macro-F1.

## Output Artifacts

Model artifacts are stored in:

```text
artifacts/cnnfin_5m/models/
```

Metrics and predictions are stored in:

```text
artifacts/cnnfin_5m/results/{model_name}/
```

Expected model names include:

- `logistic_regression`
- `xgboost`
- `mlp`
- `numeric_lstm`
- `efficientnet_b0`

## Metrics

Each model reports:

- macro-F1
- accuracy
- weighted F1
- per-class precision, recall, and F1
- confusion matrix
- bootstrap 95% confidence interval for test macro-F1

The primary comparison metric is test macro-F1.

