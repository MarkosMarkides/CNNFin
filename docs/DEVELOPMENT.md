# Development

This document covers local development, tests, smoke checks, and common failure modes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tests

Run all tests:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Current tests cover:

- triple-barrier label cases
- no future leakage in feature rows
- image generator target-symbol exclusion
- synthetic pipeline smoke build

## Notebook Smoke Checks

The model notebooks include debug modes:

```python
DEBUG_MODE = True
```

Use debug mode to verify dataloaders, forward passes, metrics, and artifact saving on a tiny subset.

Set debug mode back to:

```python
DEBUG_MODE = False
```

before full Jarvis training.

## Image Build Toggle

`image_builder.ipynb` defaults to:

```python
RUN_FULL_IMAGE_BUILD = False
```

This keeps local runs lightweight.

Set it to `True` only when full image generation is intended.

## Dependency Notes

The core dependencies are in:

```text
requirements.txt
requirements-cnnfin.txt
```

On GPU machines, PyTorch installation may depend on the CUDA version of the base image.

## Common Failure Modes

### Missing `image_manifest.pkl`

Model notebooks require:

```text
artifacts/cnnfin_5m/processed/image_manifest.pkl
```

If it is missing, run `image_builder.ipynb` with:

```python
RUN_FULL_IMAGE_BUILD = True
```

### Missing Raw Candles

If dataset building cannot find raw candles, run:

```text
exploration/data_fetching.ipynb
```

### Wrong Working Directory

The notebooks search upward for:

```text
configs/cnnfin_5m.yaml
```

They should work from the repo root or from inside `exploration/`.

### Out Of Memory During Full Image Or Model Runs

Reduce temporary batch sizes or run on a larger Jarvis instance.

For model notebooks, use `DEBUG_MODE = True` only to verify execution, not to produce final metrics.

### Unexpected Result Differences

Check:

- `configs/cnnfin_5m.yaml`
- `artifacts/cnnfin_5m/experiment_config.yaml`
- `artifacts/cnnfin_5m/processed/feature_columns.json`
- random seed in config
- whether images were regenerated after feature/config changes

## Source Of Truth

Implementation source of truth:

- `cnnfin/`
- `feature_engineering/indicators.py`
- `image_generation/image_generator.py`

Execution source of truth for Jarvis:

- `exploration/data_fetching.ipynb`
- `exploration/image_builder.ipynb`
- `exploration/ML_models.ipynb`
- `exploration/cnn_builder.ipynb`

