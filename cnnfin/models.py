from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cnnfin.config import ExperimentConfig
from cnnfin.features import load_feature_table
from cnnfin.images import load_image_manifest
from cnnfin.metrics import LABELS, bootstrap_macro_f1_ci, evaluate_predictions, save_model_outputs
from cnnfin.samples import load_split_samples
from cnnfin.utils import ensure_dir, read_json, set_global_seed, write_json


def _load_feature_columns(config: ExperimentConfig) -> dict[str, list[str]]:
    return read_json(config.artifact_path("processed", "feature_columns.json"))


def _split_samples(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    return {split: load_split_samples(config, split).copy() for split in ["train", "val", "test"]}


def _model_lookback(config: ExperimentConfig) -> int:
    return int(config.image_lookback or config.lookback)


def _flatten_sample_windows(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    columns: list[str],
    lookback: int,
) -> np.ndarray:
    values = features[columns].to_numpy(dtype=np.float32)
    row_idx = samples["row_idx"].astype(int).to_numpy()
    out = np.empty((len(samples), lookback * len(columns)), dtype=np.float32)
    for i, end in enumerate(row_idx):
        start = int(end) - lookback + 1
        out[i] = values[start : int(end) + 1].reshape(-1)
    return out


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=3).astype(float)
    total = counts.sum()
    weights = np.ones(3, dtype=np.float32)
    for cls in LABELS:
        if counts[cls] > 0:
            weights[cls] = total / (len(LABELS) * counts[cls])
    return weights


def _tabular_split_arrays(config: ExperimentConfig) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray]]:
    features = load_feature_table(config)
    columns = _load_feature_columns(config)["tabular_feature_cols"]
    lookback = _model_lookback(config)
    splits = _split_samples(config)
    xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    for split, samples in splits.items():
        xs[split] = _flatten_sample_windows(features, samples, columns, lookback)
        ys[split] = samples["label"].astype(int).to_numpy()
    return splits, xs, ys


def _prediction_frame(samples: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> pd.DataFrame:
    out = samples[["sample_id", "Open time", "split"]].copy().reset_index(drop=True)
    out["y_true"] = y_true.astype(int)
    out["y_pred"] = y_pred.astype(int)
    if y_proba is not None:
        for cls in LABELS:
            out[f"p_{cls}"] = y_proba[:, cls]
    return out


def _save_sklearn_model(config: ExperimentConfig, model_name: str, model) -> None:
    out_dir = ensure_dir(config.artifact_path("models"))
    joblib.dump(model, out_dir / f"{model_name}.joblib")


def train_tabular_baselines(config: ExperimentConfig) -> None:
    splits, xs, ys = _tabular_split_arrays(config)
    class_weight = {i: float(w) for i, w in enumerate(_class_weights(ys["train"]))}

    model_defs = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.seeds[0],
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=config.seeds[0],
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=512,
                        learning_rate_init=1e-3,
                        max_iter=100,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=config.seeds[0],
                    ),
                ),
            ]
        ),
    }

    if "xgboost" in config.models:
        try:
            from xgboost import XGBClassifier

            model_defs["xgboost"] = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=3,
                            eval_metric="mlogloss",
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.03,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=config.seeds[0],
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            write_json(
                {"status": "skipped", "reason": f"xgboost unavailable: {exc}"},
                config.artifact_path("results", "xgboost", "test_metrics.json"),
            )

    for model_name, model in model_defs.items():
        if model_name not in config.models:
            continue
        print(f"Training {model_name}")
        fit_kwargs = {}
        if model_name == "xgboost":
            sample_weight = np.array([class_weight[int(y)] for y in ys["train"]], dtype=np.float32)
            fit_kwargs["model__sample_weight"] = sample_weight
        model.fit(xs["train"], ys["train"], **fit_kwargs)
        _save_sklearn_model(config, model_name, model)

        for split in ["val", "test"]:
            y_pred = model.predict(xs[split]).astype(int)
            y_proba = model.predict_proba(xs[split]) if hasattr(model, "predict_proba") else None
            metrics = evaluate_predictions(ys[split], y_pred, class_names=config.class_names, y_proba=y_proba)
            metrics["macro_f1_ci"] = bootstrap_macro_f1_ci(
                ys[split],
                y_pred,
                iterations=config.bootstrap_iterations if split == "test" else 200,
                seed=config.seeds[0],
            )
            predictions = _prediction_frame(splits[split], ys[split], y_pred, y_proba)
            save_model_outputs(config, model_name, split, predictions, metrics)


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _torch_class_weights(y: np.ndarray, device: str):
    import torch

    return torch.tensor(_class_weights(y), dtype=torch.float32, device=device)


def train_numeric_lstm(config: ExperimentConfig) -> None:
    if "numeric_lstm" not in config.models:
        return
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    set_global_seed(config.seeds[0])
    features = load_feature_table(config)
    columns = _load_feature_columns(config)["sequence_feature_cols"]
    lookback = _model_lookback(config)
    splits = _split_samples(config)

    train_year_mask = pd.to_datetime(features["Open time"], utc=True).dt.year.isin(config.train_years)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_feature_rows = features.loc[train_year_mask, columns].to_numpy(dtype=np.float32)
    imputer.fit(train_feature_rows)
    scaler.fit(imputer.transform(train_feature_rows))
    all_x = scaler.transform(imputer.transform(features[columns].to_numpy(dtype=np.float32))).astype(np.float32)

    class SequenceDataset(Dataset):
        def __init__(self, samples: pd.DataFrame):
            self.samples = samples.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            row = self.samples.iloc[idx]
            end = int(row["row_idx"])
            start = end - lookback + 1
            x = all_x[start : end + 1]
            y = int(row["label"])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    class NumericLSTM(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_size=128, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    device = _torch_device()
    train_ds = SequenceDataset(splits["train"])
    val_ds = SequenceDataset(splits["val"])
    test_ds = SequenceDataset(splits["test"])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = NumericLSTM(input_dim=len(columns)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=_torch_class_weights(splits["train"]["label"].to_numpy(), device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_state = None
    best_score = -1.0
    bad_epochs = 0
    for epoch in range(config.num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_true, val_pred, _ = _predict_torch(model, val_loader, device)
        val_macro = f1_score(val_true, val_pred, labels=LABELS, average="macro", zero_division=0)
        print(f"numeric_lstm epoch={epoch + 1} val_macro_f1={val_macro:.4f}")
        if val_macro > best_score:
            best_score = val_macro
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ensure_dir(config.artifact_path("models"))
    torch.save({"model_state": model.state_dict(), "columns": columns}, config.artifact_path("models", "numeric_lstm.pt"))
    joblib.dump({"imputer": imputer, "scaler": scaler, "columns": columns}, config.artifact_path("models", "numeric_lstm_preprocess.joblib"))

    for split, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_pred, y_proba = _predict_torch(model, loader, device)
        metrics = evaluate_predictions(y_true, y_pred, class_names=config.class_names, y_proba=y_proba)
        metrics["macro_f1_ci"] = bootstrap_macro_f1_ci(
            y_true,
            y_pred,
            iterations=config.bootstrap_iterations if split == "test" else 200,
            seed=config.seeds[0],
        )
        predictions = _prediction_frame(splits[split], y_true, y_pred, y_proba)
        save_model_outputs(config, "numeric_lstm", split, predictions, metrics)


def _predict_torch(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            ys.append(yb.cpu().numpy())
            preds.append(proba.argmax(axis=1))
            probas.append(proba)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probas)


def train_image_cnn(config: ExperimentConfig) -> None:
    if "efficientnet_b0" not in config.models:
        return
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    set_global_seed(config.seeds[0])
    manifest = load_image_manifest(config)
    splits = {split: manifest[manifest["split"] == split].copy().reset_index(drop=True) for split in ["train", "val", "test"]}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.cnn_input_size, config.cnn_input_size)),
            transforms.RandomAffine(degrees=0, translate=(0.01, 0.01), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.cnn_input_size, config.cnn_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    class ImageDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform):
            self.frame = frame.reset_index(drop=True)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int):
            row = self.frame.iloc[idx]
            with Image.open(row["image_path"]) as img:
                x = self.transform(img.convert("RGB"))
            return x, torch.tensor(int(row["label"]), dtype=torch.long)

    device = _torch_device()
    train_ds = ImageDataset(splits["train"], train_transform)
    val_ds = ImageDataset(splits["val"], eval_transform)
    test_ds = ImageDataset(splits["test"], eval_transform)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception as exc:
        print(f"[WARN] Failed to load EfficientNet-B0 ImageNet weights ({exc}); using random initialization.")
        model = efficientnet_b0(weights=None)
    if config.freeze_cnn_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=_torch_class_weights(splits["train"]["label"].to_numpy(), device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    best_state = None
    best_score = -1.0
    bad_epochs = 0
    for epoch in range(config.num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_true, val_pred, _ = _predict_torch(model, val_loader, device)
        val_macro = f1_score(val_true, val_pred, labels=LABELS, average="macro", zero_division=0)
        print(f"efficientnet_b0 epoch={epoch + 1} val_macro_f1={val_macro:.4f}")
        if val_macro > best_score:
            best_score = val_macro
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ensure_dir(config.artifact_path("models"))
    torch.save({"model_state": model.state_dict(), "config": config.to_dict()}, config.artifact_path("models", "efficientnet_b0.pt"))

    for split, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_pred, y_proba = _predict_torch(model, loader, device)
        metrics = evaluate_predictions(y_true, y_pred, class_names=config.class_names, y_proba=y_proba)
        metrics["macro_f1_ci"] = bootstrap_macro_f1_ci(
            y_true,
            y_pred,
            iterations=config.bootstrap_iterations if split == "test" else 200,
            seed=config.seeds[0],
        )
        predictions = _prediction_frame(splits[split], y_true, y_pred, y_proba)
        save_model_outputs(config, "efficientnet_b0", split, predictions, metrics)
