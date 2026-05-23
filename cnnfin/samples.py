from __future__ import annotations

import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.features import load_feature_table
from cnnfin.utils import ensure_dir, write_json


def sample_id_from_timestamp(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%d%H%M")


def split_name_for_year(config: ExperimentConfig, year: int) -> str | None:
    if year in config.train_years:
        return "train"
    if year in config.val_years:
        return "val"
    if year in config.test_years:
        return "test"
    return None


def build_samples(config: ExperimentConfig) -> pd.DataFrame:
    df = load_feature_table(config).copy()
    df["Open time"] = pd.to_datetime(df["Open time"], utc=True)
    if "valid_sample" not in df.columns:
        raise ValueError("merged_df is missing valid_sample. Rebuild the dataset stage.")

    sample_cols = [
        "sample_id",
        "Open time",
        "row_idx",
        "split",
        "label",
        "label_status",
        "future_avg_close",
        "future_avg_log_return",
        "theta_down",
        "theta_up",
        "lookback_continuous",
        "horizon_continuous",
        "lookback_same_split",
        "horizon_same_split",
        "required_features_present",
    ]
    samples = df.loc[df["valid_sample"], sample_cols].copy()
    samples["label"] = samples["label"].astype(int)

    if config.max_samples_per_split is not None:
        parts = []
        for split in ["train", "val", "test"]:
            part = samples[samples["split"] == split]
            if len(part) > config.max_samples_per_split:
                part = part.iloc[:: max(1, len(part) // config.max_samples_per_split)].head(config.max_samples_per_split)
            parts.append(part)
        samples = pd.concat(parts, ignore_index=True).sort_values("Open time").reset_index(drop=True)

    processed_dir = ensure_dir(config.artifact_path("processed"))
    samples.to_pickle(processed_dir / "samples.pkl")
    for split in ["train", "val", "test"]:
        split_samples = samples[samples["split"] == split]
        split_samples.to_pickle(processed_dir / f"{split}_samples.pkl")
    train_val_samples = samples[samples["split"].isin(["train", "val"])].copy()
    train_val_samples.to_pickle(processed_dir / "train_val_samples.pkl")
    for stale_name in ["samples.csv", "train_sample_ids.csv", "val_sample_ids.csv", "test_sample_ids.csv"]:
        stale_path = processed_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    write_json(
        {
            "rows": int(len(samples)),
            "split_counts": {k: int(v) for k, v in samples["split"].value_counts().to_dict().items()},
            "class_counts": {
                split: {str(k): int(v) for k, v in part["label"].value_counts().sort_index().to_dict().items()}
                for split, part in samples.groupby("split")
            },
            "all_samples_have_continuous_lookback": bool(samples["lookback_continuous"].all()),
            "all_samples_have_continuous_horizon": bool(samples["horizon_continuous"].all()),
            "all_samples_stay_within_split_lookback": bool(samples["lookback_same_split"].all()),
            "all_samples_stay_within_split_horizon": bool(samples["horizon_same_split"].all()),
            "all_samples_have_required_features": bool(samples["required_features_present"].all()),
        },
        config.artifact_path("reports", "sample_report.json"),
    )
    return samples


def load_samples(config: ExperimentConfig) -> pd.DataFrame:
    pkl_path = config.artifact_path("processed", "samples.pkl")
    csv_path = config.artifact_path("processed", "samples.csv")
    if pkl_path.exists():
        samples = pd.read_pickle(pkl_path)
    elif csv_path.exists():
        samples = pd.read_csv(csv_path)
    else:
        return build_samples(config)
    required_v2_cols = {"future_avg_close", "future_avg_log_return", "theta_down", "theta_up"}
    if not required_v2_cols.issubset(samples.columns):
        return build_samples(config)
    samples["Open time"] = pd.to_datetime(samples["Open time"], utc=True)
    return samples


def load_split_samples(config: ExperimentConfig, split: str) -> pd.DataFrame:
    if split not in {"train", "val", "test", "train_val"}:
        raise ValueError(f"Unsupported split: {split}")
    path = config.artifact_path("processed", f"{split}_samples.pkl")
    if path.exists():
        samples = pd.read_pickle(path)
        samples["Open time"] = pd.to_datetime(samples["Open time"], utc=True)
        return samples
    samples = load_samples(config)
    if split == "train_val":
        return samples[samples["split"].isin(["train", "val"])].copy()
    return samples[samples["split"] == split].copy()
