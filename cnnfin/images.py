from __future__ import annotations

from pathlib import Path

import pandas as pd

from cnnfin.config import ExperimentConfig
from cnnfin.features import load_feature_table
from cnnfin.samples import load_samples
from cnnfin.utils import ensure_dir, write_json
from image_generation.image_generator import ImageGenerator


def generate_images(config: ExperimentConfig, *, force: bool = False) -> pd.DataFrame:
    features = load_feature_table(config)
    samples = load_samples(config)
    image_lookback = config.image_lookback or config.lookback

    missing = [c for c in config.image_indicators if c not in features.columns]
    if missing:
        raise ValueError(f"Missing image indicator columns: {missing}")

    image_root = ensure_dir(config.artifact_path("images"))
    generator = ImageGenerator(window_size=image_lookback, target_symbol=config.target_symbol)
    manifest_rows: list[dict[str, object]] = []

    for split in ["train", "val", "test"]:
        ensure_dir(image_root / split)

    for n, (_, row) in enumerate(samples.iterrows(), start=1):
        split = str(row["split"])
        sample_id = str(row["sample_id"])
        row_idx = int(row["row_idx"])
        start_idx = row_idx - image_lookback + 1
        if start_idx < 0:
            continue
        image_path = image_root / split / f"{sample_id}.png"
        if force or not image_path.exists():
            generator.save_four_panel_image(
                features,
                start_idx=start_idx,
                indicators=config.image_indicators,
                filepath=str(image_path),
                size_each=config.image_size_each,
                btc_close_col="Close",
                alt_suffix="_Close",
            )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "Open time": row["Open time"],
                "row_idx": row_idx,
                "split": split,
                "label": int(row["label"]),
                "image_lookback": image_lookback,
                "image_path": str(image_path),
            }
        )
        if n % 5000 == 0:
            print(f"Generated/verified {n} images")

    manifest = pd.DataFrame(manifest_rows)
    processed_dir = ensure_dir(config.artifact_path("processed"))
    manifest.to_pickle(processed_dir / "image_manifest.pkl")
    stale_csv = processed_dir / "image_manifest.csv"
    if stale_csv.exists():
        stale_csv.unlink()
    write_json(
        {
            "rows": int(len(manifest)),
            "split_counts": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
            "image_lookback": int(image_lookback),
            "image_root": str(image_root),
        },
        config.artifact_path("reports", "image_report.json"),
    )
    return manifest


def load_image_manifest(config: ExperimentConfig) -> pd.DataFrame:
    pkl_path = config.artifact_path("processed", "image_manifest.pkl")
    csv_path = config.artifact_path("processed", "image_manifest.csv")
    if pkl_path.exists():
        manifest = pd.read_pickle(pkl_path)
    elif csv_path.exists():
        manifest = pd.read_csv(csv_path)
    else:
        return generate_images(config)
    manifest["Open time"] = pd.to_datetime(manifest["Open time"], utc=True)
    return manifest
