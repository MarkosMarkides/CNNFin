from __future__ import annotations

import argparse

from cnnfin.config import load_config, save_config
from cnnfin.data import align_market_data, download_raw_candles
from cnnfin.features import build_feature_table
from cnnfin.images import generate_images
from cnnfin.metrics import collect_test_results, write_cnn_vs_best_baseline
from cnnfin.models import train_image_cnn, train_numeric_lstm, train_tabular_baselines
from cnnfin.samples import build_samples
from cnnfin.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CNNFin image-vs-tabular experiment.")
    parser.add_argument("--config", default="configs/cnnfin_1h.yaml", help="Path to experiment YAML config.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["download", "dataset", "images", "tabular", "lstm", "cnn", "train", "evaluate", "all", "smoke"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--artifact-dir", default=None, help="Override artifact directory.")
    parser.add_argument("--start-date", default=None, help="Override start date.")
    parser.add_argument("--end-date", default=None, help="Override end date.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Limit samples per split for smoke/debug runs.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override torch training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override torch training batch size.")
    parser.add_argument("--bootstrap-iterations", type=int, default=None, help="Override bootstrap iterations.")
    parser.add_argument("--force-download", action="store_true", help="Redownload raw Binance candles.")
    parser.add_argument("--force-images", action="store_true", help="Regenerate images even if present.")
    return parser.parse_args()


def build_dataset(config) -> None:
    align_market_data(config)
    build_feature_table(config)
    build_samples(config)


def run_stage(config, stage: str, *, force_download: bool = False, force_images: bool = False) -> None:
    if stage == "download":
        download_raw_candles(config, force=force_download)
        return
    if stage == "dataset":
        build_dataset(config)
        return
    if stage == "images":
        generate_images(config, force=force_images)
        return
    if stage == "tabular":
        train_tabular_baselines(config)
        return
    if stage == "lstm":
        train_numeric_lstm(config)
        return
    if stage == "cnn":
        generate_images(config, force=force_images)
        train_image_cnn(config)
        return
    if stage == "train":
        train_tabular_baselines(config)
        train_numeric_lstm(config)
        generate_images(config, force=force_images)
        train_image_cnn(config)
        return
    if stage == "evaluate":
        collect_test_results(config)
        write_cnn_vs_best_baseline(config)
        return
    if stage == "all":
        download_raw_candles(config, force=force_download)
        build_dataset(config)
        generate_images(config, force=force_images)
        train_tabular_baselines(config)
        train_numeric_lstm(config)
        train_image_cnn(config)
        collect_test_results(config)
        write_cnn_vs_best_baseline(config)
        return
    if stage == "smoke":
        config.max_samples_per_split = config.max_samples_per_split or 64
        config.num_epochs = min(config.num_epochs, 1)
        config.bootstrap_iterations = min(config.bootstrap_iterations, 50)
        download_raw_candles(config, force=force_download)
        build_dataset(config)
        generate_images(config, force=force_images)
        train_tabular_baselines(config)
        train_numeric_lstm(config)
        train_image_cnn(config)
        collect_test_results(config)
        write_cnn_vs_best_baseline(config)
        return
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config,
        artifact_dir=args.artifact_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_samples_per_split=args.max_samples_per_split,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        cnn_batch_size=args.batch_size,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    ensure_dir(config.artifact_path())
    save_config(config, config.artifact_path("experiment_config.yaml"))
    run_stage(config, args.stage, force_download=args.force_download, force_images=args.force_images)


if __name__ == "__main__":
    main()
