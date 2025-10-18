"""Orchestration for reproducing the ABMAP training loop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import AbmapConfig, load_config
from ..data.dataset import AbmapDataset, TargetNormalizer, create_datasets, load_processed_dataset
from ..data.parsing import load_supplementary_tables, save_processed_dataset
from ..data.tokenizer import AminoAcidTokenizer
from ..model.model import AbmapModel, AbmapModelConfig
from .engine import EarlyStoppingConfig, SchedulerConfig, Trainer


def prepare_dataset(config: AbmapConfig) -> pd.DataFrame:
    dataset_cfg = config.dataset
    processed_path = Path(dataset_cfg.get("processed_dataset", "data/processed/abmap.parquet"))
    if processed_path.exists():
        return load_processed_dataset(processed_path)

    raw_dir = Path(dataset_cfg.get("raw_directory", "data/raw"))
    supplementary_files = dataset_cfg.get("supplementary_files")
    if not supplementary_files:
        raise FileNotFoundError(
            "No supplementary files listed in configuration; cannot build processed dataset."
        )
    paths = [raw_dir / entry for entry in supplementary_files]
    frame = load_supplementary_tables(paths)
    save_processed_dataset(frame, processed_path)
    return frame


def build_datasets(
    frame: pd.DataFrame,
    config: AbmapConfig,
    tokenizer: AminoAcidTokenizer,
) -> Tuple[AbmapDataset, AbmapDataset, TargetNormalizer]:
    dataset_cfg = config.dataset
    validation_fraction = float(dataset_cfg.get("validation_fraction", 0.1))
    random_seed = int(dataset_cfg.get("random_seed", 42))
    max_lengths_cfg = dataset_cfg.get("max_sequence_lengths", {})
    max_lengths = (
        int(max_lengths_cfg.get("heavy", 256)),
        int(max_lengths_cfg.get("light", 256)),
        int(max_lengths_cfg.get("antigen", 128)),
    )
    return create_datasets(
        frame,
        tokenizer,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        max_lengths=max_lengths,
    )


def build_model(config: AbmapConfig, tokenizer: AminoAcidTokenizer) -> AbmapModel:
    model_cfg = config.model
    max_lengths_cfg = config.dataset.get("max_sequence_lengths", {})
    model_config = AbmapModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        transformer_layers=int(model_cfg.get("transformer_layers", 4)),
        transformer_heads=int(model_cfg.get("transformer_heads", 8)),
        transformer_ff_dim=int(model_cfg.get("transformer_ff_dim", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_lengths=(
            int(max_lengths_cfg.get("heavy", 256)),
            int(max_lengths_cfg.get("light", 256)),
            int(max_lengths_cfg.get("antigen", 128)),
        ),
        interaction_hidden_dim=int(model_cfg.get("interaction_hidden_dim", 512)),
        use_light_chain=bool(model_cfg.get("use_light_chain", True)),
    )
    return AbmapModel(model_config)


def build_dataloaders(
    train_dataset: AbmapDataset,
    val_dataset: AbmapDataset,
    config: AbmapConfig,
) -> Tuple[DataLoader, DataLoader]:
    training_cfg = config.training
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 4))
    pin_memory = bool(training_cfg.get("pin_memory", True))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def run_training(config_path: Path, output_dir: Path) -> dict:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_dataset(config)
    tokenizer = AminoAcidTokenizer()
    train_dataset, val_dataset, normalizer = build_datasets(frame, config, tokenizer)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, config)
    model = build_model(config, tokenizer)

    training_cfg = config.training
    device_spec = training_cfg.get("device")
    if device_spec in (None, "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-2))
    gradient_clip = float(training_cfg.get("gradient_clip_val", 1.0))
    max_epochs = int(training_cfg.get("max_epochs", 50))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_cfg = training_cfg.get("scheduler")
    scheduler = SchedulerConfig(**scheduler_cfg) if scheduler_cfg else None
    early_cfg = training_cfg.get("early_stopping")
    early_stopping = EarlyStoppingConfig(**early_cfg) if early_cfg else EarlyStoppingConfig()

    trainer = Trainer(
        model,
        normalizer,
        optimizer,
        device=device,
        gradient_clip_val=gradient_clip,
        scheduler_config=scheduler,
        early_stopping=early_stopping,
        save_dir=output_dir / "checkpoints",
    )
    history = trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump({
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_metrics": history.val_metrics,
            "best_metric": trainer.best_metric,
        }, handle, indent=2)

    final_checkpoint = output_dir / "final_checkpoint.pt"
    trainer.save_checkpoint(final_checkpoint)

    return {
        "history_path": str(history_path),
        "checkpoint_path": str(final_checkpoint),
        "best_spearman": trainer.best_metric,
        "epochs_completed": history.epochs,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the ABMAP training pipeline.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where checkpoints and logs will be stored.",
    )
    args = parser.parse_args(argv)

    summary = run_training(args.config, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
