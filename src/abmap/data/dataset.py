"""Dataset definitions for ABMAP training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .tokenizer import AminoAcidTokenizer


@dataclass
class TargetNormalizer:
    mean: float
    std: float

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


class AbmapDataset(Dataset):
    """Torch dataset representing heavy/light chain pairs with antigen targets."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AminoAcidTokenizer,
        *,
        max_heavy_length: int,
        max_light_length: int,
        max_antigen_length: int,
        normalizer: TargetNormalizer,
    ) -> None:
        self.tokenizer = tokenizer
        self.frame = frame.reset_index(drop=True)
        self.normalizer = normalizer
        self.max_heavy_length = max_heavy_length
        self.max_light_length = max_light_length
        self.max_antigen_length = max_antigen_length

        if "light_chain" not in self.frame:
            self.frame["light_chain"] = ""

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        heavy = row["heavy_chain"]
        light = row.get("light_chain", "")
        antigen = row["antigen_sequence"]
        target = torch.tensor(row["binding_score"], dtype=torch.float32)

        heavy_tokens = self.tokenizer.batch_encode([heavy], max_length=self.max_heavy_length)[0]
        light_tokens = self.tokenizer.batch_encode([light], max_length=self.max_light_length)[0]
        antigen_tokens = self.tokenizer.batch_encode([antigen], max_length=self.max_antigen_length)[0]
        normalized_target = self.normalizer.encode(target)

        return {
            "heavy": heavy_tokens,
            "light": light_tokens,
            "antigen": antigen_tokens,
            "target": normalized_target,
        }


def load_processed_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported processed dataset format: {path.suffix}")


def compute_target_normalizer(frame: pd.DataFrame) -> TargetNormalizer:
    values = frame["binding_score"].astype(float).to_numpy()
    mean = float(values.mean())
    std = float(values.std() or 1.0)
    return TargetNormalizer(mean=mean, std=std)


def create_datasets(
    frame: pd.DataFrame,
    tokenizer: AminoAcidTokenizer,
    *,
    validation_fraction: float,
    random_seed: int,
    max_lengths: Tuple[int, int, int],
) -> Tuple[AbmapDataset, AbmapDataset, TargetNormalizer]:
    train_frame, val_frame = train_test_split(
        frame,
        test_size=validation_fraction,
        random_state=random_seed,
        shuffle=True,
    )
    normalizer = compute_target_normalizer(train_frame)
    max_heavy, max_light, max_antigen = max_lengths

    train_dataset = AbmapDataset(
        train_frame,
        tokenizer,
        max_heavy_length=max_heavy,
        max_light_length=max_light,
        max_antigen_length=max_antigen,
        normalizer=normalizer,
    )
    val_dataset = AbmapDataset(
        val_frame,
        tokenizer,
        max_heavy_length=max_heavy,
        max_light_length=max_light,
        max_antigen_length=max_antigen,
        normalizer=normalizer,
    )
    return train_dataset, val_dataset, normalizer
