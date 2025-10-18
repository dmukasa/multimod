"""Configuration utilities for ABMAP training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AbmapConfig:
    """Container for all configuration values used across the project."""

    raw_config: Dict[str, Any]

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw_config.get("dataset", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw_config.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw_config.get("training", {})

    def resolve_path(self, key: str) -> Path:
        """Resolve a path inside the dataset configuration section."""

        path = self.dataset.get(key)
        if path is None:
            raise KeyError(f"Missing dataset path entry for '{key}'.")
        return Path(path)


def load_config(path: str | Path) -> AbmapConfig:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return AbmapConfig(raw_config=data)
