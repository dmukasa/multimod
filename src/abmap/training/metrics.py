"""Evaluation utilities for ABMAP training."""
from __future__ import annotations

from typing import Dict

import torch


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))
    return _pearsonr(x_rank.float(), y_rank.float())


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    pearson = _pearsonr(predictions, targets)
    spearman = _spearmanr(predictions, targets)
    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
    }
