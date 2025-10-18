"""Evaluation helpers for the simplified ABMAP pipeline."""

from __future__ import annotations

import math
from typing import Sequence


def binary_cross_entropy(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have the same length")
    eps = 1e-12
    total = 0.0
    for probability, label in zip(probabilities, labels):
        total -= label * math.log(probability + eps) + (1 - label) * math.log(1 - probability + eps)
    return total / float(len(probabilities)) if probabilities else 0.0


def accuracy(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have the same length")
    if not probabilities:
        return 0.0
    correct = 0
    for probability, label in zip(probabilities, labels):
        prediction = 1 if probability >= 0.5 else 0
        if prediction == label:
            correct += 1
    return correct / float(len(probabilities))
