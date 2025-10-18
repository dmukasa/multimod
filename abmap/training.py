"""Training utilities for the simplified ABMAP pipeline."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence, Tuple

from .evaluation import accuracy, binary_cross_entropy
from .model import LogisticRegression


def _batched(sequence: Sequence[Sequence[float]], labels: Sequence[int], batch_size: int):
    for start in range(0, len(sequence), batch_size):
        end = min(start + batch_size, len(sequence))
        yield sequence[start:end], labels[start:end]


def train_logistic_regression(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    val_ratio: float = 0.2,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> Tuple[LogisticRegression, List[Dict[str, float]]]:
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same length")
    if not features:
        raise ValueError("Cannot train on an empty dataset")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    total_samples = len(features)
    index_sequence = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(index_sequence)

    split_index = int(total_samples * (1.0 - val_ratio))
    train_indices = index_sequence[:split_index]
    val_indices = index_sequence[split_index:]
    if not train_indices or not val_indices:
        raise ValueError("val_ratio produced an empty train or validation set")

    train_features = [list(features[i]) for i in train_indices]
    train_labels = [int(labels[i]) for i in train_indices]
    val_features = [list(features[i]) for i in val_indices]
    val_labels = [int(labels[i]) for i in val_indices]

    model = LogisticRegression(len(train_features[0]))
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_features = list(train_features)
        epoch_labels = list(train_labels)
        combined = list(zip(epoch_features, epoch_labels))
        rng.shuffle(combined)
        epoch_features = [list(item[0]) for item in combined]
        epoch_labels = [int(item[1]) for item in combined]

        for batch_features, batch_labels in _batched(epoch_features, epoch_labels, batch_size):
            model.update_batch(batch_features, batch_labels, learning_rate)

        train_probabilities = [model.predict_proba(row) for row in train_features]
        val_probabilities = [model.predict_proba(row) for row in val_features]

        train_loss = binary_cross_entropy(train_probabilities, train_labels)
        val_loss = binary_cross_entropy(val_probabilities, val_labels)
        train_acc = accuracy(train_probabilities, train_labels)
        val_acc = accuracy(val_probabilities, val_labels)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

    return model, history


def save_training_artifacts(
    model: LogisticRegression,
    history: Sequence[Dict[str, float]],
    *,
    model_path: str,
    history_path: str,
) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2, sort_keys=True)

    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(list(history), handle, indent=2, sort_keys=True)
