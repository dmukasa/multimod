"""Data loading utilities for the simplified ABMAP pipeline.

The functions in this module avoid any third-party dependencies so the
entire example can run in a limited environment.  Datasets are expected
as CSV files with numeric feature columns and a binary target column.
"""

from __future__ import annotations

import csv
import os
import random
from typing import List, Sequence, Tuple


def load_csv_dataset(path: str) -> Tuple[List[List[float]], List[int], List[str]]:
    """Load a numeric dataset from ``path``.

    Parameters
    ----------
    path:
        Location of the CSV file. The file must include a header row.  All
        columns except the last are treated as float-valued features while
        the last column is interpreted as an integer target label.

    Returns
    -------
    features, labels, header
        ``features`` is a list-of-lists containing the floating point
        feature values, ``labels`` is a list of integer targets, and
        ``header`` provides the column names from the original file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    features: List[List[float]] = []
    labels: List[int] = []
    header: List[str] = []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - guard clause
            raise ValueError("Dataset file is empty") from exc

        if len(header) < 2:
            raise ValueError("Dataset must contain at least one feature column and one target column")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Row {row_number} in {path} has {len(row)} columns but header defines {len(header)}"
                )
            *feature_values, label_value = row
            try:
                feature_floats = [float(value) for value in feature_values]
                label_int = int(float(label_value))
            except ValueError as exc:
                raise ValueError(f"Row {row_number} contains non-numeric data: {row}") from exc

            features.append(feature_floats)
            labels.append(label_int)

    return features, labels, header


def train_val_split(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    val_ratio: float,
    *,
    seed: int = 42,
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    """Split ``features`` and ``labels`` into training and validation sets."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in the interval (0, 1)")

    combined = list(zip(features, labels))
    random.Random(seed).shuffle(combined)
    split_index = int(len(combined) * (1.0 - val_ratio))

    train_pairs = combined[:split_index]
    val_pairs = combined[split_index:]

    train_features = [list(row[0]) for row in train_pairs]
    train_labels = [int(row[1]) for row in train_pairs]
    val_features = [list(row[0]) for row in val_pairs]
    val_labels = [int(row[1]) for row in val_pairs]

    return train_features, train_labels, val_features, val_labels
