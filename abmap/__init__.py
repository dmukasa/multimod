"""Simplified implementation of the ABMAP training stack.

This module provides utilities for loading tabular datasets, training a
logistic regression model using batch gradient descent, and computing
basic evaluation metrics.  It is a lightweight, dependency-free stand-in
for the original repository's learning pipeline so it can operate in a
restricted execution environment.
"""

from .data import load_csv_dataset, train_val_split
from .model import LogisticRegression
from .training import train_logistic_regression
from . import evaluation

__all__ = [
    "load_csv_dataset",
    "train_val_split",
    "LogisticRegression",
    "train_logistic_regression",
    "evaluation",
]
