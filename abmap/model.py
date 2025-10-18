"""Lightweight binary logistic regression model implemented without NumPy."""

from __future__ import annotations

import math
from typing import List, Sequence


def _sigmoid(value: float) -> float:
    # Guard against floating point overflow when ``value`` is large.
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, n_features: int) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.weights: List[float] = [0.0 for _ in range(n_features)]
        self.bias: float = 0.0

    @property
    def n_features(self) -> int:
        return len(self.weights)

    def predict_logit(self, features: Sequence[float]) -> float:
        if len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features but received {len(features)}"
            )
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def predict_proba(self, features: Sequence[float]) -> float:
        return _sigmoid(self.predict_logit(features))

    def predict(self, features: Sequence[float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0

    def update_batch(
        self,
        batch_features: Sequence[Sequence[float]],
        batch_labels: Sequence[int],
        learning_rate: float,
    ) -> float:
        if len(batch_features) != len(batch_labels):
            raise ValueError("Features and labels batch sizes must match")
        if not batch_features:
            raise ValueError("Cannot update model with an empty batch")
        gradients = [0.0 for _ in range(self.n_features)]
        bias_gradient = 0.0
        loss = 0.0
        eps = 1e-12

        for features, label in zip(batch_features, batch_labels):
            probability = self.predict_proba(features)
            error = probability - label
            for index, value in enumerate(features):
                gradients[index] += error * value
            bias_gradient += error
            loss -= label * math.log(probability + eps) + (1 - label) * math.log(1 - probability + eps)

        batch_size = float(len(batch_features))
        for index in range(self.n_features):
            self.weights[index] -= learning_rate * gradients[index] / batch_size
        self.bias -= learning_rate * bias_gradient / batch_size

        return loss / batch_size

    def to_dict(self) -> dict:
        return {"weights": list(self.weights), "bias": self.bias}

    @classmethod
    def from_dict(cls, payload: dict) -> "LogisticRegression":
        weights = payload.get("weights")
        bias = payload.get("bias")
        if weights is None or bias is None:
            raise ValueError("Serialized model must include 'weights' and 'bias'")
        model = cls(len(weights))
        model.weights = [float(value) for value in weights]
        model.bias = float(bias)
        return model
