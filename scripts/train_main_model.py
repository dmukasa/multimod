"""Train the simplified ABMAP model on the full synthetic dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from abmap.data import load_csv_dataset
from abmap.training import save_training_artifacts, train_logistic_regression


DEFAULT_DATASET = "data/main_dataset.csv"
DEFAULT_MODEL_PATH = "artifacts/main_model.json"
DEFAULT_HISTORY_PATH = "artifacts/main_training_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to the training CSV dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs to run")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=0.15, help="Gradient descent learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data reserved for validation")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for shuffling")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Where to store the trained model")
    parser.add_argument("--history-path", type=str, default=DEFAULT_HISTORY_PATH, help="Where to store training metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels, _ = load_csv_dataset(args.dataset)

    model, history = train_logistic_regression(
        features,
        labels,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)
    save_training_artifacts(model, history, model_path=args.model_path, history_path=args.history_path)

    final_metrics = history[-1]
    print("Training complete")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
