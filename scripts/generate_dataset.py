"""Generate a synthetic dataset compatible with the simplified ABMAP pipeline."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from typing import List


DEFAULT_WEIGHTS = [1.6, -2.2, 0.9, 3.1, -1.4, 0.7, -0.5, 1.8]
DEFAULT_BIAS = -0.35


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def generate_dataset(
    *,
    samples: int,
    seed: int,
    noise: float,
    output_path: str,
) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = [f"feature_{index+1}" for index in range(len(DEFAULT_WEIGHTS))]
    header.append("label")

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for _ in range(samples):
            features: List[float] = [rng.gauss(0.0, 1.0) for _ in DEFAULT_WEIGHTS]
            linear = sum(weight * value for weight, value in zip(DEFAULT_WEIGHTS, features)) + DEFAULT_BIAS
            linear += rng.gauss(0.0, noise)
            probability = _sigmoid(linear)
            label = 1 if rng.random() < probability else 0
            writer.writerow([f"{value:.6f}" for value in features] + [label])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed controlling generation")
    parser.add_argument("--noise", type=float, default=0.6, help="Standard deviation of Gaussian noise")
    parser.add_argument(
        "--output",
        type=str,
        default="data/main_dataset.csv",
        help="Destination CSV file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(samples=args.samples, seed=args.seed, noise=args.noise, output_path=args.output)


if __name__ == "__main__":
    main()
