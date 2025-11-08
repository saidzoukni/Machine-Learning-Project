"""
Train a simple logistic regression model for vehicle classification
Inputs/features:
 - height (meters)
 - number_of_wheels

Target/classes:
 - 0: Camion (truck)
 - 1: Touristique (passenger)

The decision boundary follows intuitive rules with some noise:
 - Camion more likely when wheels >= 8 or height >= 3.8m
 - Touristique otherwise

Saves model to models_ai/logreg_model.pkl (pickle format)
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def generate_dataset(num_samples: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    possible_wheels = [4, 6, 8, 10, 12, 14, 16, 18]
    heights = np_rng.uniform(2.0, 5.5, size=num_samples)
    wheels = np.array([rng.choice(possible_wheels) for _ in range(num_samples)], dtype=float)

    # Base rule with noise
    logits = (
        1.6 * (heights - 3.8) + 0.5 * ((wheels - 8.0) / 2.0)
    )  # positive => Camion, negative => Touristique
    noise = np_rng.normal(0.0, 0.6, size=num_samples)
    logits = logits + noise
    probs_camion = 1 / (1 + np.exp(-logits))
    y_camion = (probs_camion > 0.5).astype(int)  # 1 => Camion

    # Map to labels required by views: 0 Camion, 1 Touristique
    # y_camion==1 means Camion => label 0; else 1
    y = np.where(y_camion == 1, 0, 1)

    X = np.column_stack([heights, wheels])
    return X.astype(float), y.astype(int)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save(model_path: str | os.PathLike[str]) -> None:
    X, y = generate_dataset()
    model = build_model()
    model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "logreg_model.pkl"
    train_and_save(out_path)


