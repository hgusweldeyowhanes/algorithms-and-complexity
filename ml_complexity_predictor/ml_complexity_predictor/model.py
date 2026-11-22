"""
Multiclass softmax (logistic) classifier — pure Python, no sklearn/numpy.

Trained on curve-shape features to map timing curves → Big-O labels.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from ml_complexity_predictor.algorithms import CLASSES
from ml_complexity_predictor.features import FEATURE_DIM


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


@dataclass
class ComplexityClassifier:
    """Softmax regression over Big-O classes."""

    classes: tuple[str, ...] = CLASSES
    # weights[k][j] for class k, feature j; bias[k]
    weights: list[list[float]] = field(default_factory=list)
    bias: list[float] = field(default_factory=list)
    trained: bool = False

    def _init_params(self, dim: int, rng: random.Random) -> None:
        k = len(self.classes)
        scale = 0.01
        self.weights = [[rng.uniform(-scale, scale) for _ in range(dim)] for _ in range(k)]
        self.bias = [0.0] * k

    def _logits(self, x: list[float]) -> list[float]:
        out: list[float] = []
        for k in range(len(self.classes)):
            z = self.bias[k] + sum(w * xi for w, xi in zip(self.weights[k], x))
            out.append(z)
        return out

    def predict_proba(self, x: list[float]) -> dict[str, float]:
        if not self.trained:
            raise RuntimeError("model not trained")
        probs = _softmax(self._logits(x))
        return {c: p for c, p in zip(self.classes, probs)}

    def predict(self, x: list[float]) -> tuple[str, float]:
        probs = self.predict_proba(x)
        best = max(probs, key=probs.get)  # type: ignore[arg-type]
        return best, probs[best]

    def fit(
        self,
        X: list[list[float]],
        y: list[str],
        *,
        epochs: int = 400,
        lr: float = 0.15,
        l2: float = 1e-3,
        seed: int = 0,
        batch_size: int = 32,
    ) -> dict[str, float]:
        if len(X) != len(y) or not X:
            raise ValueError("X/y length mismatch or empty")
        dim = len(X[0])
        if dim != FEATURE_DIM:
            # allow flexible dim if features change
            pass
        label_to_i = {c: i for i, c in enumerate(self.classes)}
        for lab in y:
            if lab not in label_to_i:
                raise ValueError(f"unknown label {lab!r}; expected one of {self.classes}")

        rng = random.Random(seed)
        self._init_params(dim, rng)
        k = len(self.classes)
        indices = list(range(len(X)))

        for _epoch in range(epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                # gradients
                gw = [[0.0] * dim for _ in range(k)]
                gb = [0.0] * k
                for idx in batch:
                    x = X[idx]
                    ti = label_to_i[y[idx]]
                    probs = _softmax(self._logits(x))
                    for c in range(k):
                        err = probs[c] - (1.0 if c == ti else 0.0)
                        gb[c] += err
                        for j in range(dim):
                            gw[c][j] += err * x[j]
                m = float(len(batch))
                for c in range(k):
                    self.bias[c] -= lr * (gb[c] / m)
                    for j in range(dim):
                        self.weights[c][j] -= lr * (gw[c][j] / m + l2 * self.weights[c][j])

        self.trained = True
        return self.evaluate(X, y)  # noqa: trained already set

    def evaluate(self, X: list[list[float]], y: list[str]) -> dict[str, float]:
        correct = 0
        for x, lab in zip(X, y):
            pred, _ = self.predict(x)
            if pred == lab:
                correct += 1
        acc = correct / len(X) if X else 0.0
        return {"accuracy": acc, "n": float(len(X))}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "classes": list(self.classes),
            "weights": self.weights,
            "bias": self.bias,
            "trained": self.trained,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ComplexityClassifier:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            classes=tuple(data["classes"]),
            weights=data["weights"],
            bias=data["bias"],
            trained=bool(data["trained"]),
        )
