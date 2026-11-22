"""
Pure-Python regressors (no numpy/sklearn/xgboost required):

  • RandomForest  — bagged shallow regression trees
  • XGBoost-style — residual gradient boosting with trees
  • NeuralNet     — tiny MLP with SGD
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: Sequence[float], mu: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs) if mu is None else mu
    return sum((x - m) ** 2 for x in xs) / len(xs)


@dataclass
class TreeNode:
    prediction: float = 0.0
    feature: int | None = None
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _best_split(
    X: list[list[float]],
    y: list[float],
    indices: list[int],
    *,
    rng: random.Random,
    max_features: int,
) -> tuple[int, float, list[int], list[int]] | None:
    if len(indices) < 4:
        return None
    n_features = len(X[0])
    feat_order = list(range(n_features))
    rng.shuffle(feat_order)
    feat_order = feat_order[: max(1, min(max_features, n_features))]

    parent_var = _variance([y[i] for i in indices])
    best: tuple[float, int, float, list[int], list[int]] | None = None

    for j in feat_order:
        vals = sorted({X[i][j] for i in indices})
        if len(vals) < 2:
            continue
        # Midpoints between consecutive unique values (works for one-hots too)
        candidates = [(vals[k] + vals[k + 1]) / 2.0 for k in range(len(vals) - 1)]
        if len(candidates) > 12:
            step = max(1, len(candidates) // 12)
            candidates = candidates[::step]
        for t in candidates:
            left = [i for i in indices if X[i][j] <= t]
            right = [i for i in indices if X[i][j] > t]
            if len(left) < 2 or len(right) < 2:
                continue
            var = (
                len(left) * _variance([y[i] for i in left])
                + len(right) * _variance([y[i] for i in right])
            ) / len(indices)
            gain = parent_var - var
            if best is None or gain > best[0]:
                best = (gain, j, t, left, right)
    if best is None or best[0] <= 1e-12:
        return None
    return best[1], best[2], best[3], best[4]


def _build_tree(
    X: list[list[float]],
    y: list[float],
    indices: list[int],
    *,
    depth: int,
    max_depth: int,
    rng: random.Random,
    max_features: int,
) -> TreeNode:
    pred = _mean([y[i] for i in indices])
    if depth >= max_depth or len(indices) < 4:
        return TreeNode(prediction=pred)
    split = _best_split(X, y, indices, rng=rng, max_features=max_features)
    if split is None:
        return TreeNode(prediction=pred)
    j, thr, left, right = split
    return TreeNode(
        prediction=pred,
        feature=j,
        threshold=thr,
        left=_build_tree(
            X, y, left, depth=depth + 1, max_depth=max_depth, rng=rng, max_features=max_features
        ),
        right=_build_tree(
            X, y, right, depth=depth + 1, max_depth=max_depth, rng=rng, max_features=max_features
        ),
    )


def _predict_tree(node: TreeNode, x: list[float]) -> float:
    while not node.is_leaf:
        assert node.feature is not None and node.left and node.right
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction


@dataclass
class RandomForestRegressorPure:
    n_estimators: int = 40
    max_depth: int = 8
    max_features: str | int = "sqrt"
    seed: int = 0
    trees: list[TreeNode] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[float]) -> RandomForestRegressorPure:
        rng = random.Random(self.seed)
        n, d = len(X), len(X[0])
        if self.max_features == "sqrt":
            mf = max(1, int(math.sqrt(d)))
        elif self.max_features == "log2":
            mf = max(1, int(math.log2(d)))
        elif self.max_features == "all":
            mf = d
        else:
            mf = int(self.max_features)
        self.trees = []
        for _ in range(self.n_estimators):
            bag = [rng.randrange(n) for _ in range(n)]
            self.trees.append(
                _build_tree(X, y, bag, depth=0, max_depth=self.max_depth, rng=rng, max_features=mf)
            )
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        return [_mean([_predict_tree(t, x) for t in self.trees]) for x in X]


@dataclass
class GradientBoostingRegressorPure:
    """XGBoost-style residual boosting with shallow regression trees."""

    n_estimators: int = 60
    max_depth: int = 3
    learning_rate: float = 0.1
    seed: int = 0
    base: float = 0.0
    trees: list[TreeNode] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[float]) -> GradientBoostingRegressorPure:
        rng = random.Random(self.seed)
        n, d = len(X), len(X[0])
        mf = d  # small feature set — use all columns each split
        self.base = _mean(y)
        preds = [self.base] * n
        self.trees = []
        indices = list(range(n))
        for _ in range(self.n_estimators):
            resid = [yi - pi for yi, pi in zip(y, preds)]
            tree = _build_tree(
                X, resid, indices, depth=0, max_depth=self.max_depth, rng=rng, max_features=mf
            )
            self.trees.append(tree)
            for i, x in enumerate(X):
                preds[i] += self.learning_rate * _predict_tree(tree, x)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        out = []
        for x in X:
            p = self.base
            for t in self.trees:
                p += self.learning_rate * _predict_tree(t, x)
            out.append(p)
        return out


def _relu(z: float) -> float:
    return z if z > 0 else 0.0


def _relu_grad(z: float) -> float:
    return 1.0 if z > 0 else 0.0


@dataclass
class NeuralNetRegressorPure:
    """2-layer MLP (64→32) trained with SGD + MSE."""

    hidden1: int = 32
    hidden2: int = 16
    lr: float = 0.01
    epochs: int = 250
    seed: int = 0
    # params filled on fit
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    W1: list[list[float]] = field(default_factory=list)
    b1: list[float] = field(default_factory=list)
    W2: list[list[float]] = field(default_factory=list)
    b2: list[float] = field(default_factory=list)
    W3: list[float] = field(default_factory=list)
    b3: float = 0.0

    def _init(self, d: int, rng: random.Random) -> None:
        def layer(a: int, b: int) -> list[list[float]]:
            scale = math.sqrt(2.0 / max(1, a))
            return [[rng.gauss(0, scale) for _ in range(b)] for _ in range(a)]

        self.W1 = layer(d, self.hidden1)
        self.b1 = [0.0] * self.hidden1
        self.W2 = layer(self.hidden1, self.hidden2)
        self.b2 = [0.0] * self.hidden2
        self.W3 = [rng.gauss(0, 0.1) for _ in range(self.hidden2)]
        self.b3 = 0.0

    def _normalize(self, X: list[list[float]], fit: bool) -> list[list[float]]:
        d = len(X[0])
        if fit:
            self.mean = [_mean([row[j] for row in X]) for j in range(d)]
            self.std = [
                math.sqrt(_variance([row[j] for row in X], self.mean[j])) or 1.0 for j in range(d)
            ]
        return [
            [(row[j] - self.mean[j]) / self.std[j] for j in range(d)] for row in X
        ]

    def _forward(self, x: list[float]) -> tuple[list[float], list[float], float]:
        h1 = [_relu(sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]) for j in range(self.hidden1)]
        h2 = [
            _relu(sum(h1[i] * self.W2[i][j] for i in range(self.hidden1)) + self.b2[j])
            for j in range(self.hidden2)
        ]
        out = sum(h2[j] * self.W3[j] for j in range(self.hidden2)) + self.b3
        return h1, h2, out

    def fit(self, X: list[list[float]], y: list[float]) -> NeuralNetRegressorPure:
        rng = random.Random(self.seed)
        Xn = self._normalize(X, fit=True)
        d = len(Xn[0])
        self._init(d, rng)
        n = len(Xn)
        order = list(range(n))
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            for idx in order:
                x = Xn[idx]
                target = y[idx]
                h1, h2, pred = self._forward(x)
                err = pred - target
                # backprop
                dW3 = [err * h2[j] for j in range(self.hidden2)]
                db3 = err
                dh2 = [err * self.W3[j] * _relu_grad(h2[j]) for j in range(self.hidden2)]
                dW2 = [
                    [dh2[j] * h1[i] for j in range(self.hidden2)] for i in range(self.hidden1)
                ]
                db2 = dh2[:]
                dh1 = [
                    sum(dh2[j] * self.W2[i][j] for j in range(self.hidden2)) * _relu_grad(h1[i])
                    for i in range(self.hidden1)
                ]
                dW1 = [[dh1[j] * x[i] for j in range(self.hidden1)] for i in range(d)]
                db1 = dh1[:]

                for j in range(self.hidden2):
                    self.W3[j] -= self.lr * dW3[j]
                self.b3 -= self.lr * db3
                for i in range(self.hidden1):
                    for j in range(self.hidden2):
                        self.W2[i][j] -= self.lr * dW2[i][j]
                for j in range(self.hidden2):
                    self.b2[j] -= self.lr * db2[j]
                for i in range(d):
                    for j in range(self.hidden1):
                        self.W1[i][j] -= self.lr * dW1[i][j]
                for j in range(self.hidden1):
                    self.b1[j] -= self.lr * db1[j]
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        Xn = self._normalize(X, fit=False)
        return [self._forward(x)[2] for x in Xn]
