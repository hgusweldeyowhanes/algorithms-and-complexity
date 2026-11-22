"""
Random Forest, XGBoost(-style), and Neural Network regressors.

Uses scikit-learn / xgboost when installed; otherwise pure-Python fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

BACKEND = "pure"


class Regressor(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


def _rows(X: Any) -> list[list[float]]:
    try:
        import numpy as np

        if isinstance(X, np.ndarray):
            return X.astype(float).tolist()
    except ImportError:
        pass
    if hasattr(X, "tolist"):
        return [list(map(float, r)) for r in X.tolist()]
    return [list(map(float, row)) for row in X]


def _vec(y: Any) -> list[float]:
    try:
        import numpy as np

        if isinstance(y, np.ndarray):
            return y.astype(float).ravel().tolist()
    except ImportError:
        pass
    return [float(v) for v in y]


@dataclass
class _PureAdapter:
    """Make pure-Python regressors accept numpy arrays transparently."""

    inner: Any

    def fit(self, X: Any, y: Any) -> _PureAdapter:
        self.inner.fit(_rows(X), _vec(y))
        return self

    def predict(self, X: Any) -> list[float]:
        return self.inner.predict(_rows(X))


@dataclass
class DualTargetModel:
    name: str
    time_model: Regressor
    memory_model: Regressor

    def fit(self, X: Any, y_time: Any, y_mem: Any) -> DualTargetModel:
        self.time_model.fit(X, y_time)
        self.memory_model.fit(X, y_mem)
        return self

    def predict(self, X: Any) -> tuple[Any, Any]:
        return self.time_model.predict(X), self.memory_model.predict(X)


def _try_sklearn_zoo(seed: int) -> list[DualTargetModel] | None:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    def rf(rs: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=2,
            random_state=rs,
            n_jobs=-1,
        )

    def mlp(rs: int) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=600,
                        random_state=rs,
                        early_stopping=True,
                        validation_fraction=0.15,
                    ),
                ),
            ]
        )

    models: list[DualTargetModel] = [
        DualTargetModel(name="RandomForest", time_model=rf(seed), memory_model=rf(seed + 1)),
    ]

    try:
        from xgboost import XGBRegressor

        def xgb(rs: int) -> Any:
            return XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            )

        models.append(
            DualTargetModel(name="XGBoost", time_model=xgb(seed), memory_model=xgb(seed + 1))
        )
    except ImportError:
        from resource_predictor.pure_ml import GradientBoostingRegressorPure

        models.append(
            DualTargetModel(
                name="XGBoost",
                time_model=_PureAdapter(GradientBoostingRegressorPure(seed=seed)),
                memory_model=_PureAdapter(GradientBoostingRegressorPure(seed=seed + 1)),
            )
        )

    models.append(
        DualTargetModel(name="NeuralNet", time_model=mlp(seed), memory_model=mlp(seed + 1))
    )
    return models


def _pure_zoo(seed: int) -> list[DualTargetModel]:
    from resource_predictor.pure_ml import (
        GradientBoostingRegressorPure,
        NeuralNetRegressorPure,
        RandomForestRegressorPure,
    )

    return [
        DualTargetModel(
            name="RandomForest",
            time_model=_PureAdapter(
                RandomForestRegressorPure(
                    n_estimators=50, max_depth=10, max_features="all", seed=seed
                )
            ),
            memory_model=_PureAdapter(
                RandomForestRegressorPure(
                    n_estimators=50, max_depth=10, max_features="all", seed=seed + 1
                )
            ),
        ),
        DualTargetModel(
            name="XGBoost",
            time_model=_PureAdapter(
                GradientBoostingRegressorPure(
                    n_estimators=100, max_depth=4, learning_rate=0.08, seed=seed
                )
            ),
            memory_model=_PureAdapter(
                GradientBoostingRegressorPure(
                    n_estimators=100, max_depth=4, learning_rate=0.08, seed=seed + 1
                )
            ),
        ),
        DualTargetModel(
            name="NeuralNet",
            time_model=_PureAdapter(
                NeuralNetRegressorPure(hidden1=48, hidden2=24, epochs=300, lr=0.015, seed=seed)
            ),
            memory_model=_PureAdapter(
                NeuralNetRegressorPure(
                    hidden1=48, hidden2=24, epochs=300, lr=0.015, seed=seed + 1
                )
            ),
        ),
    ]


def build_model_zoo(*, seed: int = 0) -> list[DualTargetModel]:
    global BACKEND
    zoo = _try_sklearn_zoo(seed)
    if zoo is not None:
        BACKEND = "sklearn"
        return zoo
    BACKEND = "pure"
    return _pure_zoo(seed)
