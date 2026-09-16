import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold


class ModelEvaluator:
    """
    Performs stratified cross-validation and performance metric tracking
    for compliance classification models.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.cv: StratifiedKFold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def evaluate_cv(
        self, model: object, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> dict[str, float]:
        """Executes stratified k-fold cross-validation and returns mean and standard deviation of accuracy."""
        scores: list[float] = []

        for train_idx, val_idx in self.cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds: NDArray[np.int64] = model.predict(X_val)

            # Vectorized accuracy calculation
            accuracy: float = float(np.mean(preds == y_val))
            scores.append(accuracy)

        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
        }
