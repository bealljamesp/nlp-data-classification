import numpy as np
from numpy.typing import NDArray

from src.evaluation import ModelEvaluator
from src.models import ComplianceClassifier


def test_model_evaluator() -> None:
    """Validates stratified cross-validation scoring for the compliance classifier."""
    X: NDArray[np.float64] = np.random.rand(50, 16).astype(np.float64)
    y: NDArray[np.int64] = np.random.choice([0, 1], size=50).astype(np.int64)

    evaluator: ModelEvaluator = ModelEvaluator(n_splits=5)
    classifier: ComplianceClassifier = ComplianceClassifier()

    metrics: dict[str, float] = evaluator.evaluate_cv(classifier, X, y)

    assert "mean_accuracy" in metrics
    assert "std_accuracy" in metrics
    assert 0.0 <= metrics["mean_accuracy"] <= 1.0
