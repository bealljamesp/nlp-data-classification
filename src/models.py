from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import config


class ComplianceClassifier(BaseEstimator, ClassifierMixin):
    """
    Supervised classification head for operational compliance severity grading,
    operating directly on dense, C-contiguous embedding arrays.
    """

    def __init__(self, random_state: int = config.random_state) -> None:
        self.model: LogisticRegression = LogisticRegression(random_state=random_state)

    def fit(self, X: NDArray[np.float64], y: NDArray[Any]) -> "ComplianceClassifier":
        """Fits the logistic regression classification head to dense embeddings."""
        self.model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Generates C-contiguous discrete class predictions."""
        preds: np.ndarray = self.model.predict(X)
        return np.ascontiguousarray(preds, dtype=np.int64)


class TopicModeler:
    """
    Unsupervised topic modeling pipeline using Latent Dirichlet Allocation (LDA)
    to extract latent structural themes from administrative text records.
    """

    def __init__(
        self, n_topics: int = 5, random_state: int = config.random_state
    ) -> None:
        self.vectorizer: CountVectorizer = CountVectorizer(stop_words="english")
        self.lda: LatentDirichletAllocation = LatentDirichletAllocation(
            n_components=n_topics, random_state=random_state
        )

    def fit_transform(self, texts: list[str]) -> NDArray[np.float64]:
        """Transforms text documents into a C-contiguous document-topic matrix."""
        counts: Any = self.vectorizer.fit_transform(texts)
        doc_topics: np.ndarray = self.lda.fit_transform(counts)
        return np.ascontiguousarray(doc_topics, dtype=np.float64)
