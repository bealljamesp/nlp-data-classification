import numpy as np
from numpy.typing import NDArray

from src.models import ComplianceClassifier, TopicModeler


def test_compliance_classifier() -> None:
    """Verifies supervised classification head output formats and memory layouts."""
    X: NDArray[np.float64] = np.random.rand(20, 16).astype(np.float64)
    y: NDArray[np.int64] = np.random.randint(0, 2, size=20).astype(np.int64)

    classifier: ComplianceClassifier = ComplianceClassifier()
    classifier.fit(X, y)
    preds: NDArray[np.int64] = classifier.predict(X)

    assert preds.shape == (20,)
    assert preds.dtype == np.int64
    assert preds.flags["C_CONTIGUOUS"]


def test_topic_modeler() -> None:
    """Validates LDA topic matrix generation and C-array constraints."""
    texts: list[str] = [
        "compliance audit report financial fee leakage",
        "investigative case notes regarding transaction anomaly",
        "survey write in concerning operational risk exposure",
        "audit findings on exception logging protocols",
    ]

    modeler: TopicModeler = TopicModeler(n_topics=2)
    doc_topics: NDArray[np.float64] = modeler.fit_transform(texts)

    assert doc_topics.shape == (4, 2)
    assert doc_topics.dtype == np.float64
    assert doc_topics.flags["C_CONTIGUOUS"]
