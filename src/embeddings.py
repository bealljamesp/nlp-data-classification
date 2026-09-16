from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.config import config


class TextEmbeddingGenerator:
    """
    Encodes unstructured textual data into dense vector representations,
    mapping outputs directly into contiguous C-arrays of type NDArray[np.float64].
    """

    def __init__(self, model_name: str = config.model_name) -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """
        Transforms a sequence of clean text strings into a contiguous,
        SIMD-aligned double-precision embedding matrix.
        """
        # Batch encode via underlying optimized transformer engines
        embeddings: np.ndarray = self.model.encode(
            list(texts),
            batch_size=config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Enforce strict C-contiguous memory layout and np.float64 precision
        return np.ascontiguousarray(embeddings, dtype=np.float64)
