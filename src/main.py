from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.clean import clean_text_series
from src.embeddings import TextEmbeddingGenerator
from src.evaluation import ModelEvaluator
from src.ingest import ChunkedIngestionEngine
from src.models import ComplianceClassifier, TopicModeler


def run_pipeline() -> None:
    """
    Executes the end-to-end NLP administrative data classification pipeline,
    streaming records via chunked ingestion and enforcing C-contiguous memory layouts.
    """
    print("[*] Initializing NLP Administrative Data Classification Pipeline...")

    file_path = Path("data/compliance_samples.csv")
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please create the sample CSV first."
        )

    # 1. Chunked Ingestion
    print("[*] Streaming records via ChunkedIngestionEngine...")
    ingestion_engine = ChunkedIngestionEngine(file_path=file_path, chunk_size=100)

    chunks = list(ingestion_engine.stream_csv_chunks())
    raw_data: pd.DataFrame = pd.concat(chunks, ignore_index=True)
    print(
        f"    -> Loaded corpus containing {len(raw_data)} records across {len(chunks)} chunk(s)."
    )

    # 2. Vectorized Text Preprocessing (Anti-loop mandate enforced)
    print("[*] Executing vectorized text cleaning...")
    cleaned_text: pd.Series = clean_text_series(raw_data["text"])

    # 3. Dense Transformer Embedding Generation (C-array NDArray[np.float64])
    print("[*] Generating dense vector representations via Transformer encoder...")
    embedder = TextEmbeddingGenerator()
    embeddings: NDArray[np.float64] = embedder.encode(cleaned_text.tolist())
    print(f"    -> Embedding Matrix Shape : {embeddings.shape}")
    print(
        f"    -> Data Type & Memory Map : {embeddings.dtype} | C-Contiguous: {embeddings.flags['C_CONTIGUOUS']}"
    )

    # 4. Unsupervised Topic Modeling (LDA)
    print("[*] Extracting latent themes via unsupervised LDA topic modeling...")
    topic_modeler = TopicModeler(n_topics=2)
    doc_topics: NDArray[np.float64] = topic_modeler.fit_transform(cleaned_text.tolist())
    print(f"    -> Document-Topic Matrix  : {doc_topics.shape}")

    # 5. Supervised Compliance Classification & Cross-Validation Evaluation
    print("[*] Evaluating classifier performance via stratified cross-validation...")
    y: NDArray[np.int64] = raw_data["label"].to_numpy(dtype=np.int64, copy=False)
    classifier = ComplianceClassifier()

    evaluator = ModelEvaluator(n_splits=3)  # Small split for 6-record sample
    metrics = evaluator.evaluate_cv(classifier, embeddings, y)
    print(
        f"    -> Mean CV Accuracy       : {metrics['mean_accuracy']:.2f} (±{metrics['std_accuracy']:.2f})"
    )
    print("[+] Pipeline execution completed successfully.")


if __name__ == "__main__":
    run_pipeline()
