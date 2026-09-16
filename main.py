import numpy as np
import pandas as pd
from numpy.typing import NDArray
from src.embeddings import TextEmbeddingGenerator

from src.clean import clean_text_series
from src.models import ComplianceClassifier, TopicModeler


def run_pipeline() -> None:
    """
    Executes the end-to-end NLP administrative data classification pipeline,
    enforcing C-contiguous memory layouts and vectorized data transformations.
    """
    print("[*] Initializing NLP Administrative Data Classification Pipeline...")

    # 1. Ingest / Construct administrative records dataset
    raw_data: pd.DataFrame = pd.DataFrame(
        {
            "record_id": [f"REC-{i:03d}" for i in range(1, 7)],
            "text": [
                "URGENT: Check https://example.com/audit for compliance fee leakage violations!!!",
                "Investigation case note: routine transaction anomaly flagged under Section 404.",
                "Survey feedback: operational risk exposure is increasing across branch networks.",
                "Audit findings confirm severe exception logging failures in core database engine.",
                "Minor variance noted in quarterly reconciliation reports, no immediate action.",
                "Critical compliance breach detected regarding cross-border capital flow reporting.",
            ],
            "label": [1, 1, 0, 1, 0, 1],
        }
    )

    print(f"[*] Loaded corpus containing {len(raw_data)} records.")

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

    # 5. Supervised Compliance Severity Classification Head
    print("[*] Training supervised compliance classifier on dense embeddings...")
    y: NDArray[np.int64] = raw_data["label"].to_numpy(dtype=np.int64, copy=False)
    classifier = ComplianceClassifier()
    classifier.fit(embeddings, y)

    predictions: NDArray[np.int64] = classifier.predict(embeddings)
    print(f"    -> Actual Labels          : {y}")
    print(f"    -> Predicted Severity     : {predictions}")
    print("[+] Pipeline execution completed successfully.")


if __name__ == "__main__":
    run_pipeline()
