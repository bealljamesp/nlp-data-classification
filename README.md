[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pytest](https://img.shields.io/badge/pytest-100%25-green.svg)](https://docs.pytest.org/)

# NLP Administrative Data Classification Pipeline

An automated, high-throughput text-mining pipeline engineered to ingest unstructured textual records (e.g., investigative case notes, survey write-ins, and compliance audit reports), clean them via vectorized SIMD operations, and perform dual-engine analytics combining Transformer embeddings, unsupervised topic modeling (LDA), and supervised risk severity classification.

## Architectural Highlights

- **Forced Vectorization & Anti-Loop Mandate**: Complete ban on explicit Python loops and pandas `.apply()` for string transformations and numerical calculations. Employs compiled regular expressions and contiguous C-arrays (`np.typing.NDArray[np.float64]`).
- **Memory-Safe Streaming**: Chunked ingestion engine utilizing Polars/pandas blocks to process massive enterprise corpora without memory bloat.
- **Dual-Engine Analytics**: 
  - *Unsupervised*: Latent Dirichlet Allocation (LDA) for structural theme extraction.
  - *Supervised*: Logistic regression classification heads validated via stratified cross-validation.
- **Production CI/CD**: Automated linting (Ruff) and test execution (Pytest) via GitHub Actions.

## Repository Structure

```text
nlp-data-classification/
├── src/
│   ├── __init__.py
│   ├── config.py         # Pydantic validation & environment settings
│   ├── ingest.py         # Memory-safe chunked data ingestion
│   ├── clean.py          # Vectorized text cleaning & regex transformations
│   ├── embeddings.py     # Transformer embedding generator (NDArray[np.float64])
│   ├── models.py         # Supervised classifier & unsupervised LDA modeler
│   ├── evaluation.py     # Stratified cross-validation engine
│   └── main.py           # End-to-end execution runner
├── tests/
│   ├── __init__.py
│   ├── test_clean.py
│   ├── test_models.py
│   └── test_evaluation.py
├── data/
│   └── compliance_samples.csv
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

---

## Getting Started

```
conda create -n nlp-class python=3.12 -y
conda activate nlp-class
pip install --upgrade pip
pip install -e .
python src/main.py
```

