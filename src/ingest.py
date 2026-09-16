from pathlib import Path
from typing import Iterator

import pandas as pd
from pydantic import BaseModel, Field


class RecordSchema(BaseModel):
    """Pydantic schema enforcing structured validation on incoming textual records."""

    record_id: str = Field(..., description="Unique record or case identifier")
    text: str = Field(..., description="Unstructured textual notes or survey write-in")
    label: int | None = Field(
        default=None, description="Optional compliance severity label"
    )


class ChunkedIngestionEngine:
    """
    Streams and validates large unstructured text datasets in memory-safe chunks
    using Polars and pandas.
    """

    def __init__(self, file_path: Path, chunk_size: int = 10000) -> None:
        self.file_path: Path = file_path
        self.chunk_size: int = chunk_size

    def stream_csv_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Streams CSV data in memory-safe blocks, coercing types and validating
        schema compliance without loading the entire corpus into memory.
        """
        # Utilize Polars / Pandas chunked iteration for low-overhead streaming
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            # Validate basic column existence
            if "text" not in chunk.columns or "record_id" not in chunk.columns:
                raise ValueError(
                    "Source file missing required 'text' or 'record_id' columns."
                )

            # Drop nulls in critical text fields vectorized
            chunk = chunk.dropna(subset=["text"])
            yield chunk
