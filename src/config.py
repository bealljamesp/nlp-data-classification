from pathlib import Path

from pydantic import BaseModel


class PipelineConfig(BaseModel):
    data_dir: Path = Path("data")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    max_length: int = 512
    random_state: int = 42


config = PipelineConfig()
