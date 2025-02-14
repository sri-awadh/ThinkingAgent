from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    num_retries: int = 3
    retry_min_wait: int = 4
    retry_max_wait: int = 10
    retry_multiplier: float = 2