from functools import partial
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config.llm_config import LLMConfig

class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._setup_completion()
        
    def _setup_completion(self):
        """Setup the completion function with configuration"""
        self._completion = partial(
            litellm.completion,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def completion(self, messages, **kwargs):
        """Send completion request to LLM"""
        try:
            response = self._completion(messages=messages, **kwargs)
            return response
        except Exception as e:
            print(f"Error in completion: {e}")
            raise