import os
import tomli
from dotenv import load_dotenv
from .llm_config import LLMConfig

def load_config(config_path: str = "config.toml") -> LLMConfig:
    # Load environment variables
    load_dotenv()
    
    # Load config.toml
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    # Create LLMConfig instance
    llm_config = LLMConfig(
        model=config["llm"]["model"],
        api_key=os.getenv("LLM_API_KEY", config["llm"]["api_key"]),
        temperature=config["llm"]["temperature"],
        max_output_tokens=config["llm"]["max_output_tokens"],
        num_retries=config["llm"]["num_retries"],
        retry_min_wait=config["llm"]["retry_min_wait"],
        retry_max_wait=config["llm"]["retry_max_wait"],
        retry_multiplier=config["llm"]["retry_multiplier"]
    )
    
    return llm_config