# Building a LiteLLM Integration Project from Scratch

1. Project Structure Setup

your_project/
├── config/
│   ├── __init__.py
│   └── llm_config.py      # LLM configuration dataclass
├── core/
│   ├── __init__.py
│   ├── message.py         # Message handling
│   └── exceptions.py      # Custom exceptions
├── llm/
│   ├── __init__.py
│   ├── llm.py            # Base LLM class
│   ├── async_llm.py      # Async support
│   └── streaming_llm.py  # Streaming support
├── config.toml           # Configuration file
└── requirements.txt

2. Dependencies
Required packages in requirements.txt:

litellm>=1.0.0
pydantic>=2.0.0
tenacity>=8.0.0
python-dotenv>=1.0.0
tomli>=2.0.0  # For TOML config reading

3. Configuration Setup
config.toml

[llm]
model = "claude-3-5-sonnet-20241022"
api_key = "your-api-key"  # Can be overridden by environment variables
temperature = 0.0
max_output_tokens = 4096
num_retries = 8
retry_min_wait = 15
retry_max_wait = 120
retry_multiplier = 2

llm_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    num_retries: int = 8
    retry_min_wait: int = 15
    retry_max_wait: int = 120
    retry_multiplier: float = 2

4. Message Handling
Create a message class that's compatible with litellm but can be extended:

from typing import Literal, List
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    
    def to_dict(self):
        """Convert to litellm-compatible format"""
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name
        }

5. Base LLM Implementation
from functools import partial
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

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
        stop=stop_after_attempt(8),
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


        
api key
Can you find out how does the integration with litellm works?

Think step by step

Let me analyze how OpenHands integrates with litellm step by step.
Core Integration Class Structure
The main integration happens through the LLM class, with two specialized variants:
Base LLM class (non-streaming)
AsyncLLM class (async operations)
StreamingLLM class (streaming responses)
Configuration
The LLMConfig dataclass defines all configuration options that can be passed to litellm, including:
Model name
API keys
Base URLs
Retry settings
Temperature and other model parameters
Basic LLM Setup
When an LLM instance is created:
)
It creates a partial function with litellm's completion method, configured with all the necessary parameters from LLMConfig.
Model Information Initialization
:
The system tries to get model information from litellm to set proper defaults for:
Maximum input tokens
Maximum output tokens
Model capabilities (like vision support)
Retry Mechanism
:
The integration includes a retry mechanism using tenacity, configured to handle common API errors and rate limits.
Message Handling
OpenHands has its own Message class that's compatible with litellm's message format:
The system handles conversion between OpenHands messages and litellm's expected format.
Async Support
)
For async operations, it uses litellm's acompletion instead of completion.
Streaming Support
chunk
For streaming responses, it yields chunks from litellm's streaming response.
Provider Support
The integration supports multiple providers through litellm:
Direct providers (OpenAI, Anthropic, etc.)
LiteLLM proxy
OpenAI-compatible endpoints
Cost Tracking
0
The integration includes cost tracking using litellm's cost calculation features.
Key Integration Points:
Uses litellm as an abstraction layer for multiple LLM providers
Maintains compatibility with litellm's message format
Leverages litellm's retry mechanisms and error handling
Uses litellm's token counting and cost tracking
Supports both sync and async operations
Handles streaming responses
Supports various authentication methods and configurations
The integration is designed to be flexible and provider-agnostic, allowing OpenHands to work with any provider that litellm supports while maintaining consistent behavior across different LLM backends.
I want to create a project from scratch that uses the same approach for llms, I want the user to be able to use litellm fully like in this codebase, could you create a file with instructions of how this repository does it so that I could do it after. Explain  also how to use the config with the config.toml

Think step by step and write it plain text

Here's a step-by-step guide to implement a litellm integration similar to OpenHands:
Setting Up LLM Integration with litellm
1. Project Structure
your_project/
├── config/
│   ├── __init__.py
│   └── llm_config.py      # LLM configuration dataclass
├── core/
│   ├── __init__.py
│   ├── message.py         # Message handling
│   └── exceptions.py      # Custom exceptions
├── llm/
│   ├── __init__.py
│   ├── llm.py            # Base LLM class
│   ├── async_llm.py      # Async support
│   └── streaming_llm.py  # Streaming support
├── config.toml           # Configuration file
└── requirements.txt
2. Dependencies
Add to requirements.txt:
litellm>=1.0.0
pydantic>=2.0.0
tenacity>=8.0.0
python-dotenv>=1.0.0
tomli>=2.0.0  # For TOML config reading
3. Configuration Setup
config.toml
[llm]
model = "claude-3-5-sonnet-20241022"
api_key = "your-api-key"  # Can be overridden by environment variables
temperature = 0.0
max_output_tokens = 4096
num_retries = 8
retry_min_wait = 15
retry_max_wait = 120
retry_multiplier = 2

4. Message Handling
Create a message class that's compatible with litellm but can be extended:
from typing import Literal, List
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    
    def to_dict(self):
        """Convert to litellm-compatible format"""
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name
        }

5. Base LLM Implementation
from functools import partial
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

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
        stop=stop_after_attempt(8),
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

6. Loading Configuration
import os
import tomli
from dotenv import load_dotenv

def load_config():
    # Load environment variables
    load_dotenv()
    
    # Load config.toml
    with open("config.toml", "rb") as f:
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

7. Usage Example
from your_project.config import load_config
from your_project.llm import LLM
from your_project.core.message import Message

def main():
    # Load configuration
    config = load_config()
    
    # Initialize LLM
    llm = LLM(config)
    
    # Create messages
    messages = [
        Message(role="user", content="Hello!").to_dict()
    ]
    
    # Get completion
    response = llm.completion(messages=messages)
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()

8. Environment Variables
Create a .env file:
LLM_API_KEY=your-api-key
LLM_BASE_URL=optional-base-url