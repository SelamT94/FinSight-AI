"""Chat models (Ollama or vLLM), embeddings helpers, and resource monitoring."""

from .chat_backend import create_chat_client, llm_backend
from .llama_model import LlamaModel
from .mistral_model import MistralModel
from .qwen_model import QwenModel
from .ollama_client import OllamaClient
from .openai_compat_client import DEFAULT_VLLM_CHAT_MODEL, OpenAICompatClient
from .resource_monitor import ResourceMonitor

__all__ = [
    "DEFAULT_VLLM_CHAT_MODEL",
    "LlamaModel",
    "MistralModel",
    "QwenModel",
    "create_chat_client",
    "llm_backend",
    "OpenAICompatClient",
    "OllamaClient",
    "ResourceMonitor",
]
