"""Select Ollama vs OpenAI-compatible (vLLM) chat clients from environment."""

from __future__ import annotations

import os

from .ollama_client import OllamaClient
from .openai_compat_client import DEFAULT_VLLM_CHAT_MODEL, OpenAICompatClient

ChatClient = OllamaClient | OpenAICompatClient


def llm_backend() -> str:
    return os.getenv("LLM_BACKEND", "vllm").strip().lower()


def openai_compat_base_url(explicit: str | None = None, model_env: str | None = None) -> str:
    """Resolve OpenAI-compatible API base (``…/v1``) using model-specific or default env vars."""
    if explicit:
        url = explicit.rstrip("/")
    else:
        url = ""
        if model_env:
            url_env = model_env.replace("_MODEL", "_BASE_URL")
            url = (os.getenv(url_env) or "").strip().rstrip("/")
        if not url:
            url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip().rstrip("/")
        if not url:
            origin = os.getenv("VLLM_BASE_URL", "http://localhost:5000").strip().rstrip("/")
            if "vllm-chat" in origin:
                if model_env == "LLAMA_MODEL":
                    origin = origin.replace("vllm-chat", "vllm-llama3")
                elif model_env == "QWEN_MODEL":
                    origin = origin.replace("vllm-chat", "vllm-qwen")
            url = origin if origin.endswith("/v1") else f"{origin}/v1"
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def resolve_vllm_model_name(model_env: str, default: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return (
        os.getenv(model_env)
        or os.getenv("VLLM_CHAT_MODEL")
        or default
    )


def create_chat_client(
    *,
    model_env: str,
    default_ollama: str,
    default_vllm: str = DEFAULT_VLLM_CHAT_MODEL,
    model_name: str | None = None,
    base_url: str | None = None,
) -> ChatClient:
    backend = llm_backend()
    if backend in ("vllm", "openai", "openai_compat"):
        url = openai_compat_base_url(base_url, model_env)
        name = resolve_vllm_model_name(model_env, default_vllm, model_name)
        return OpenAICompatClient(model_name=name, base_url=url)
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    name = model_name or os.getenv(model_env, default_ollama)
    return OllamaClient(model_name=name, base_url=url)
