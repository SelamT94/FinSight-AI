"""HTTP client for OpenAI-compatible chat completions (e.g. vLLM ``/v1/chat/completions``)."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

# Default aligns with docker-compose.yml vllm-chat --model …
DEFAULT_VLLM_CHAT_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


class OpenAICompatClient:
    """Thin wrapper around OpenAI-compatible ``/v1/chat/completions`` and ``/v1/models``."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:5000/v1",
    ) -> None:
        self.model_name = model_name
        self.base_url = _normalize_base_url(base_url)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

        self._check_models_on_init()

    def _api_origin(self) -> str:
        b = self.base_url
        if b.endswith("/v1"):
            return b[:-3].rstrip("/")
        return b

    def _fetch_models_payload(self) -> dict[str, Any] | None:
        url = f"{self.base_url}/models"
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            self.logger.debug("_fetch_models_payload: %s", e)
            return None

    def _fetch_health_ok(self) -> bool:
        origin = self._api_origin()
        if not origin:
            return False
        url = f"{origin}/health"
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=10) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                return 200 <= int(code) < 300
        except Exception:
            return False

    def _model_listed(self, data: dict[str, Any]) -> bool:
        items = data.get("data") or []
        ids = [str(x.get("id", "")) for x in items if isinstance(x, dict)]
        target = self.model_name
        if not ids:
            return False
        tail = target.split("/")[-1]
        for mid in ids:
            if mid == target or mid.endswith(tail) or tail in mid or mid in target:
                return True
        return False

    def _check_models_on_init(self) -> None:
        try:
            data = self._fetch_models_payload()
            if data is not None:
                if not self._model_listed(data):
                    self.logger.warning(
                        "Model %r not listed in GET /v1/models (check OPENAI_COMPAT_BASE_URL "
                        "and LLAMA_MODEL / MISTRAL_MODEL).",
                        self.model_name,
                    )
            elif not self._fetch_health_ok():
                self.logger.warning(
                    "Could not reach OpenAI-compatible API at %s "
                    "(tried GET /v1/models and GET /health on %s).",
                    self.base_url,
                    self._api_origin(),
                )
        except Exception as e:
            self.logger.warning("Availability check failed: %s", e)

    def is_available(self) -> bool:
        """True if ``/v1/models`` lists this model, or (empty listing + healthy) during warmup."""
        try:
            data = self._fetch_models_payload()
            if data is not None:
                if self._model_listed(data):
                    return True
                items = data.get("data") or []
                if not items and self._fetch_health_ok():
                    return True
                return False
            return self._fetch_health_ok()
        except Exception as e:
            self.logger.debug("is_available: %s", e)
            return False

    @staticmethod
    def _messages_for_vllm(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        """Build user-only messages (Mistral on vLLM rejects role=system)."""
        user_content = prompt
        if system_prompt:
            user_content = f"{system_prompt.strip()}\n\n{prompt}"
        return [{"role": "user", "content": user_content}]

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """POST non-streaming chat completion; returns the same metadata shape as the old Ollama client."""
        url = f"{self.base_url}/chat/completions"
        messages = self._messages_for_vllm(prompt, system_prompt)

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body.update(kwargs)
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            url,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_wall0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            self.logger.exception("OpenAI-compat HTTPError %s: %s", e.code, err_body)
            raise RuntimeError(f"chat/completions failed: HTTP {e.code} {err_body}") from e
        except urllib.error.URLError as e:
            self.logger.exception("OpenAI-compat URLError: %s", e)
            raise RuntimeError(
                f"OpenAI-compatible API unreachable at {self.base_url}: {e}"
            ) from e
        except Exception as e:
            self.logger.exception("generate failed: %s", e)
            raise

        t_wall1 = time.perf_counter()
        inference_time_sec = t_wall1 - t_wall0

        choices = payload.get("choices") or []
        choice0 = choices[0] if choices else {}
        msg = choice0.get("message") or {}
        text = str(msg.get("content") or "")

        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        response_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + response_tokens))

        # Some servers omit counts when empty
        if total_tokens <= 0 and text:
            response_tokens = max(len(text.split()), 1)
            total_tokens = prompt_tokens + response_tokens

        tokens_per_sec = (
            response_tokens / inference_time_sec if inference_time_sec > 0 else 0.0
        )

        return {
            "response": text,
            "model": str(payload.get("model", self.model_name)),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "inference_time_sec": float(inference_time_sec),
            "tokens_per_sec": float(tokens_per_sec),
            "raw": payload,
        }
