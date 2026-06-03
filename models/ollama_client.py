"""HTTP client for Ollama generate/chat with timing and token-ish counts."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any


class OllamaClient:
    """Thin wrapper around Ollama ``/api/tags`` and ``/api/generate``."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

        self._check_tags_on_init()

    def _get_tags(self) -> dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _check_tags_on_init(self) -> None:
        try:
            data = self._get_tags()
            if not self._model_listed(data):
                self.logger.warning(
                    "Model %r not found in /api/tags (pull with: ollama pull %s)",
                    self.model_name,
                    self.model_name,
                )
        except Exception as e:
            self.logger.warning("Could not reach Ollama at %s: %s", self.base_url, e)

    def _model_listed(self, tags_payload: dict[str, Any]) -> bool:
        names = [m.get("name", "") for m in tags_payload.get("models", [])]
        target = self.model_name
        for n in names:
            if n == target or n.startswith(target + ":") or target.startswith(n.split(":")[0]):
                return True
            if target in n:
                return True
        return False

    def is_available(self) -> bool:
        """Return True if Ollama responds and the model appears in ``/api/tags``."""
        try:
            data = self._get_tags()
            return self._model_listed(data)
        except Exception as e:
            self.logger.debug("is_available: %s", e)
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call ``/api/generate`` (non-streaming). Returns unified metadata dict."""
        url = f"{self.base_url}/api/generate"
        body: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if kwargs:
            opts = body["options"]
            if "repetition_penalty" in kwargs:
                opts["repeat_penalty"] = kwargs["repetition_penalty"]
            if "frequency_penalty" in kwargs:
                opts["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                opts["presence_penalty"] = kwargs["presence_penalty"]
            for k, v in kwargs.items():
                if k not in ("repetition_penalty", "frequency_penalty", "presence_penalty"):
                    opts[k] = v

        if system_prompt:
            body["system"] = system_prompt

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
            self.logger.exception("Ollama HTTPError %s: %s", e.code, err_body)
            raise RuntimeError(f"Ollama generate failed: HTTP {e.code} {err_body}") from e
        except urllib.error.URLError as e:
            self.logger.exception("Ollama URLError: %s", e)
            raise RuntimeError(f"Ollama unreachable at {self.base_url}: {e}") from e
        except Exception as e:
            self.logger.exception("generate failed: %s", e)
            raise

        t_wall1 = time.perf_counter()
        inference_time_sec = t_wall1 - t_wall0

        text = str(payload.get("response", ""))
        prompt_tokens = int(
            payload.get("prompt_eval_count")
            or payload.get("prompt_tokens")
            or 0
        )
        response_tokens = int(
            payload.get("eval_count")
            or payload.get("response_tokens")
            or 0
        )
        total_tokens = prompt_tokens + response_tokens
        if total_tokens == 0 and text:
            # Rough fallback when API omits counts
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
