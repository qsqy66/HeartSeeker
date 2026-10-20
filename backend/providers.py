"""Remote OpenAI-compatible model providers.

Both DeepSeek and Qwen expose the OpenAI chat-completions shape, so the rest of
the application only depends on :func:`chat`, :func:`stream` and
:func:`chat_json`.  Keeping the HTTP details here makes it possible to add a
local provider later without touching retrieval or the agent loop.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional

PROVIDERS = {
    "deepseek": "https://api.deepseek.com/chat/completions",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}
MODEL_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com/models",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
}


class ProviderError(RuntimeError):
    """A remote model call failed.  ``status`` mirrors the HTTP status when known."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def _provider_url(provider: str) -> str:
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise ProviderError("仅支持 deepseek 或 qwen")
    return PROVIDERS[provider]


def _error_message(raw: bytes, fallback: str) -> str:
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return fallback
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if message:
            return str(message)[:300]
    if isinstance(error, str):
        return error[:300]
    return fallback


def _open(request: urllib.request.Request, timeout: float):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raw = error.read()
        if error.code in (401, 403):
            message = "API Key 无效或没有访问该模型的权限"
        else:
            message = _error_message(raw, f"模型服务返回 HTTP {error.code}")
        raise ProviderError(message, error.code) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ProviderError("模型服务暂不可用，请检查网络后重试") from error


def _payload(model: str, messages: list[dict[str, str]], temperature: float,
             stream: bool = False) -> bytes:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if stream:
        body["stream"] = True
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def chat(provider: str, key: str, model: str, messages: list[dict[str, str]],
         temperature: float = 0.1, timeout: float = 60.0) -> str:
    """Return a complete assistant message."""
    request = urllib.request.Request(
        _provider_url(provider),
        data=_payload(model, messages, temperature),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with _open(request, timeout) as response:
        raw = response.read()
    try:
        data = json.loads(raw.decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ProviderError("模型返回了无法解析的响应") from error
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("模型返回了空响应")
    return content


def stream(provider: str, key: str, model: str, messages: list[dict[str, str]],
           temperature: float = 0.1, timeout: float = 120.0) -> Iterator[str]:
    """Yield text deltas from an OpenAI-compatible streaming response."""
    request = urllib.request.Request(
        _provider_url(provider),
        data=_payload(model, messages, temperature, stream=True),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with _open(request, timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                yield piece


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of one JSON object from model output.

    Models sometimes wrap JSON in ```json fences or add a sentence around it.
    ``raw_decode`` from the first ``{`` handles both cases without a brittle
    regex for nested objects.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except ValueError:
        end = text.rfind("}")
        if end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def chat_json(provider: str, key: str, model: str, messages: list[dict[str, str]],
              temperature: float = 0.0, timeout: float = 90.0) -> dict[str, Any]:
    """Call the model and require one JSON object back."""
    raw = chat(provider, key, model, messages, temperature=temperature, timeout=timeout)
    parsed = extract_json_object(raw)
    if parsed is None:
        raise ProviderError("模型没有返回合法的 JSON")
    return parsed


def list_models(provider: str, key: str, timeout: float = 20.0) -> list[str]:
    provider = provider.lower().strip()
    if provider not in MODEL_ENDPOINTS:
        raise ProviderError("仅支持 deepseek 或 qwen")
    request = urllib.request.Request(
        MODEL_ENDPOINTS[provider],
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    with _open(request, timeout) as response:
        raw = response.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ProviderError("模型列表响应无法解析") from error
    models = sorted({
        item["id"] for item in data.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    })
    if not models:
        raise ProviderError("供应商未返回可用模型列表")
    return models
