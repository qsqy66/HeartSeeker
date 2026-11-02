"""Protocol-aware remote model clients.

Vendors are configuration; protocols are behaviour.  The application supports
three wire protocols today:

* ``openai``    – OpenAI-compatible ``/chat/completions`` (OpenAI, DeepSeek,
                  Qwen/DashScope, Moonshot, Zhipu, SiliconFlow, OpenRouter,
                  Groq, Mistral, Ollama and most self-hosted gateways)
* ``anthropic`` – Anthropic Messages API (``x-api-key`` + ``anthropic-version``)
* ``gemini``    – Google Gemini ``generateContent`` API

Adding a vendor is a :class:`ProviderSpec` entry; adding a genuinely different
wire format is a new protocol adapter in this module.  Callers should pass a
resolved :class:`ProviderConfig` when they have one (so custom base URLs and
protocols survive), or a provider id string for the built-in defaults.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Union

from . import token_budget

MAX_TOKENS = 4096
ANTHROPIC_VERSION = "2023-06-01"

# Human-readable protocol names used by the settings UI.
PROTOCOLS = {
    "openai": "OpenAI 兼容（Chat Completions）",
    "anthropic": "Anthropic Messages",
    "gemini": "Google Gemini",
}

PROTOCOL_DEFAULTS = {
    "openai": {
        "chat_path": "/chat/completions",
        "models_path": "/models",
        "auth": "bearer",
    },
    "anthropic": {
        "chat_path": "/messages",
        "models_path": "/models",
        "auth": "x-api-key",
    },
    "gemini": {
        "chat_path": "/models/{model}:generateContent",
        "stream_path": "/models/{model}:streamGenerateContent?alt=sse",
        "models_path": "/models",
        "auth": "x-goog-api-key",
    },
}


# Configured by the API layer; lets this low-level module stay storage-agnostic.
_usage_recorder: Optional[Callable[[dict[str, Any]], None]] = None


def set_usage_recorder(recorder: Optional[Callable[[dict[str, Any]], None]]) -> None:
    global _usage_recorder
    _usage_recorder = recorder


def _emit_usage(config: ProviderConfig, model: str, messages: list[dict[str, str]],
                content: str, usage: Optional[dict[str, Any]] = None,
                kind: str = "chat") -> None:
    recorder = _usage_recorder
    if recorder is None:
        return
    prompt = completion = None
    if isinstance(usage, dict) and usage:
        prompt = _usage_int(usage.get("prompt_tokens"))
        completion = _usage_int(usage.get("completion_tokens"))
    estimated = prompt is None or completion is None
    if estimated:
        prompt = token_budget.estimate_messages(messages)
        completion = token_budget.estimate_tokens(content)
    payload = {
        "provider": config.id,
        "model": model,
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int((prompt or 0) + (completion or 0)),
        "estimated": estimated,
        "kind": kind,
    }
    try:
        recorder(payload)
    except Exception:  # pragma: no cover - telemetry is best-effort
        pass


def _usage_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _usage_from_response(config: ProviderConfig,
                         data: dict[str, Any]) -> dict[str, Any]:
    if config.protocol == "anthropic":
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return {"prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens")}
    if config.protocol == "gemini":
        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        return {"prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount")}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


@dataclass(frozen=True)
class ProviderSpec:
    """A vendor preset.  ``protocol`` determines request/response handling."""

    id: str
    label: str
    protocol: str
    base_url: str = ""
    default_model: str = ""
    requires_key: bool = True
    supports_model_list: bool = True
    auth: str = ""  # empty means use the protocol default

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "requires_key": self.requires_key,
            "supports_model_list": self.supports_model_list,
        }


@dataclass(frozen=True)
class ProviderConfig:
    """A resolved endpoint: vendor metadata plus protocol-specific details."""

    id: str
    label: str
    protocol: str
    base_url: str
    chat_path: str
    stream_path: str
    models_path: str
    auth: str
    default_model: str
    requires_key: bool
    supports_model_list: bool

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "requires_key": self.requires_key,
            "supports_model_list": self.supports_model_list,
        }


# Presets.  The OpenAI-compatible family deliberately includes several vendors
# so users can pick a familiar name instead of remembering each base URL.
PROVIDERS: dict[str, ProviderSpec] = {
    spec.id: spec
    for spec in [
        ProviderSpec("openai", "OpenAI", "openai",
                     "https://api.openai.com/v1", "gpt-4.1-mini"),
        ProviderSpec("deepseek", "DeepSeek", "openai",
                     "https://api.deepseek.com", "deepseek-chat"),
        ProviderSpec("qwen", "Qwen / 通义千问", "openai",
                     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        ProviderSpec("anthropic", "Anthropic Claude", "anthropic",
                     "https://api.anthropic.com/v1", "claude-sonnet-5-5"),
        ProviderSpec("gemini", "Google Gemini", "gemini",
                     "https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash"),
        ProviderSpec("moonshot", "Moonshot / Kimi", "openai",
                     "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        ProviderSpec("zhipu", "智谱 GLM", "openai",
                     "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
        ProviderSpec("siliconflow", "SiliconFlow", "openai",
                     "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
        ProviderSpec("openrouter", "OpenRouter", "openai",
                     "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ProviderSpec("groq", "Groq", "openai",
                     "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ProviderSpec("mistral", "Mistral AI", "openai",
                     "https://api.mistral.ai/v1", "mistral-small-latest"),
        ProviderSpec("ollama", "Ollama（本地）", "openai",
                     "http://127.0.0.1:11434/v1", "qwen2.5:7b", requires_key=False),
        ProviderSpec("custom", "自定义（按协议）", "openai",
                     "", "", requires_key=False),
    ]
}


class ProviderError(RuntimeError):
    """A remote model call failed.  ``status`` mirrors the HTTP status when known."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def public_catalog() -> list[dict[str, Any]]:
    """Provider presets safe to expose to the frontend (never contains keys)."""
    return [spec.public() for spec in PROVIDERS.values()]


def default_model(provider: str) -> str:
    spec = PROVIDERS.get(str(provider).lower().strip())
    return spec.default_model if spec else ""


def _normalise_base(base: str, protocol: str) -> str:
    """Accept either a base URL or a pasted chat endpoint.

    Users commonly paste ``.../v1/chat/completions`` into a field labelled
    "Base URL".  Strip only the protocol's terminal endpoint so model listing
    and chat can both be derived from one value.
    """
    suffix = {
        "openai": "/chat/completions",
        "anthropic": "/messages",
    }.get(protocol)
    if suffix and base.endswith(suffix):
        base = base[: -len(suffix)].rstrip("/")
    return base


def resolve_config(provider: Union[str, ProviderConfig, None],
                   protocol: Optional[str] = None,
                   base_url: Optional[str] = None) -> ProviderConfig:
    """Resolve a provider id (or existing config) into a concrete endpoint.

    For built-in vendors the protocol is fixed; ``protocol`` is only honoured
    for the ``custom`` vendor.  A base URL override is accepted for every
    vendor so users can point a preset at a proxy or a self-hosted gateway.
    """
    if isinstance(provider, ProviderConfig):
        if base_url is None and protocol is None:
            return provider
        provider = provider.id
    provider = (provider or "").lower().strip()
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ProviderError("未知的模型供应商，请从设置中选择")

    chosen_protocol = spec.protocol
    if spec.id == "custom":
        chosen_protocol = (protocol or spec.protocol or "openai").lower().strip()
    if chosen_protocol not in PROTOCOL_DEFAULTS:
        raise ProviderError("未知的协议类型，仅支持 openai / anthropic / gemini")

    defaults = PROTOCOL_DEFAULTS[chosen_protocol]
    base = (base_url if base_url is not None else spec.base_url).strip().rstrip("/")
    if spec.id == "custom" and not base:
        raise ProviderError("自定义供应商需要填写 Base URL")
    if base:
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProviderError("Base URL 必须是有效的 http(s) 地址")
        if parsed.username or parsed.password:
            raise ProviderError("Base URL 不能包含用户名或密码")
        base = _normalise_base(base, chosen_protocol)

    return ProviderConfig(
        id=spec.id,
        label=spec.label,
        protocol=chosen_protocol,
        base_url=base,
        chat_path=defaults["chat_path"],
        stream_path=defaults.get("stream_path", defaults["chat_path"]),
        models_path=defaults["models_path"],
        auth=spec.auth or defaults["auth"],
        default_model=spec.default_model,
        requires_key=spec.requires_key,
        supports_model_list=spec.supports_model_list,
    )


def _endpoint(config: ProviderConfig, path: str, model: Optional[str] = None) -> str:
    if path.startswith(("http://", "https://")):
        return path
    rendered = path.replace("{model}", urllib.parse.quote(model or "", safe=""))
    if not config.base_url:
        raise ProviderError("尚未配置 Base URL")
    if config.base_url.rstrip("/").endswith(rendered.lstrip("/")):
        return config.base_url
    return config.base_url.rstrip("/") + "/" + rendered.lstrip("/")


def _headers(config: ProviderConfig, key: str, accept: str = "application/json") -> dict[str, str]:
    key = (key or "").strip()
    if config.requires_key and not key:
        raise ProviderError("尚未配置该供应商的 API Key")
    headers = {"Content-Type": "application/json", "Accept": accept}
    if key:
        if config.auth == "x-api-key":
            headers["x-api-key"] = key
        elif config.auth == "x-goog-api-key":
            headers["x-goog-api-key"] = key
        elif config.auth == "api-key":
            headers["api-key"] = key
        else:
            headers["Authorization"] = "Bearer " + key
    if config.protocol == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    return headers


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
    if isinstance(data, dict) and isinstance(data.get("message"), str):
        return data["message"][:300]
    return fallback


def _open(request: urllib.request.Request, timeout: float, secret: str = ""):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raw = error.read()
        if error.code in (401, 403):
            message = "API Key 无效或没有访问该模型的权限"
        elif error.code == 404:
            message = _error_message(raw, "接口不存在，请检查 Base URL 与协议")
        else:
            message = _error_message(raw, f"模型服务返回 HTTP {error.code}")
        if secret:
            message = message.replace(secret, "***")
        raise ProviderError(message, error.code) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ProviderError("模型服务暂不可用，请检查网络或 Base URL 后重试") from error


def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        normalised.append({"role": role, "content": content})
    return normalised


def _openai_payload(model: str, messages: list[dict[str, str]], temperature: float,
                    stream: bool) -> dict[str, Any]:
    return {"model": model, "messages": messages, "temperature": temperature,
            **({"stream": True} if stream else {})}


def _anthropic_payload(model: str, messages: list[dict[str, str]], temperature: float,
                       stream: bool, max_tokens: int) -> dict[str, Any]:
    system = "\n\n".join(message["content"] for message in messages
                         if message["role"] in {"system", "developer"} and message["content"])
    chat = [
        {"role": "assistant" if message["role"] == "assistant" else "user",
         "content": message["content"]}
        for message in messages if message["role"] in {"user", "assistant"}
    ]
    if not chat:
        chat = [{"role": "user", "content": system or "请回答。"}]
    payload: dict[str, Any] = {
        "model": model,
        "messages": chat,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        payload["system"] = system
    if stream:
        payload["stream"] = True
    return payload


def _gemini_payload(model: str, messages: list[dict[str, str]], temperature: float,
                    max_tokens: int) -> dict[str, Any]:
    system = "\n\n".join(message["content"] for message in messages
                         if message["role"] in {"system", "developer"} and message["content"])
    contents = [
        {"role": "model" if message["role"] == "assistant" else "user",
         "parts": [{"text": message["content"]}]}
        for message in messages if message["role"] in {"user", "assistant"}
    ]
    if not contents:
        contents = [{"role": "user", "parts": [{"text": system or "请回答。"}]}]
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


def _request(config: ProviderConfig, key: str, model: str, messages: list[dict[str, str]],
             temperature: float, stream: bool, max_tokens: int) -> urllib.request.Request:
    if not model.strip() or model != model.strip():
        raise ProviderError("请先配置模型名称")
    path = config.stream_path if stream else config.chat_path
    if config.protocol == "openai":
        body = _openai_payload(model, messages, temperature, stream)
    elif config.protocol == "anthropic":
        body = _anthropic_payload(model, messages, temperature, stream, max_tokens)
    else:
        body = _gemini_payload(model, messages, temperature, max_tokens)
        # Model list responses use the "models/<id>" form; accept either form.
        path = path.replace(
            "{model}", urllib.parse.quote(model.removeprefix("models/"), safe="")
        )
    return urllib.request.Request(
        _endpoint(config, path, model),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=_headers(config, key),
    )


def _json_response(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ProviderError("模型返回了无法解析的响应") from error
    if not isinstance(data, dict):
        raise ProviderError("模型返回了无法解析的响应")
    return data


def _text_from_openai(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return content if isinstance(content, str) else ""


def _text_from_anthropic(data: dict[str, Any]) -> str:
    blocks = data.get("content") or []
    if isinstance(blocks, str):
        return blocks
    return "".join(
        block.get("text", "") for block in blocks
        if isinstance(block, dict) and block.get("type", "text") == "text"
        and isinstance(block.get("text"), str)
    )


def _text_from_gemini(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        if reason:
            raise ProviderError(f"Gemini 拒绝了该请求：{reason}")
        return ""
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    return "".join(
        part.get("text", "") for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _read_text(config: ProviderConfig, data: dict[str, Any]) -> str:
    if config.protocol == "anthropic":
        return _text_from_anthropic(data)
    if config.protocol == "gemini":
        return _text_from_gemini(data)
    return _text_from_openai(data)


def chat(provider: Union[str, ProviderConfig], key: str, model: str,
         messages: list[dict[str, str]], temperature: float = 0.1,
         timeout: float = 60.0, max_tokens: int = MAX_TOKENS) -> str:
    """Return a complete assistant message for any supported protocol."""
    config = resolve_config(provider)
    normalised = _normalise_messages(messages)
    request = _request(config, key, model, normalised, temperature, False, max_tokens)
    with _open(request, timeout, key) as response:
        raw = response.read()
    data = _json_response(raw)
    content = _read_text(config, data)
    _emit_usage(config, model, normalised, content, _usage_from_response(config, data))
    if not content.strip():
        raise ProviderError("模型返回了空响应")
    return content


def _iter_sse(lines: Any) -> Iterator[dict[str, Any]]:
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if isinstance(chunk, dict):
            yield chunk


def stream(provider: Union[str, ProviderConfig], key: str, model: str,
           messages: list[dict[str, str]], temperature: float = 0.1,
           timeout: float = 120.0, max_tokens: int = MAX_TOKENS) -> Iterator[str]:
    """Yield text deltas from an OpenAI, Anthropic or Gemini SSE response."""
    config = resolve_config(provider)
    normalised = _normalise_messages(messages)
    request = _request(config, key, model, normalised, temperature, True, max_tokens)
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    try:
        with _open(request, timeout, key) as response:
            for chunk in _iter_sse(response):
                if config.protocol == "openai":
                    if isinstance(chunk.get("usage"), dict):
                        usage.update(_usage_from_response(config, chunk))
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, list):
                        piece = "".join(
                            part.get("text", "") for part in piece
                            if isinstance(part, dict) and isinstance(part.get("text"), str)
                        )
                elif config.protocol == "anthropic":
                    if chunk.get("type") == "error":
                        error = chunk.get("error") or {}
                        message = error.get("message") if isinstance(error, dict) else None
                        raise ProviderError(str(message or "模型流式响应出错")[:300])
                    if chunk.get("type") == "message_start":
                        message = chunk.get("message") or {}
                        chunk_usage = message.get("usage") or {}
                        usage.update({"prompt_tokens": chunk_usage.get("input_tokens"),
                                      "completion_tokens": chunk_usage.get("output_tokens")})
                        continue
                    if chunk.get("type") == "message_delta":
                        chunk_usage = chunk.get("usage") or {}
                        usage.update({"completion_tokens": chunk_usage.get("output_tokens")})
                        continue
                    if chunk.get("type") != "content_block_delta":
                        continue
                    delta = chunk.get("delta") or {}
                    piece = delta.get("text") if delta.get("type", "text_delta") == "text_delta" else None
                else:
                    if isinstance(chunk.get("usageMetadata"), dict):
                        usage.update(_usage_from_response(config, chunk))
                    piece = _text_from_gemini(chunk)
                if isinstance(piece, str) and piece:
                    pieces.append(piece)
                    yield piece
    finally:
        if pieces or usage.get("prompt_tokens") is not None:
            _emit_usage(config, model, normalised, "".join(pieces), usage, kind="stream")


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


def chat_json(provider: Union[str, ProviderConfig], key: str, model: str,
              messages: list[dict[str, str]], temperature: float = 0.0,
              timeout: float = 90.0, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    """Call the model and require one JSON object back."""
    raw = chat(provider, key, model, messages, temperature=temperature,
               timeout=timeout, max_tokens=max_tokens)
    parsed = extract_json_object(raw)
    if parsed is None:
        raise ProviderError("模型没有返回合法的 JSON")
    return parsed


def list_models(provider: Union[str, ProviderConfig], key: str = "",
                timeout: float = 20.0) -> list[str]:
    """List models using the protocol's native discovery shape.

    Not every gateway implements discovery; callers should let users type a
    model id manually when this fails.
    """
    config = resolve_config(provider)
    if not config.supports_model_list:
        raise ProviderError("该供应商不支持读取模型列表，请手动填写模型名称")
    request = urllib.request.Request(
        _endpoint(config, config.models_path),
        headers=_headers(config, key),
    )
    with _open(request, timeout, key) as response:
        raw = response.read()
    data = _json_response(raw)
    if config.protocol == "gemini":
        models = {
            str(item["name"]).removeprefix("models/")
            for item in data.get("models", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
            and (not item.get("supportedGenerationMethods")
                 or "generateContent" in item["supportedGenerationMethods"])
        }
    else:
        models = {
            item["id"] for item in data.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    if not models:
        raise ProviderError("供应商未返回可用模型列表")
    return sorted(models)
