"""API key storage backed by the operating system credential manager."""
from __future__ import annotations

from typing import Optional

from .providers import PROVIDERS

# Non-chat services that still need their own credential slot.  Keeping them
# out of PROVIDERS means they never appear in the chat-provider picker.
EXTRA_KEY_SOURCES = {"qwen-rerank", "qwen-nli", "aliyun-ocr"}
SERVICE_NAME = "heartseeker-med"


class CredentialError(RuntimeError):
    """The system credential store is unavailable."""


def key_store(provider: str, value: Optional[str] = None) -> Optional[str]:
    """Read or write a provider key in keyring.

    Keys are namespaced by the provider/key source.  Chat providers use their
    preset id; semantic embeddings use ``qwen`` (DashScope) even when the chat
    provider is someone else, so switching chat vendors never sends a foreign
    key to the embedding endpoint.

    ``keyring`` is imported lazily so importing the backend never fails on a
    headless machine; only an explicit key operation reports the problem.
    """
    provider = provider.lower().strip()
    if provider not in PROVIDERS and provider not in EXTRA_KEY_SOURCES:
        raise CredentialError("未知的模型供应商")
    try:
        import keyring  # type: ignore
        if value is None:
            return keyring.get_password(SERVICE_NAME, provider)
        keyring.set_password(SERVICE_NAME, provider, value)
        return value
    except Exception as error:  # keyring has platform-specific exception types
        raise CredentialError("系统安全凭据库不可用，请安装并启用 keyring 后再配置 API Key") from error


def safe_key(provider: str) -> Optional[str]:
    try:
        return key_store(provider)
    except CredentialError:
        return None
