"""API key storage backed by the operating system credential manager."""
from __future__ import annotations

from typing import Optional

from .providers import PROVIDERS


class CredentialError(RuntimeError):
    """The system credential store is unavailable."""


def key_store(provider: str, value: Optional[str] = None) -> Optional[str]:
    """Read or write a provider key in keyring.

    ``keyring`` is imported lazily so importing the backend never fails on a
    headless machine; only an explicit key operation reports the problem.
    """
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise CredentialError("仅支持 deepseek 或 qwen")
    try:
        import keyring  # type: ignore
        if value is None:
            return keyring.get_password("paper-reader", provider)
        keyring.set_password("paper-reader", provider, value)
        return value
    except Exception as error:  # keyring has platform-specific exception types
        raise CredentialError("系统安全凭据库不可用，请安装并启用 keyring 后再配置 API Key") from error


def safe_key(provider: str) -> Optional[str]:
    try:
        return key_store(provider)
    except CredentialError:
        return None
