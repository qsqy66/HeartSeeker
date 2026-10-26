from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import providers  # noqa: E402


class FakeResponse:
    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None) -> None:
        self.body = body
        self.lines = lines if lines is not None else body.splitlines(keepends=True)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.body

    def __iter__(self):
        return iter(self.lines)


def _headers(request) -> dict[str, str]:
    return {key.lower(): value for key, value in request.header_items()}


class ProviderProtocolTestCase(unittest.TestCase):
    def _capture(self, response: FakeResponse):
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        return captured, mock.patch.object(providers.urllib.request, "urlopen", fake_urlopen)

    def test_custom_provider_requires_base_url_and_accepts_protocol(self) -> None:
        with self.assertRaises(providers.ProviderError):
            providers.resolve_config("custom", "openai", None)
        config = providers.resolve_config("custom", "anthropic", "https://gateway.example/v1")
        self.assertEqual(config.protocol, "anthropic")
        self.assertEqual(config.chat_path, "/messages")
        self.assertEqual(config.auth, "x-api-key")
        self.assertEqual(providers.resolve_config("openai").protocol, "openai")
        self.assertEqual(providers.resolve_config("gemini").protocol, "gemini")
        pasted = providers.resolve_config(
            "custom", "openai", "https://gateway.example/v1/chat/completions"
        )
        self.assertEqual(pasted.base_url, "https://gateway.example/v1")

    def test_openai_chat_request_and_response(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": "答案"}}]}).encode()
        captured, patcher = self._capture(FakeResponse(body))
        with patcher:
            answer = providers.chat("openai", "sk-openai", "gpt-4.1-mini",
                                    [{"role": "user", "content": "你好"}])
        request = captured["request"]
        self.assertEqual(answer, "答案")
        self.assertTrue(request.full_url.endswith("/chat/completions"))
        self.assertEqual(_headers(request)["authorization"], "Bearer sk-openai")
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "你好"}])
        self.assertNotIn("stream", payload)

    def test_anthropic_messages_request_and_response(self) -> None:
        body = json.dumps({
            "content": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]
        }).encode()
        captured, patcher = self._capture(FakeResponse(body))
        with patcher:
            answer = providers.chat(
                providers.resolve_config("anthropic"), "sk-ant", "claude-sonnet-5-5",
                [{"role": "system", "content": "系统提示"},
                 {"role": "user", "content": "问题"}],
            )
        request = captured["request"]
        self.assertEqual(answer, "第一段第二段")
        self.assertTrue(request.full_url.endswith("/v1/messages"))
        headers = _headers(request)
        self.assertEqual(headers["x-api-key"], "sk-ant")
        self.assertIn("anthropic-version", headers)
        self.assertNotIn("authorization", headers)
        payload = json.loads(request.data)
        self.assertEqual(payload["system"], "系统提示")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "问题"}])
        self.assertEqual(payload["max_tokens"], providers.MAX_TOKENS)

    def test_gemini_request_and_response(self) -> None:
        body = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "Gemini 回答"}]}}]
        }).encode()
        captured, patcher = self._capture(FakeResponse(body))
        with patcher:
            answer = providers.chat(
                "gemini", "goog-key", "gemini-2.5-flash",
                [{"role": "system", "content": "系统"},
                 {"role": "user", "content": "问题"}],
            )
        request = captured["request"]
        self.assertEqual(answer, "Gemini 回答")
        self.assertIn("/models/gemini-2.5-flash:generateContent", request.full_url)
        self.assertEqual(_headers(request)["x-goog-api-key"], "goog-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "系统"}]})
        self.assertEqual(payload["contents"], [{"role": "user", "parts": [{"text": "问题"}]}])

    def test_streaming_parses_each_protocol(self) -> None:
        openai_lines = [
            b'data: {"choices":[{"delta":{"content":"A"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"B"}}]}\n',
            b"data: [DONE]\n",
        ]
        captured, patcher = self._capture(FakeResponse(lines=openai_lines))
        with patcher:
            self.assertEqual(list(providers.stream("openai", "k", "m", [{"role": "user", "content": "q"}])),
                             ["A", "B"])

        anthropic_lines = [
            'event: content_block_delta\n'.encode(),
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"你"}}\n'.encode(),
            'event: content_block_delta\n'.encode(),
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"好"}}\n'.encode(),
        ]
        captured, patcher = self._capture(FakeResponse(lines=anthropic_lines))
        with patcher:
            self.assertEqual(list(providers.stream("anthropic", "k", "m", [{"role": "user", "content": "q"}])),
                             ["你", "好"])

        gemini_lines = [
            b'data: {"candidates":[{"content":{"parts":[{"text":"G"}]}}]}\n',
            b'data: {"candidates":[{"content":{"parts":[{"text":"2"}]}}]}\n',
        ]
        captured, patcher = self._capture(FakeResponse(lines=gemini_lines))
        with patcher:
            self.assertEqual(list(providers.stream("gemini", "k", "gemini-2.5-flash",
                                                   [{"role": "user", "content": "q"}])),
                             ["G", "2"])

    def test_model_listing_shapes(self) -> None:
        captured, patcher = self._capture(FakeResponse(
            json.dumps({"data": [{"id": "b-model"}, {"id": "a-model"}]}).encode()
        ))
        with patcher:
            self.assertEqual(providers.list_models("openai", "k"), ["a-model", "b-model"])

        captured, patcher = self._capture(FakeResponse(
            json.dumps({"data": [{"id": "claude-sonnet-5-5"}]}).encode()
        ))
        with patcher:
            self.assertEqual(providers.list_models("anthropic", "k"), ["claude-sonnet-5-5"])
        self.assertEqual(_headers(captured["request"])["x-api-key"], "k")

        captured, patcher = self._capture(FakeResponse(json.dumps({
            "models": [
                {"name": "models/gemini-2.5-flash",
                 "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004",
                 "supportedGenerationMethods": ["embedContent"]},
            ]
        }).encode()))
        with patcher:
            self.assertEqual(providers.list_models("gemini", "k"), ["gemini-2.5-flash"])

    def test_custom_base_url_is_used_for_chat_and_models(self) -> None:
        config = providers.resolve_config("custom", "openai", "https://llm.example/v1")
        captured, patcher = self._capture(FakeResponse(
            json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        ))
        with patcher:
            providers.chat(config, "local", "my-model", [{"role": "user", "content": "q"}])
        self.assertEqual(captured["request"].full_url, "https://llm.example/v1/chat/completions")

        captured, patcher = self._capture(FakeResponse(json.dumps({"data": [{"id": "my-model"}]}).encode()))
        with patcher:
            providers.list_models(config, "local")
        self.assertEqual(captured["request"].full_url, "https://llm.example/v1/models")

    def test_keyless_provider_is_allowed_and_auth_errors_are_generic(self) -> None:
        captured, patcher = self._capture(FakeResponse(
            json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        ))
        with patcher:
            providers.chat("ollama", "", "qwen2.5:7b", [{"role": "user", "content": "q"}])
        self.assertNotIn("authorization", _headers(captured["request"]))

        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions", 401, "Unauthorized",
            {}, io.BytesIO(b'{"error":{"message":"invalid key sk-secret"}}'),
        )
        with mock.patch.object(providers.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(providers.ProviderError) as context:
                providers.chat("openai", "sk-secret", "gpt-4.1-mini",
                               [{"role": "user", "content": "q"}])
        self.assertNotIn("sk-secret", str(context.exception))
        self.assertEqual(context.exception.status, 401)

        upstream = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions", 500, "Server Error",
            {}, io.BytesIO(b'{"error":{"message":"upstream rejected sk-secret"}}'),
        )
        with mock.patch.object(providers.urllib.request, "urlopen", side_effect=upstream):
            with self.assertRaises(providers.ProviderError) as context:
                providers.chat("openai", "sk-secret", "gpt-4.1-mini",
                               [{"role": "user", "content": "q"}])
        self.assertNotIn("sk-secret", str(context.exception))
        self.assertIn("***", str(context.exception))


if __name__ == "__main__":
    unittest.main()
