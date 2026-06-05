"""Tests for model_context.py — local endpoint detection, token estimation, known model lookup."""

import sys
import types

import pytest

import src.model_context as model_context
from src.model_context import _is_local_endpoint, estimate_tokens, _lookup_known


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return ("eq", self.name, value)


class _ModelEndpoint:
    is_enabled = _Column("is_enabled")


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *conditions):
        for condition in conditions:
            if isinstance(condition, tuple) and condition[0] == "eq":
                _, field, value = condition
                self.rows = [row for row in self.rows if getattr(row, field) == value]
        return self

    def all(self):
        return list(self.rows)


class _Db:
    def __init__(self, rows):
        self.rows = rows

    def query(self, model):
        return _Query(self.rows)

    def close(self):
        pass


def _install_endpoint_db(monkeypatch, rows):
    mod = types.ModuleType("core.database")
    mod.ModelEndpoint = _ModelEndpoint
    mod.SessionLocal = lambda: _Db(rows)
    monkeypatch.setitem(sys.modules, "core.database", mod)


class TestIsLocalEndpoint:
    def test_localhost(self):
        assert _is_local_endpoint("http://localhost:5000/v1/chat/completions") is True

    def test_loopback_ipv4(self):
        assert _is_local_endpoint("http://127.0.0.1:8080/v1/chat/completions") is True

    def test_private_192_168(self):
        assert _is_local_endpoint("http://192.168.1.1:11434/v1/chat/completions") is True

    def test_private_10(self):
        assert _is_local_endpoint("http://10.0.0.5:8000/v1/chat/completions") is True

    def test_tailscale_100(self):
        # 100.64.0.0/10 is the CGNAT range Tailscale uses.
        assert _is_local_endpoint("http://100.64.0.1:5000/v1/chat/completions") is True

    def test_configured_tailscale_proxy_is_remote(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://100.117.136.97:34521/v1",
                endpoint_kind="proxy",
                api_key="fake-key",
                is_enabled=True,
            )
        ])

        assert _is_local_endpoint("http://100.117.136.97:34521/v1/chat/completions") is False

    def test_openai_is_remote(self):
        assert _is_local_endpoint("https://api.openai.com/v1/chat/completions") is False

    def test_anthropic_is_remote(self):
        assert _is_local_endpoint("https://api.anthropic.com/v1/messages") is False

    def test_empty_url(self):
        assert _is_local_endpoint("") is False

    def test_malformed_url(self):
        assert _is_local_endpoint("not-a-url") is False

    def test_docker_service_hostname_resolving_private_is_local(self, monkeypatch):
        monkeypatch.setattr(
            model_context.socket,
            "getaddrinfo",
            lambda *args, **kwargs: [
                (model_context.socket.AF_INET, model_context.socket.SOCK_STREAM, 6, "", ("172.20.0.4", 0))
            ],
        )

        assert _is_local_endpoint("http://llama-hermes:8000/v1/chat/completions") is True


class TestEstimateTokens:
    def test_empty_list(self):
        assert estimate_tokens([]) == 0

    def test_single_short_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        tokens = estimate_tokens(messages)
        # 4 overhead + int(5 * 0.3) = 4 + 1 = 5
        assert tokens == 5

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there"},
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 0
        # Each message adds 4 overhead + chars * 0.3
        assert tokens == 4 + int(16 * 0.3) + 4 + int(8 * 0.3)

    def test_multimodal_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        # 4 overhead + int(19 * 0.3) for the text item; image_url is ignored
        assert tokens == 4 + int(19 * 0.3)

    def test_missing_content_key(self):
        messages = [{"role": "assistant"}]
        tokens = estimate_tokens(messages)
        # 4 overhead + 0 content
        assert tokens == 4

    def test_scales_with_length(self):
        short = estimate_tokens([{"role": "user", "content": "short"}])
        long_text = "a" * 10000
        long = estimate_tokens([{"role": "user", "content": long_text}])
        assert long > short * 10


class TestLookupKnown:
    def test_claude_sonnet(self):
        assert _lookup_known("claude-sonnet-4-5") == 200000

    def test_gpt4o(self):
        assert _lookup_known("gpt-4o") == 128000

    def test_deepseek_r1(self):
        assert _lookup_known("deepseek-r1") == 64000

    def test_gemini_pro(self):
        assert _lookup_known("gemini-2.5-pro") == 1048576

    def test_unknown_model(self):
        assert _lookup_known("totally-unknown-model-xyz") is None

    def test_namespaced_model(self):
        """Models prefixed with provider/ should still match."""
        result = _lookup_known("openrouter/deepseek-r1")
        assert result == 64000

    def test_model_with_tag(self):
        """Models with :free or :extended suffixes should still match."""
        result = _lookup_known("deepseek-r1:free")
        assert result == 64000

    def test_o1_mini_not_shadowed_by_o1(self):
        """'o1' (200k) precedes 'o1-mini' (128k) in the table; longest match wins."""
        assert _lookup_known("o1-mini") == 128000

    def test_o1_full(self):
        assert _lookup_known("o1") == 200000

    def test_gpt4o_mini_not_shadowed_by_gpt4(self):
        assert _lookup_known("gpt-4o-mini") == 128000

    def test_gpt4_base(self):
        assert _lookup_known("gpt-4") == 8192


class TestGetContextLength:
    def setup_method(self):
        model_context._context_cache.clear()

    def test_local_endpoint_requeries_same_model_after_restart(self, monkeypatch):
        calls = []

        def fake_query(endpoint_url, model):
            calls.append((endpoint_url, model))
            return 8192 if len(calls) == 1 else 27000

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)

        endpoint = "http://127.0.0.1:8000/v1/chat/completions"
        model = "Qwen/Qwen3-14B"

        first = model_context.get_context_length(endpoint, model)
        second = model_context.get_context_length(endpoint, model)

        assert first == 8192
        assert second == 27000
        assert len(calls) == 2

    def test_remote_endpoint_keeps_cached_context(self, monkeypatch):
        calls = []

        def fake_query(endpoint_url, model):
            calls.append((endpoint_url, model))
            return 200000 if len(calls) == 1 else 12345

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)

        endpoint = "https://api.openai.com/v1/chat/completions"
        model = "gpt-5"

        first = model_context.get_context_length(endpoint, model)
        second = model_context.get_context_length(endpoint, model)

        assert first == 200000
        assert second == 200000
        assert len(calls) == 1

    def test_configured_proxy_uses_default_without_model_listing(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://100.117.136.97:34521/v1",
                endpoint_kind="proxy",
                api_key="fake-key",
                is_enabled=True,
            )
        ])
        calls = []

        def fake_get(*args, **kwargs):
            calls.append(args)
            raise AssertionError("/models should not be queried for configured proxy context")

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        first = model_context.get_context_length(endpoint, "unknown-proxy-model")
        second = model_context.get_context_length(endpoint, "unknown-proxy-model")

        assert first == model_context.DEFAULT_CONTEXT
        assert second == model_context.DEFAULT_CONTEXT
        assert calls == []


class TestRuntimeCapabilities:
    def test_llama_slots_override_known_model_context(self, monkeypatch):
        monkeypatch.setattr(model_context, "_is_local_endpoint", lambda url: True)

        class _Response:
            is_success = True

            def json(self):
                return [{"id": 0, "n_ctx": 4096}]

        monkeypatch.setattr(model_context.httpx, "get", lambda *args, **kwargs: _Response())

        capabilities = model_context.get_runtime_capabilities(
            "http://llama-hermes:8000/v1/chat/completions",
            "Hermes-2-Pro-Mistral-7B.Q6_K.gguf",
        )

        assert capabilities == {
            "context_length": 4096,
            "parallel_slots": 1,
            "source": "llama.cpp /slots",
        }

    def test_slot_count_reports_parallel_capacity(self, monkeypatch):
        monkeypatch.setattr(model_context, "_is_local_endpoint", lambda url: True)

        class _Response:
            is_success = True

            def json(self):
                return [{"id": 0, "n_ctx": 8192}, {"id": 1, "n_ctx": 8192}]

        monkeypatch.setattr(model_context.httpx, "get", lambda *args, **kwargs: _Response())

        capabilities = model_context.get_runtime_capabilities(
            "http://llama:8000/v1/chat/completions", "local-model",
        )

        assert capabilities["context_length"] == 8192
        assert capabilities["parallel_slots"] == 2
