"""
model_context.py

Query and cache model context window sizes from OpenAI-compatible APIs.
Provides token estimation for context usage tracking.
"""

import logging
import ipaddress
import socket
import sys
from typing import Dict, List, Optional, Tuple, TypedDict

from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}
_PRIVATE_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                     "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                     "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                     "172.30.", "172.31.", "192.168.", "100.")


class RuntimeCapabilities(TypedDict):
    """Capabilities exposed by the model's active serving runtime."""

    context_length: int
    parallel_slots: Optional[int]
    source: str


def _normalize_base_for_compare(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/models", "/completions", "/v1/messages"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url


def _configured_endpoint_kind(url: str) -> Optional[str]:
    """Return configured endpoint kind for a chat/base URL when available."""
    target = _normalize_base_for_compare(url)
    if not target:
        return None
    if "core.database" not in sys.modules:
        return None
    try:
        from core.database import SessionLocal, ModelEndpoint
        db = SessionLocal()
        try:
            rows = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True).all()
            for ep in rows:
                base = _normalize_base_for_compare(getattr(ep, "base_url", "") or "")
                if not base:
                    continue
                if target != base and not target.startswith(base + "/"):
                    continue
                kind = (getattr(ep, "endpoint_kind", None) or "auto").strip().lower()
                if kind in ("local", "api", "proxy"):
                    return kind
                if getattr(ep, "api_key", None):
                    parsed = urlparse(base)
                    host = (parsed.hostname or "").lower()
                    path = (parsed.path or "").rstrip("/")
                    if parsed.port != 11434 and "ollama" not in host and (path.endswith("/v1") or "/openai" in path):
                        return "proxy"
                return "auto"
        finally:
            db.close()
    except Exception:
        return None


def _is_local_endpoint(url: str) -> bool:
    """Check if URL points to a local/private/tailscale address."""
    kind = _configured_endpoint_kind(url)
    if kind in ("api", "proxy"):
        return False
    if kind == "local":
        return True
    try:
        host = urlparse(url).hostname or ""
        if host in _LOCAL_HOSTS or _is_private_address(host):
            return True
        # Docker service names such as ``llama-hermes`` are not IP-shaped.
        # Resolve them and treat the endpoint as local only when DNS returns a
        # private address. Explicit API/proxy endpoint kinds above still win.
        for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
            address = info[4][0]
            if _is_private_address(address):
                return True
        return False
    except Exception:
        return False


def _is_private_address(value: str) -> bool:
    """Return whether a hostname/IP literal is local, private, or Tailscale."""
    if not value:
        return False
    value = value.split("%", 1)[0]
    if value.startswith(_PRIVATE_PREFIXES):
        return True
    try:
        address = ipaddress.ip_address(value)
        return address.is_private or address.is_loopback or address.is_link_local
    except ValueError:
        return False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONTEXT = 128000
REQUEST_TIMEOUT = 5

# Known context windows for major API models (used as fallback when /models
# endpoint doesn't report context_length).
# Substring matching — use the shortest unique prefix so variants get caught.
KNOWN_CONTEXT_WINDOWS = {
    # --- Anthropic ---
    'claude-sonnet-4-5': 200000,
    'claude-sonnet-4-6': 200000,
    'claude-sonnet-4': 200000,
    'claude-opus-4': 200000,
    'claude-haiku-4': 200000,
    'claude-haiku-3-5': 200000,
    'claude-3-5-sonnet': 200000,
    'claude-3-5-haiku': 200000,
    'claude-3-opus': 200000,
    'claude-3-sonnet': 200000,
    'claude-3-haiku': 200000,

    # --- OpenAI ---
    'gpt-5': 400000,
    'gpt-4.1': 1047576,
    'gpt-4.1-mini': 1047576,
    'gpt-4.1-nano': 1047576,
    'gpt-4o': 128000,
    'gpt-4o-mini': 128000,
    'gpt-4-turbo': 128000,
    'gpt-4': 8192,
    'gpt-3.5-turbo': 16385,
    'o1': 200000,
    'o1-mini': 128000,
    'o1-pro': 200000,
    'o3': 200000,
    'o3-mini': 200000,
    'o4-mini': 200000,

    # --- DeepSeek ---
    'deepseek-chat': 64000,
    'deepseek-coder': 64000,
    'deepseek-reasoner': 64000,
    'deepseek-r1': 64000,
    'deepseek-v3': 64000,
    'deepseek-v2': 64000,

    # --- Google ---
    'gemini-2.5-pro': 1048576,
    'gemini-2.5-flash': 1048576,
    'gemini-2.0-flash': 1048576,
    'gemini-1.5-pro': 1048576,
    'gemini-1.5-flash': 1048576,
    'gemma-4': 262144,
    'gemma-3': 128000,
    'gemma-2': 8192,

    # --- Mistral ---
    'mistral-large': 128000,
    'mistral-medium': 32000,
    'mistral-small': 32000,
    'mistral-nemo': 128000,
    'mistral-7b': 32000,
    'mixtral': 32000,
    'codestral': 32000,
    'pixtral': 128000,

    # --- xAI ---
    'grok-4': 131072,
    'grok-3': 131072,
    'grok-2': 131072,

    # --- Meta / Llama ---
    'llama-4': 1048576,
    'llama-3.3': 131072,
    'llama-3.2': 131072,
    'llama-3.1': 131072,
    'llama-3': 131072,

    # --- Qwen ---
    'qwen3': 131072,
    'qwen2.5': 131072,
    'qwen2': 32768,
    'qwq': 32768,

    # --- Cohere ---
    'command-r-plus': 128000,
    'command-r': 128000,
    'command-a': 256000,

    # --- Perplexity ---
    'sonar-pro': 200000,
    'sonar': 128000,

    # --- MiniMax ---
    'minimax': 1000000,

    # --- Moonshot / Kimi ---
    'moonshot': 128000,
    'kimi': 128000,

    # --- Microsoft ---
    'phi-4': 16000,
    'phi-3': 128000,

    # --- Nvidia ---
    'nemotron': 131072,

    # --- Yi ---
    'yi-large': 32768,
    'yi-1.5': 16384,

    # --- 01.ai ---
    'yi-lightning': 16384,

    # --- Nous ---
    'hermes': 131072,
    'nous-hermes': 131072,

    # --- Open community ---
    'dolphin': 32768,
    'mythomax': 4096,
    'wizard': 32768,
    'openchat': 8192,
    'solar': 32768,
}

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_context_cache: Dict[Tuple[str, str], int] = {}


def get_context_length(endpoint_url: str, model: str) -> int:
    """Get the context window size for a model.

    Queries /v1/models on the endpoint and looks for context_length
    or context_window fields. Caches result per (endpoint, model).
    Falls back to DEFAULT_CONTEXT if unavailable.
    """
    configured_kind = _configured_endpoint_kind(endpoint_url)
    is_local = _is_local_endpoint(endpoint_url)
    # Key on (endpoint_url, model): the same model id can be served by two
    # different remote endpoints with different real context windows (e.g. a
    # capped proxy vs. the full provider), so caching by model id alone would
    # serve one endpoint's window for the other (issue #2603).
    cache_key = (endpoint_url, model)
    if not is_local and cache_key in _context_cache:
        return _context_cache[cache_key]

    ctx = _query_context_length(endpoint_url, model)
    # Only cache non-default values to allow retry on next request.
    # Local endpoints can restart with a different --max-model-len while keeping
    # the same model id, so always re-query them instead of serving stale cache.
    if not is_local and (ctx != DEFAULT_CONTEXT or configured_kind in ("api", "proxy")):
        _context_cache[cache_key] = ctx
    logger.info(f"Context length for {model}: {ctx}")
    return ctx


def get_runtime_capabilities(endpoint_url: str, model: str) -> RuntimeCapabilities:
    """Discover the active serving context and parallel request capacity.

    llama.cpp's ``/slots`` response is preferred because it reports the
    runtime's actual ``--ctx-size`` and ``--parallel`` values. Other endpoints
    fall back to model API metadata, known model limits, then DEFAULT_CONTEXT.
    """
    known = _lookup_known(model)
    configured_kind = _configured_endpoint_kind(endpoint_url)
    is_local = _is_local_endpoint(endpoint_url)

    if configured_kind in ("api", "proxy"):
        return {
            "context_length": known or DEFAULT_CONTEXT,
            "parallel_slots": None,
            "source": "known model" if known else "default",
        }

    if is_local:
        slots = _query_llama_slots(endpoint_url)
        if slots:
            contexts = [
                slot.get("n_ctx") for slot in slots
                if isinstance(slot, dict)
                and isinstance(slot.get("n_ctx"), int)
                and slot.get("n_ctx") > 0
            ]
            if contexts:
                context_length = min(contexts)
                logger.info(
                    "llama.cpp /slots reports n_ctx=%s and %s slot(s) for %s",
                    context_length, len(slots), model,
                )
                return {
                    "context_length": context_length,
                    "parallel_slots": len(slots),
                    "source": "llama.cpp /slots",
                }

    api_ctx = _query_models_context(endpoint_url, model)
    if api_ctx and known:
        if is_local and api_ctx < known:
            logger.info(
                "Local endpoint reports %s for %s (known max: %s) - using API value",
                api_ctx, model, known,
            )
            context_length = api_ctx
        else:
            context_length = max(api_ctx, known)
    else:
        context_length = api_ctx or known or DEFAULT_CONTEXT

    source = "model API" if api_ctx else ("known model" if known else "default")
    return {
        "context_length": context_length,
        "parallel_slots": None,
        "source": source,
    }


def _lookup_known(model: str) -> Optional[int]:
    """Check known context windows by substring match.

    Picks the LONGEST matching key so a short key never shadows a more specific
    one. Without this, 'o1' (200k) precedes 'o1-mini' (128k) in the table and a
    first-match return would report o1-mini's window as 200k.
    """
    name = model.lower()
    basename = name.split("/")[-1] if "/" in name else name
    basename = basename.split(":")[0]  # strip :free, :extended etc.
    best_key: Optional[str] = None
    best_ctx: Optional[int] = None
    for key, ctx in KNOWN_CONTEXT_WINDOWS.items():
        if key in basename or key in name:
            if best_key is None or len(key) > len(best_key):
                best_key, best_ctx = key, ctx
    return best_ctx


def _query_context_length(endpoint_url: str, model: str) -> int:
    """Query the model API for context length."""
    return get_runtime_capabilities(endpoint_url, model)["context_length"]


def _endpoint_root(endpoint_url: str) -> str:
    """Return the server root for an OpenAI-compatible endpoint URL."""
    endpoint_url = (endpoint_url or "").strip().rstrip("/")
    if "/v1" in endpoint_url:
        return endpoint_url.split("/v1", 1)[0]
    return _normalize_base_for_compare(endpoint_url)


def _query_llama_slots(endpoint_url: str) -> Optional[List[Dict]]:
    try:
        response = httpx.get(f"{_endpoint_root(endpoint_url)}/slots", timeout=REQUEST_TIMEOUT)
        slots = response.json() if response.is_success else None
        return slots if isinstance(slots, list) and slots else None
    except Exception as exc:
        logger.debug("Failed to query llama.cpp slots: %s", exc)
        return None


def _query_models_context(endpoint_url: str, model: str) -> Optional[int]:
    """Read context metadata for one model from an OpenAI-compatible API."""
    api_ctx = None

    # GitHub Copilot's /models endpoint needs special auth headers. If this is
    # a Copilot base URL, skip probing and let get_runtime_capabilities() fall
    # back to known model metadata/defaults.
    try:
        from src.copilot import is_copilot_base
        if is_copilot_base(endpoint_url):
            return None
    except Exception:
        pass

    models_url = endpoint_url.replace("/chat/completions", "/models")
    try:
        r = httpx.get(models_url, timeout=REQUEST_TIMEOUT)
        if r.is_success:
            data = r.json()
            models_list = data.get("data") or []

            for m in models_list:
                mid = m.get("id", "")
                if mid == model or mid.split("/")[-1] == model.split("/")[-1]:
                    for field in (
                        "context_length",
                        "context_window",
                        "max_model_len",
                        "max_context_length",
                        "max_seq_len",
                    ):
                        val = m.get(field)
                        if val and isinstance(val, (int, float)) and val > 0:
                            api_ctx = int(val)
                            break

                    if not api_ctx:
                        meta = m.get("meta") or m.get("model_extra") or {}
                        if isinstance(meta, dict):
                            # n_ctx is the actual serving context (set via -c flag in llama.cpp)
                            for field in ("n_ctx", "context_length", "context_window", "max_model_len"):
                                val = meta.get(field)
                                if val and isinstance(val, (int, float)) and val > 0:
                                    api_ctx = int(val)
                                    break
                    break
    except Exception as e:
        logger.debug(f"Failed to query context length for {model}: {e}")
    return api_ctx


def estimate_tokens(messages: List[Dict]) -> int:
    """Rough token estimate for a list of messages.

    Uses chars * 0.3 which is closer to real BPE tokenizer output
    than the commonly-cited chars/4 (which underestimates by ~20-30%).
    Also adds ~4 tokens per message for role/formatting overhead.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead (role, separators)
        content = msg.get("content", "")
        if isinstance(content, str):
            total += int(len(content) * 0.3)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total += int(len(item.get("text", "")) * 0.3)
    return total
