import asyncio
import json
import sys
import types

from src.deep_research import DeepResearcher
from src.model_context import estimate_tokens


def _researcher(**kwargs):
    return DeepResearcher(
        llm_endpoint="http://llama-hermes:8000/v1/chat/completions",
        llm_model="Hermes-2-Pro-Mistral-7B.Q6_K.gguf",
        **kwargs,
    )


def test_runtime_capabilities_clamp_extraction_concurrency(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.deep_research.get_runtime_capabilities",
        lambda endpoint, model: {
            "context_length": 4096,
            "parallel_slots": 1,
            "source": "llama.cpp /slots",
        },
    )
    researcher = _researcher(extraction_concurrency=3, progress_callback=events.append)

    asyncio.run(researcher._configure_runtime_capabilities())

    assert researcher.runtime_context_tokens == 4096
    assert researcher.safe_context_tokens == int(4096 * 0.9)
    assert researcher.extraction_concurrency == 1
    assert any(event.get("phase") == "warning" for event in events)
    assert any("concurrency" in warning.lower() for warning in researcher.adaptive_warnings)


def test_small_context_never_receives_oversized_research_calls(monkeypatch):
    captured = []
    llm_core = types.ModuleType("src.llm_core")

    async def fake_llm_call_async(**kwargs):
        captured.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "Final Output Format" in prompt:
            return json.dumps({
                "rational": "relevant",
                "evidence": "evidence",
                "summary": "useful content",
            })
        return "complete report " * 500

    llm_core.llm_call_async = fake_llm_call_async
    monkeypatch.setitem(sys.modules, "src.llm_core", llm_core)

    search_mod = types.ModuleType("src.search")
    search_mod.fetch_webpage_content = lambda url, timeout: {
        "success": True,
        "content": "webpage evidence " * 5000,
        "title": "Page",
        "og_image": "",
    }
    monkeypatch.setitem(sys.modules, "src.search", search_mod)

    events = []
    researcher = _researcher(
        max_report_tokens=8192,
        max_content_chars=100000,
        progress_callback=events.append,
    )
    researcher.runtime_context_tokens = 4096
    researcher.safe_context_tokens = int(4096 * 0.9)
    researcher.runtime_slots = 1

    finding = {
        "url": "https://example.test",
        "title": "Long finding",
        "summary": "finding evidence " * 3000,
    }
    extraction = asyncio.run(
        researcher._fetch_and_extract("https://example.test", "question", "Page")
    )
    synthesis = asyncio.run(
        researcher._synthesize("question", [finding] * 12, "old report " * 5000)
    )
    final = asyncio.run(researcher._final_report("question", synthesis))

    assert extraction
    assert final
    assert len(captured) >= 3
    for call in captured:
        assert estimate_tokens(call["messages"]) + call["max_tokens"] <= researcher.safe_context_tokens

    rendered = researcher._with_adaptive_warnings(final)
    assert rendered.startswith("> **Research limitations:**")
    assert any(event.get("phase") == "warning" for event in events)


def test_no_findings_message_is_actionable():
    researcher = _researcher()
    researcher.runtime_context_tokens = 4096
    researcher.runtime_slots = 1
    researcher._last_llm_error = "request exceeds available context"

    report = researcher._no_findings_report()

    assert "could not produce usable findings" in report
    assert "request exceeds available context" in report
    assert "4096 context tokens" in report
    assert "No information could be gathered" not in report
