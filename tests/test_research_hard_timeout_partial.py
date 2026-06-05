import asyncio

from src.research_handler import ResearchHandler


class _PartialResearcher:
    def __init__(self):
        self.evolving_report = ""
        self.findings = [{
            "url": "https://example.test/source",
            "title": "Useful source",
            "summary": "A usable extracted finding.",
        }]

    def _fallback_report(self, question, findings):
        return f"# {question}\n\n{findings[0]['summary']}"

    def _with_adaptive_warnings(self, report):
        return "> **Research limitations:** synthesis timed out.\n\n" + report

    def get_stats(self):
        return {"Rounds": 1, "Queries": 2, "URLs": 1, "Context": "4096 tokens", "Slots": 1}


def test_hard_timeout_saves_findings_when_synthesis_has_no_report():
    handler = ResearchHandler.__new__(ResearchHandler)
    handler._legacy_engine = None
    handler._active_tasks = {}
    saved = []
    handler._save_result = lambda session_id, entry: saved.append((session_id, dict(entry)))

    async def stalled_call(*args, **kwargs):
        kwargs["_task_entry"]["researcher"] = _PartialResearcher()
        await asyncio.sleep(1)

    handler.call_research_service = stalled_call

    async def run():
        handler.start_research(
            "timeout-partial",
            "research question",
            "http://llama:8000/v1/chat/completions",
            "model",
            hard_timeout=0.01,
        )
        await handler._active_tasks["timeout-partial"]["task"]

    asyncio.run(run())

    entry = handler._active_tasks["timeout-partial"]
    assert entry["status"] == "done"
    assert "A usable extracted finding" in entry["result"]
    assert entry["raw_report"].startswith("> **Research limitations:**")
    assert entry["stats"]["Context"] == "4096 tokens"
    assert saved and saved[0][0] == "timeout-partial"
