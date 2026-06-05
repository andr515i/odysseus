import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


REPO_ROOT = Path(__file__).resolve().parents[1]

for mod in ["src.agent_tools", "src.tool_parsing", "src.tool_schemas", "src.tool_execution"]:
    sys.modules.pop(mod, None)

for mod in [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext", "sqlalchemy.ext.declarative",
    "sqlalchemy.ext.hybrid", "sqlalchemy.sql", "sqlalchemy.sql.expression",
    "src.database", "core.models", "core.database", "core.auth",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()


from src.agent_questions import normalize_question_payload, question_message_content
from src.agent_tools import TOOL_TAGS
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block


def test_normalize_question_payload_accepts_plain_text():
    payload = normalize_question_payload("Which scope should I use?")

    assert payload["question"] == "Which scope should I use?"
    assert payload["question_id"].startswith("ask_")
    assert payload["choices"] == []
    assert payload["allow_free_text"] is True


def test_normalize_question_payload_normalizes_choice_shapes():
    payload = normalize_question_payload({
        "question_id": "scope",
        "question": "Pick scope",
        "choices": [
            "Small",
            {"label": "Full", "value": "full-plan", "description": "Do the whole thing"},
        ],
        "allow_free_text": "false",
    })

    assert payload["question_id"] == "scope"
    assert payload["choices"] == [
        {"label": "Small", "value": "Small"},
        {"label": "Full", "value": "full-plan", "description": "Do the whole thing"},
    ]
    assert payload["allow_free_text"] is False


def test_question_message_content_includes_plain_text_question():
    content = question_message_content({
        "question": "Which scope should I use?",
        "choices": ["Small", "Full"],
    })

    assert "I need your input to continue." in content
    assert "Question: Which scope should I use?" in content


def test_ask_user_fenced_tool_block_parses():
    text = """Before I continue:

```ask_user
{"question": "Which option?", "choices": ["A", "B"]}
```
"""

    blocks = parse_tool_blocks(text)

    assert "ask_user" in TOOL_TAGS
    assert len(blocks) == 1
    assert blocks[0].tool_type == "ask_user"
    assert json.loads(blocks[0].content)["question"] == "Which option?"


def test_ask_user_native_function_call_converts_to_tool_block():
    block = function_call_to_tool_block(
        "ask_user",
        json.dumps({
            "question": "Proceed?",
            "choices": [{"label": "Yes", "value": "yes"}],
            "allow_free_text": False,
            "question_id": "q1",
        }),
    )

    assert block is not None
    assert block.tool_type == "ask_user"
    payload = json.loads(block.content)
    assert payload == {
        "question": "Proceed?",
        "choices": [{"label": "Yes", "value": "yes"}],
        "allow_free_text": False,
        "question_id": "q1",
    }


def test_parse_invoke_tool_ask_user_json_block():
    from src.tool_parsing import parse_tool_blocks, strip_tool_blocks
    from src.agent_questions import normalize_question_payload

    text = """
<invoke_tool>
{
  "name": "ask_user",
  "arguments": {
    "question": "Would you like a minimal or a robust implementation? (Or describe your preference)"
  }
}
</invoke_tool>
"""

    blocks = parse_tool_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "ask_user"

    payload = normalize_question_payload(blocks[0].content)
    assert payload["question"] == "Would you like a minimal or a robust implementation? (Or describe your preference)"

    cleaned = strip_tool_blocks(text)
    assert "<invoke_tool>" not in cleaned
    assert "ask_user" not in cleaned


def _parse_single_ask_payload(text):
    blocks = parse_tool_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "ask_user"
    return json.loads(blocks[0].content)


def test_parse_direct_ask_user_string_call():
    payload = _parse_single_ask_payload('ask_user("Minimal or robust?")')

    assert payload["question"] == "Minimal or robust?"
    assert payload["choices"] == []
    assert payload["allow_free_text"] is True


def test_parse_direct_ask_user_question_keyword_call():
    payload = _parse_single_ask_payload('ask_user(question="Minimal or robust?")')

    assert payload["question"] == "Minimal or robust?"
    assert payload["choices"] == []
    assert payload["allow_free_text"] is True


def test_parse_direct_ask_user_json_object_call():
    payload = _parse_single_ask_payload(
        'ask_user({"question": "Minimal or robust?", "choices": ["Minimal", "Robust"], "allow_free_text": true})'
    )

    assert payload["question"] == "Minimal or robust?"
    assert payload["choices"] == ["Minimal", "Robust"]
    assert payload["allow_free_text"] is True


def test_parse_bare_json_ask_user_message():
    payload = _parse_single_ask_payload(
        json.dumps({"tool": "ask_user", "message": "Minimal or robust?"})
    )

    assert payload["question"] == "Minimal or robust?"
    assert payload["choices"] == []
    assert payload["allow_free_text"] is True


def test_parse_fenced_json_bare_ask_user_message():
    text = """```json
{"tool": "ask_user", "message": "Minimal or robust?"}
```"""

    payload = _parse_single_ask_payload(text)

    assert payload["question"] == "Minimal or robust?"


def test_parse_bare_json_ask_user_arguments():
    payload = _parse_single_ask_payload(
        json.dumps({
            "tool": "ask_user",
            "arguments": {
                "question": "Minimal or robust?",
                "choices": ["Minimal", "Robust"],
                "allow_free_text": True,
            },
        })
    )

    assert payload["question"] == "Minimal or robust?"
    assert payload["choices"] == ["Minimal", "Robust"]
    assert payload["allow_free_text"] is True


def test_parse_fenced_python_single_direct_ask_as_ask_user():
    text = """```python
ask_user("Minimal or robust?")
```"""

    payload = _parse_single_ask_payload(text)

    assert payload["question"] == "Minimal or robust?"


def test_strip_tool_blocks_removes_direct_ask_user_calls():
    text = """Before
ask_user("Minimal or robust?")
After

```python
ask_user("Hidden?")
```
Done"""

    cleaned = strip_tool_blocks(text)

    assert "ask_user" not in cleaned
    assert "Before" in cleaned
    assert "After" in cleaned
    assert "Done" in cleaned


def test_strip_tool_blocks_removes_bare_json_ask_user_forms():
    bare = json.dumps({"tool": "ask_user", "message": "Minimal or robust?"})
    fenced = f"""Before
```json
{bare}
```
After"""

    assert strip_tool_blocks(bare) == ""
    cleaned = strip_tool_blocks(fenced)
    assert "ask_user" not in cleaned
    assert "Minimal or robust" not in cleaned
    assert "Before" in cleaned
    assert "After" in cleaned


def test_direct_ask_user_does_not_parse_inline_or_non_exact_fences():
    inline = 'Use ask_user("Minimal or robust?") in examples.'
    fenced = """```text
ask_user("Minimal or robust?")
print("done")
```"""

    assert parse_tool_blocks(inline) == []
    assert strip_tool_blocks(inline) == inline
    assert parse_tool_blocks(fenced) == []
    assert strip_tool_blocks(fenced) == fenced


def test_bare_json_ask_user_does_not_parse_or_strip_non_ask_json():
    normal = json.dumps({"tool": "bash", "message": "Minimal or robust?"})
    fenced = f"""```json
{normal}
```"""

    assert parse_tool_blocks(normal) == []
    assert strip_tool_blocks(normal) == normal
    assert parse_tool_blocks(fenced) == []
    assert strip_tool_blocks(fenced) == fenced


def test_ask_user_guidance_prefers_structured_choices():
    agent_loop = (REPO_ROOT / "src" / "agent_loop.py").read_text(encoding="utf-8")
    ask_schema = next(s["function"] for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == "ask_user")

    assert '"choices": [{"label": "Minimal", "value": "minimal"}, {"label": "Robust", "value": "robust"}]' in agent_loop
    assert "Plain `ask_user(\"...\")` is acceptable only when there are no obvious discrete options." in agent_loop
    assert "Plain ask_user(\"...\") is acceptable only" in ask_schema["description"]
    assert "Structured choices to render as buttons" in ask_schema["parameters"]["properties"]["choices"]["description"]


def test_assistant_question_answer_template_resumes_previous_request_safely():
    renderer = (REPO_ROOT / "static" / "js" / "chatRenderer.js").read_text(encoding="utf-8")

    assert "Question I am answering: ${questionText}" in renderer
    assert "Use this answer to continue the previous user request." in renderer
    assert "Do not invent a new task." in renderer
    assert "Do not create files, run tools, or implement anything unless the previous user request explicitly asked for that." in renderer
    assert "Continue from the plan you were building." not in renderer
