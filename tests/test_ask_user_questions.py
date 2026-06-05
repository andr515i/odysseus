import json
import sys
from unittest.mock import MagicMock


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
from src.tool_parsing import parse_tool_blocks
from src.tool_schemas import function_call_to_tool_block


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
