"""Helpers for assistant question events in agent/planning flows."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List


DEFAULT_QUESTION = "What would you like to do next?"


def _clean_string(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _normalize_choices(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []

    choices: List[Dict[str, str]] = []
    for item in value[:8]:
        if isinstance(item, dict):
            label = _clean_string(item.get("label") or item.get("text") or item.get("value"), 120)
            if not label:
                continue
            choice = {
                "label": label,
                "value": _clean_string(item.get("value") or label, 240),
            }
            description = _clean_string(item.get("description") or item.get("hint"), 240)
            if description:
                choice["description"] = description
            choices.append(choice)
        else:
            label = _clean_string(item, 120)
            if label:
                choices.append({"label": label, "value": label})
    return choices


def normalize_question_payload(content: Any) -> Dict[str, Any]:
    """Return a stable assistant_question payload from JSON or plain text."""

    payload: Dict[str, Any]
    if isinstance(content, dict):
        payload = dict(content)
    else:
        text = str(content or "").strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                payload = parsed if isinstance(parsed, dict) else {"question": text}
            except json.JSONDecodeError:
                payload = {"question": text}
        else:
            payload = {"question": text}

    question = _clean_string(
        payload.get("question") or payload.get("prompt") or payload.get("text"),
        2000,
    ) or DEFAULT_QUESTION
    question_id = _clean_string(payload.get("question_id") or payload.get("id"), 80)
    if not question_id:
        question_id = "ask_" + uuid.uuid4().hex[:12]

    allow_free_text = payload.get("allow_free_text", True)
    if not isinstance(allow_free_text, bool):
        allow_free_text = str(allow_free_text).strip().lower() not in {"0", "false", "no", "off"}

    return {
        "question_id": question_id,
        "question": question,
        "choices": _normalize_choices(payload.get("choices") or payload.get("options")),
        "allow_free_text": allow_free_text,
    }


def question_message_content(payload: Dict[str, Any], preface: str = "") -> str:
    """Plain assistant message body stored beside structured question metadata."""

    parts = []
    if preface:
        parts.append(str(preface).strip())
    parts.append("I need your input to continue.")
    question = _clean_string(payload.get("question"), 2000)
    if question:
        parts.append(f"Question: {question}")
    return "\n\n".join(p for p in parts if p)
