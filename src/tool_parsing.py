"""
tool_parsing.py

Regex-based parsing of tool invocations from LLM response text.
Supports fenced code blocks, [TOOL_CALL] blocks, and XML-style <invoke> blocks.
"""

import re
import json
import logging
import ast
from typing import List, Optional

from src.agent_tools import ToolBlock, TOOL_TAGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Pattern 1: ```bash ... ``` fenced code blocks
_TOOL_BLOCK_RE = re.compile(
    r"```(" + "|".join(TOOL_TAGS) + r")\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_ANY_FENCE_RE = re.compile(
    r"```[^\n`]*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_FENCE_LINE_RE = re.compile(r"^[ \t]*```")

# Pattern 2: [TOOL_CALL] ... [/TOOL_CALL] blocks (some models use this format)
# Matches: {tool => "shell", args => {--command "ls -la"}} etc.
_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALL\]\s*\{([\s\S]*?)\}\s*\[/TOOL_CALL\]",
    re.IGNORECASE,
)

# Pattern 3: XML-style tool calls (minimax, some other models)
# <minimax:tool_call><invoke name="bash"><parameter name="command">...</parameter></invoke></minimax:tool_call>
# Also handles: <tool_call><invoke ...>, <function_call><invoke ...>, plain <invoke ...>
_XML_TOOL_CALL_RE = re.compile(
    r"<(?:[\w]+:)?(?:tool_call|function_call)>\s*([\s\S]*?)</(?:[\w]+:)?(?:tool_call|function_call)>",
    re.IGNORECASE,
)
_XML_INVOKE_RE = re.compile(
    r'<invoke\s+name=["\'](\w+)["\']>\s*([\s\S]*?)</invoke>',
    re.IGNORECASE,
)

_XML_INVOKE_TOOL_RE = re.compile(
    r"<invoke_tool>\s*([\s\S]*?)</invoke_tool>",
    re.IGNORECASE,
)

_XML_PARAM_RE = re.compile(
    r'<parameter\s+name=["\'](\w+)["\']>([\s\S]*?)</parameter>',
    re.IGNORECASE,
)

# Pattern 4: <tool_code> blocks (MiniMax-M2.5 style)
# {tool => 'tool_name', args => '<param>value</param>'}
_TOOL_CODE_RE = re.compile(
    r"<tool_code>\s*\{([\s\S]*?)\}\s*</tool_code>",
    re.IGNORECASE,
)

_DIRECT_ASK_NAMES = {"ask_user", "ask", "clarify"}
_JSON_ASK_NAMES = _DIRECT_ASK_NAMES | {"clarification"}
_DIRECT_ASK_CALL_RE = re.compile(
    r"^\s*(ask_user|ask|clarify)\s*\(([\s\S]*)\)\s*$",
    re.IGNORECASE,
)

# Pattern 5: DeepSeek DSML markup leaking into content. When deepseek
# models can't emit structured tool_calls (e.g. we sent no tool schemas
# that round, or the API didn't parse them), they fall back to raw
# markup using fullwidth-pipe delimiters:
#   <｜｜DSML｜｜tool_calls>
#     <｜｜DSML｜｜invoke name="web_search">
#       <｜｜DSML｜｜parameter name="query" string="true">QUERY</｜｜DSML｜｜parameter>
#     </｜｜DSML｜｜invoke>
#   </｜｜DSML｜｜tool_calls>
# We normalize it into the standard <invoke>/<parameter> form so the
# existing XML parser + stripper handle it (parse → execute; strip →
# never show the garbage to the user). The pipe run is tolerant of
# fullwidth (U+FF5C) and ascii '|' in any count.
_DSML_PIPES = r"[｜|]+"
def _normalize_dsml(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "DSML" not in text:
        return text
    t = text
    t = re.sub(rf"<\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*tool_calls\s*>", "<tool_call>", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*/\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*tool_calls\s*>", "</tool_call>", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*invoke\s+name=", "<invoke name=", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*/\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*invoke\s*>", "</invoke>", t, flags=re.IGNORECASE)
    # parameter open tag — drop any extra attrs (e.g. string="true").
    t = re.sub(rf'<\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*parameter\s+name=(["\'][^"\']+["\'])[^>]*>',
               r"<parameter name=\1>", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*/\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*parameter\s*>", "</parameter>", t, flags=re.IGNORECASE)
    return t

# Map model tool names to our tool types
_TOOL_NAME_MAP = {
    "shell": "bash",
    "bash": "bash",
    "terminal": "bash",
    "command": "bash",
    "execute": "bash",
    "run": "bash",
    "python": "python",
    "code": "python",
    "search": "web_search",
    "web_search": "web_search",
    "websearch": "web_search",
    "google_search": "web_search",
    "google_search_retrieval": "web_search",
    "google_search_grounding": "web_search",
    "web_fetch": "web_fetch",
    "webfetch": "web_fetch",
    "fetch_url": "web_fetch",
    "fetch": "web_fetch",
    "ask_user": "ask_user",
    "ask": "ask_user",
    "question": "ask_user",
    "clarify": "ask_user",
    "clarification": "ask_user",
    "read": "read_file",
    "read_file": "read_file",
    "cat": "read_file",
    "write": "write_file",
    "write_file": "write_file",
    "save": "write_file",
    "document": "update_document",
    "update_document": "update_document",
    "create_document": "create_document",
    "edit": "edit_document",
    "edit_document": "edit_document",
    "search_chats": "search_chats",
    "search_conversations": "search_chats",
    "find_chat": "search_chats",
    "chat_with_model": "chat_with_model",
    "ask_model": "chat_with_model",
    "chat_model": "chat_with_model",
    "create_session": "create_session",
    "new_session": "create_session",
    "list_sessions": "list_sessions",
    "send_to_session": "send_to_session",
    "message_session": "send_to_session",
    "pipeline": "pipeline",
    "chain": "pipeline",
    "manage_session": "manage_session",
    "session_control": "manage_session",
    "manage_memory": "manage_memory",
    "memory": "manage_memory",
    "manage_tasks": "manage_tasks",
    "tasks": "manage_tasks",
    "schedule": "manage_tasks",
    "list_models": "list_models",
    "models": "list_models",
    "available_models": "list_models",
    "ui_control": "ui_control",
    "ui": "ui_control",
    "control": "ui_control",
    "api_call": "api_call",
    "api": "api_call",
    "integration": "api_call",
    "ask_teacher": "ask_teacher",
    "teacher": "ask_teacher",
    "manage_skills": "manage_skills",
    "skills": "manage_skills",
    "skill": "manage_skills",
    "suggest_document": "suggest_document",
    "suggest": "suggest_document",
    "review_document": "suggest_document",
    "manage_endpoints": "manage_endpoints",
    "endpoints": "manage_endpoints",
    "manage_mcp": "manage_mcp",
    "mcp_servers": "manage_mcp",
    "manage_webhooks": "manage_webhooks",
    "webhooks": "manage_webhooks",
    "manage_tokens": "manage_tokens",
    "tokens": "manage_tokens",
    "manage_documents": "manage_documents",
    "documents": "manage_documents",
    "manage_research": "manage_research",
    "list_research": "manage_research",
    "read_research": "manage_research",
    "open_research": "manage_research",
    "delete_research": "manage_research",
    "manage_settings": "manage_settings",
    "settings": "manage_settings",
    "preferences": "manage_settings",
    "manage_notes": "manage_notes",
    "notes": "manage_notes",
    "todo": "manage_notes",
    "todos": "manage_notes",
}


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def _literal_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _direct_ask_args(raw: str) -> Optional[dict]:
    """Return ask_user args for one exact direct ask call, or None."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or "\n" in text or "\r" in text:
        return None

    m = _DIRECT_ASK_CALL_RE.match(text)
    if not m or m.group(1).lower() not in _DIRECT_ASK_NAMES:
        return None

    arg_src = m.group(2).strip()
    if not arg_src:
        return None

    if arg_src.startswith("{") and arg_src.endswith("}"):
        try:
            payload = json.loads(arg_src)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("question"), str):
            return payload
        return None

    try:
        expr = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None

    if not isinstance(expr, ast.Call):
        return None
    if not isinstance(expr.func, ast.Name) or expr.func.id.lower() not in _DIRECT_ASK_NAMES:
        return None

    if len(expr.args) == 1 and not expr.keywords:
        question = _literal_str(expr.args[0])
        return {"question": question} if question is not None else None

    if not expr.args and len(expr.keywords) == 1 and expr.keywords[0].arg == "question":
        question = _literal_str(expr.keywords[0].value)
        return {"question": question} if question is not None else None

    return None


def _normalize_ask_json_payload(payload: object) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    raw_tool = (
        payload.get("tool")
        or payload.get("name")
        or payload.get("function")
        or payload.get("tool_name")
    )
    tool_name = str(raw_tool or "").strip().lower().replace("-", "_")
    if tool_name not in _JSON_ASK_NAMES:
        return None

    source = None
    if isinstance(payload.get("arguments"), dict):
        source = payload["arguments"]
    elif isinstance(payload.get("args"), dict):
        source = payload["args"]
    else:
        source = payload

    args = dict(source)
    if "question" not in args:
        for key in ("message", "prompt", "text"):
            if key in args:
                args["question"] = args[key]
                break
    if "choices" not in args and "options" in args:
        args["choices"] = args["options"]

    if not isinstance(args.get("question"), str):
        return None
    return args


def _bare_json_ask_args(raw: str) -> Optional[dict]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _normalize_ask_json_payload(payload)


def _ask_args_to_tool_block(args: Optional[dict]) -> Optional[ToolBlock]:
    if args is None:
        return None
    from src.tool_schemas import function_call_to_tool_block
    return function_call_to_tool_block("ask_user", json.dumps(args))


def _parse_direct_ask_call(raw: str) -> Optional[ToolBlock]:
    return _ask_args_to_tool_block(_direct_ask_args(raw))


def _parse_ask_compat_block(raw: str) -> Optional[ToolBlock]:
    args = _direct_ask_args(raw)
    if args is None:
        args = _bare_json_ask_args(raw)
    return _ask_args_to_tool_block(args)


def _span_overlaps(span, spans) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _strip_ask_compat_fences(text: str) -> str:
    def repl(match) -> str:
        body = match.group(1).strip()
        return "" if (_direct_ask_args(body) is not None or _bare_json_ask_args(body) is not None) else match.group(0)

    return _ANY_FENCE_RE.sub(repl, text)


def _strip_standalone_direct_ask_lines(text: str) -> str:
    lines = text.splitlines(keepends=True)
    in_fence = False
    kept = []
    for line in lines:
        if _FENCE_LINE_RE.match(line):
            kept.append(line)
            in_fence = not in_fence
            continue
        if not in_fence and _direct_ask_args(line.strip()) is not None:
            continue
        kept.append(line)
    return "".join(kept)


def _iter_lines_outside_fences(text: str):
    in_fence = False
    for line in text.splitlines():
        if _FENCE_LINE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield line


def _parse_tool_call_block(raw: str) -> Optional[ToolBlock]:
    """Parse a [TOOL_CALL] block into a ToolBlock.

    Handles formats like:
      {tool => "shell", args => {--command "ls -la"}}
      {tool: "shell", command: "ls -la"}
    """
    # Try to extract tool name
    tool_match = re.search(r'tool\s*(?:=>|:|=)\s*["\']?(\w+)["\']?', raw, re.IGNORECASE)
    if not tool_match:
        return None

    tool_name = tool_match.group(1).lower()
    # Fall back to the raw name when it's a real tool but not in the alias
    # map, so known tools (e.g. manage_calendar) aren't silently dropped.
    mapped = _TOOL_NAME_MAP.get(tool_name) or (tool_name if tool_name in TOOL_TAGS else None)
    if not mapped:
        return None

    # Extract the command/content — try several patterns
    content = None

    # Pattern: --command "value" or --command 'value'
    cmd_match = re.search(r'--command\s+["\'](.+?)["\']', raw, re.DOTALL)
    if cmd_match:
        content = cmd_match.group(1)

    # Pattern: command => "value" or command: "value"
    if not content:
        cmd_match = re.search(r'command\s*(?:=>|:|=)\s*["\'](.+?)["\']', raw, re.DOTALL)
        if cmd_match:
            content = cmd_match.group(1)

    # Pattern: args => {content} — extract everything inside the nested braces
    if not content:
        args_match = re.search(r'args\s*(?:=>|:|=)\s*\{([\s\S]*)\}', raw, re.DOTALL)
        if args_match:
            inner = args_match.group(1).strip()
            # Strip quotes and key prefixes
            inner = re.sub(r'^--?\w+\s+', '', inner)
            inner = inner.strip('\'"')
            if inner:
                content = inner

    # Pattern: query/path/code => "value"
    if not content:
        for key in ("query", "path", "code", "content", "text", "file"):
            m = re.search(rf'{key}\s*(?:=>|:|=)\s*["\'](.+?)["\']', raw, re.DOTALL)
            if m:
                content = m.group(1)
                break

    # Last resort: take everything after the tool declaration
    if not content:
        rest = raw[tool_match.end():].strip()
        rest = re.sub(r'^[,;]\s*', '', rest)
        rest = rest.strip('{} \t\n\'"')
        if rest:
            content = rest

    if content:
        return ToolBlock(mapped, content.strip())
    return None


def _parse_xml_invoke(inv_match) -> Optional[ToolBlock]:
    """Parse an <invoke name="tool"><parameter ...>...</parameter></invoke> match.

    Delegates content-shaping to function_call_to_tool_block — the SAME
    converter used for native function calls — so the full tool set (every
    name in TOOL_TAGS, plus email + MCP tools) and the correct per-tool
    content format are handled in ONE place. The previous version duplicated
    a partial, hand-maintained tool-name map plus a `key: value` serializer:
    any tool missing from that map (e.g. `manage_calendar`) was silently
    dropped, and JSON-arg tools got an unparseable `k: v` blob. Both bugs
    made deepseek's DSML `create_event` calls vanish with no execution.
    """
    # Lowercase the tool name: models often emit capitalized invoke names
    # (e.g. <invoke name="Bash">) and function_call_to_tool_block matches
    # case-sensitively against the lowercase _TOOL_NAME_MAP / TOOL_TAGS, so a
    # raw capitalized name would be silently dropped.
    tool_name = inv_match.group(1).lower()
    body = inv_match.group(2)
    params = {}
    for pm in _XML_PARAM_RE.finditer(body):
        params[pm.group(1)] = pm.group(2).strip()
    # Local import to avoid a circular import at module load.
    from src.tool_schemas import function_call_to_tool_block
    return function_call_to_tool_block(tool_name, json.dumps(params))


def _parse_tool_code_block(raw: str) -> Optional[ToolBlock]:
    """Parse a <tool_code>{tool => 'name', args => '...'}</tool_code> block (MiniMax style)."""
    # Extract tool name
    tool_match = re.search(r"tool\s*=>\s*['\"](\S+?)['\"]", raw)
    if not tool_match:
        return None
    tool_name = tool_match.group(1).lower().replace('-', '_')
    # Strip MCP prefixes like "mcp__server__" or "cli-mcp-server-"
    for prefix in ("mcp__", "cli_mcp_server_", "desktop_commander_", "mcp_code_executor_"):
        if tool_name.startswith(prefix):
            tool_name = tool_name[len(prefix):]
            break

    mapped = _TOOL_NAME_MAP.get(tool_name)

    # Extract args content
    args_match = re.search(r"args\s*=>\s*['\"]?\s*([\s\S]*?)\s*['\"]?\s*$", raw, re.DOTALL)
    args_body = args_match.group(1).strip().strip("'\"") if args_match else ""

    # Parse XML params inside args (e.g. <command>ls</command>)
    xml_params = {}
    for pm in re.finditer(r"<(\w+)>([\s\S]*?)</\1>", args_body):
        xml_params[pm.group(1)] = pm.group(2).strip()

    # When the model gave structured params, hand them to the canonical
    # converter (same as native calls + <invoke>) so the full tool set and
    # correct per-tool content format apply — not a partial map + k:v blob.
    if xml_params:
        from src.tool_schemas import function_call_to_tool_block
        block = function_call_to_tool_block(mapped or tool_name, json.dumps(xml_params))
        if block:
            return block

    # No structured params: args_body is a raw single value (e.g. a bash
    # command). Keep the freeform special-casing for the simple tools.
    if mapped:
        if mapped == "bash":
            content = xml_params.get("command", args_body)
        elif mapped == "python":
            content = xml_params.get("code", args_body)
        elif mapped == "web_search":
            content = xml_params.get("query", args_body)
        elif mapped == "web_fetch":
            content = xml_params.get("url", args_body)
        elif mapped in ("read_file", "write_file"):
            content = xml_params.get("path", xml_params.get("file_path", args_body))
        else:
            content = "\n".join(f"{k}: {v}" for k, v in xml_params.items()) if xml_params else args_body
        if content:
            return ToolBlock(mapped, content.strip())
    elif tool_name and args_body:
        # Unknown tool — try as MCP tool call
        content = "\n".join(f"{k}: {v}" for k, v in xml_params.items()) if xml_params else args_body
        return ToolBlock(tool_name, content.strip())
    return None


def parse_tool_blocks(text: str) -> List[ToolBlock]:
    """Extract executable tool blocks from LLM response text.

    Supports multiple formats:
    1. ```bash ... ``` fenced code blocks (standard)
    2. [TOOL_CALL] ... [/TOOL_CALL] blocks (some models)
    3. XML-style <tool_call>/<invoke> blocks
    4. <tool_code> blocks (MiniMax-M2.5 style)
    5. DeepSeek DSML markup (normalized to <invoke> first)
    6. Bare JSON ask_user objects
    7. Standalone direct ask_user(...) calls
    """
    blocks = []

    # Normalize DeepSeek DSML markup into standard <invoke> form so the
    # XML patterns below catch it.
    text = _normalize_dsml(text)

    ask_compat_fence_spans = []
    for m in _ANY_FENCE_RE.finditer(text):
        block = _parse_ask_compat_block(m.group(1).strip())
        if block:
            blocks.append(block)
            ask_compat_fence_spans.append(m.span())

    # Pattern 1: fenced code blocks
    for m in _TOOL_BLOCK_RE.finditer(text):
        if _span_overlaps(m.span(), ask_compat_fence_spans):
            continue
        tag = m.group(1).lower()
        content = m.group(2).strip()
        if not content:
            continue
        # If a code block's content is an <invoke> XML call (some models wrap
        # tool calls in ```python or ```xml fences), parse the invoke instead.
        if '<invoke' in content:
            invoked = False
            for inv in _XML_INVOKE_RE.finditer(content):
                block = _parse_xml_invoke(inv)
                if block:
                    blocks.append(block)
                    invoked = True
            if invoked:
                continue
        blocks.append(ToolBlock(tag, content))

    # Pattern 2: [TOOL_CALL] blocks (only if no fenced blocks found)
    if not blocks:
        for m in _TOOL_CALL_RE.finditer(text):
            block = _parse_tool_call_block(m.group(1))
            if block:
                blocks.append(block)

    # Pattern 3: XML-style <tool_call>/<invoke> blocks
    if not blocks:
        # Try JSON-style: <invoke_tool>{"name": "...", "arguments": {...}}</invoke_tool>
        for m in _XML_INVOKE_TOOL_RE.finditer(text):
            block = _parse_invoke_tool_json(m.group(1))
            if block:
                blocks.append(block)

        # Try wrapped: <tool_call><invoke ...>...</invoke></tool_call>
        if not blocks:
            for m in _XML_TOOL_CALL_RE.finditer(text):
                for inv in _XML_INVOKE_RE.finditer(m.group(1)):
                    block = _parse_xml_invoke(inv)
                    if block:
                        blocks.append(block)

        # Try bare <invoke> without wrapper
        if not blocks:
            for inv in _XML_INVOKE_RE.finditer(text):
                block = _parse_xml_invoke(inv)
                if block:
                    blocks.append(block)

    # Pattern 4: <tool_code> blocks (MiniMax-M2.5 style)
    if not blocks:
        for m in _TOOL_CODE_RE.finditer(text):
            block = _parse_tool_code_block(m.group(1))
            if block:
                blocks.append(block)

    # Pattern 6: Qwen-style bare JSON ask_user object as the whole response
    if not blocks:
        block = _parse_ask_compat_block(text)
        if block:
            blocks.append(block)

    # Pattern 7: Qwen-style direct ask_user(...) call as a standalone line
    if not blocks:
        for line in _iter_lines_outside_fences(text):
            block = _parse_direct_ask_call(line)
            if block:
                blocks.append(block)

    return blocks


def strip_tool_blocks(text: str) -> str:
    """Remove executable tool blocks from text for clean display."""
    # Normalize DSML first so its markup gets stripped by the <invoke>
    # / <tool_call> removers below instead of leaking to the user.
    text = _normalize_dsml(text)
    cleaned = _strip_ask_compat_fences(text)
    cleaned = _TOOL_BLOCK_RE.sub('', cleaned)
    cleaned = _TOOL_CALL_RE.sub('', cleaned)
    cleaned = _XML_INVOKE_TOOL_RE.sub('', cleaned)
    cleaned = _XML_TOOL_CALL_RE.sub('', cleaned)
    cleaned = _TOOL_CODE_RE.sub('', cleaned)
    # Strip bare <invoke> blocks not wrapped in <tool_call>
    cleaned = re.sub(r'<invoke\s+name=["\'].*?</invoke>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = _strip_standalone_direct_ask_lines(cleaned)
    if _bare_json_ask_args(cleaned) is not None:
        cleaned = ''
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _parse_invoke_tool_json(raw: str) -> Optional[ToolBlock]:
    """Parse <invoke_tool>{...}</invoke_tool> JSON blocks.

    Handles model output like:
      <invoke_tool>
      {"name": "ask_user", "arguments": {"question": "..."}}
      </invoke_tool>
    """
    try:
        payload = json.loads((raw or "").strip())
    except json.JSONDecodeError as exc:
        logger.debug("Failed to parse invoke_tool JSON: %s", exc)
        return None

    if not isinstance(payload, dict):
        return None

    tool_name = (
        payload.get("name")
        or payload.get("tool")
        or payload.get("tool_name")
    )
    if not tool_name:
        return None

    tool_name = str(tool_name).strip().lower().replace("-", "_")
    mapped = _TOOL_NAME_MAP.get(tool_name) or (tool_name if tool_name in TOOL_TAGS else tool_name)

    args = payload.get("arguments", payload.get("args", {}))
    if args is None:
        args = {}

    # Some models double-encode arguments as a JSON string.
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args)
            if isinstance(parsed_args, dict):
                args = parsed_args
        except json.JSONDecodeError:
            if mapped == "ask_user":
                args = {"question": args}
            else:
                args = {"input": args}

    if not isinstance(args, dict):
        args = {"input": str(args)}

    from src.tool_schemas import function_call_to_tool_block
    return function_call_to_tool_block(mapped, json.dumps(args))
