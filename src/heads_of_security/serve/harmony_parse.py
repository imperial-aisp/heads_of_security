"""Parse gpt-oss Harmony completions into OpenAI-shaped fields.

``server.py`` used to decode generations with ``skip_special_tokens=True``, which is
correct for Llama but destroys gpt-oss: the channel markers *are* the structure, so
stripping them collapses chain-of-thought, the user-visible answer and tool calls
into one undifferentiated string. vLLM does this parsing server-side via
``--reasoning-parser openai_gptoss``; for the gates arm to be comparable against a
vLLM-served baseline, our server has to produce the same shape.

A completion (after the ``<|start|>assistant`` generation prompt) is a sequence of

    <|channel|>HEADER<|message|>BODY TERMINATOR

where HEADER is ``analysis``, ``final``, or ``commentary to=functions.NAME ...``,
and TERMINATOR is ``<|end|>``, ``<|call|>`` or ``<|return|>``. Turns may be
re-opened with ``<|start|>assistant`` in between.

Mapping, matching vLLM:
  analysis                     -> reasoning        (chain-of-thought)
  final                        -> content          (user-visible answer)
  commentary + ``to=functions``-> tool_calls
  commentary, no recipient     -> reasoning        (preamble, not an answer)
"""

from __future__ import annotations

import json
import re
import uuid

# Body runs up to a terminator, the next channel, a re-opened turn, or the end of a
# truncated generation (hit max_tokens mid-message — keep what we have).
_SEGMENT = re.compile(
    r"<\|channel\|>(?P<header>.*?)<\|message\|>(?P<body>.*?)"
    r"(?=<\|end\|>|<\|call\|>|<\|return\|>|<\|start\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)
_RECIPIENT = re.compile(r"to=(?P<name>[\w.\-]+)")


def parse_harmony_completion(text: str) -> dict:
    """Split a raw Harmony completion into ``reasoning`` / ``content`` / ``tool_calls``.

    Returns a dict with those three keys plus ``finish_reason``. Text before any
    ``<|channel|>`` marker is treated as analysis: a model that begins mid-thought
    (or a decode that lost the leading marker) should not have its output silently
    dropped.
    """
    reasoning: list[str] = []
    content: list[str] = []
    tool_calls: list[dict] = []

    matches = list(_SEGMENT.finditer(text))
    if not matches:
        # No channel structure at all. Don't guess it is an answer: an unparsed blob
        # presented as `content` would look like a clean refusal to the benchmark.
        stripped = _strip_markers(text)
        return {
            "reasoning": stripped or None,
            "content": None,
            "tool_calls": [],
            "finish_reason": "stop",
        }

    if matches[0].start() > 0:
        lead = _strip_markers(text[: matches[0].start()])
        if lead:
            reasoning.append(lead)

    for m in matches:
        header = m.group("header").strip()
        body = m.group("body")
        recipient = _RECIPIENT.search(header)

        if recipient:
            name = recipient.group("name").split(".")[-1]
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": _as_json_string(body)},
            })
        elif header.startswith("final"):
            content.append(body)
        else:
            # analysis, or commentary with no recipient
            reasoning.append(body)

    return {
        "reasoning": "".join(reasoning).strip() or None,
        "content": "".join(content).strip() or None,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }


def _as_json_string(body: str) -> str:
    """Tool arguments as a JSON *string*, which is what OpenAI clients expect.

    The model usually emits valid JSON. When it does not, pass the raw text through
    rather than inventing ``{}``: an empty object would be a silently wrong tool call,
    while the raw text surfaces as a parse error the caller can see.
    """
    body = body.strip()
    if not body:
        return "{}"
    try:
        json.loads(body)
    except (ValueError, TypeError):
        return body
    return body


def _strip_markers(text: str) -> str:
    return re.sub(r"<\|[a-z_]+\|>", "", text).strip()
