"""Parse Llama-3.x tool calls out of a plain-text completion.

Llama-3.1+ answers a tool-bearing prompt with one JSON object per call, as its
chat template asks::

    {"name": "get_weather", "parameters": {"city": "Paris"}}

optionally prefixed with ``<|python_tag|>`` and, for several calls, separated by
``;``. vLLM's ``Llama3JsonToolParser`` is the same 20 lines wrapped in the OpenAI
frontend's request/response types, which ``heads_of_security.vllm_gates.server`` bypasses;
this is the dependency-free copy. The output shape matches
:func:`heads_of_security.serve.harmony_parse.parse_harmony_completion` so the two families
return identical wire messages.

Rules (vLLM's): every ``{`` outside an already-parsed object starts a candidate;
a candidate is a call iff it json-decodes to a mapping with a string ``name`` and
either ``parameters`` or ``arguments``. Anything else is left in ``content``.
Arguments go on the wire as a JSON *string*, as OpenAI does.
"""

from __future__ import annotations

import json
import uuid

PYTHON_TAG = "<|python_tag|>"
_DECODER = json.JSONDecoder()


def _arguments_as_json_string(args) -> str:
    if isinstance(args, str):
        # Already a JSON object encoded as a string (the form the chat template
        # renders for prior calls): pass through rather than double-encode.
        try:
            if isinstance(json.loads(args), dict):
                return args
        except ValueError:
            pass
    return json.dumps(args, ensure_ascii=False)


def parse_llama_tool_calls(text: str) -> dict:
    """Split a Llama completion into ``content`` and OpenAI-shaped ``tool_calls``.

    Returns ``{"content", "tool_calls", "finish_reason"}``: ``tool_calls`` is a
    list (empty when there is none), ``content`` is the text left over after the
    call objects are removed (``None`` when nothing is left), ``finish_reason`` is
    ``"tool_calls"`` or ``"stop"``.
    """
    tool_calls: list[dict] = []
    keep: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        start = text.find("{", pos)
        if start < 0:
            keep.append(text[pos:])
            break
        try:
            # raw_decode returns the ABSOLUTE end index, not a length.
            obj, end = _DECODER.raw_decode(text, start)
        except ValueError:
            # Not JSON here: keep the brace as text and look past it.
            keep.append(text[pos:start + 1])
            pos = start + 1
            continue
        if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and ("parameters" in obj or "arguments" in obj)):
            args = obj["arguments"] if "arguments" in obj else obj["parameters"]
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": obj["name"],
                             "arguments": _arguments_as_json_string(args)},
            })
            keep.append(text[pos:start])
        else:
            # Valid JSON but not a call (a literal, a list, an object without a
            # name): it is content.
            keep.append(text[pos:end])
        pos = end
    if tool_calls:
        # Drop the `;` / whitespace the template uses between calls.
        content = "".join(keep).replace(PYTHON_TAG, "").replace(";", " ").strip() or None
    else:
        # No call: the completion is the content, unchanged (as vLLM returns it).
        content = text if text else None
    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }
