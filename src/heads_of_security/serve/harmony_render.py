"""Render OpenAI-format messages with gpt-oss's native Harmony renderer.

Why this exists: `server.py` rendered prompts with HF's Jinja chat template, while
vLLM renders with `openai_harmony`. With the conversation start date pinned in both,
three real differences remain -- all in tool-related turns:

  1. tool namespace: Harmony emits a JSON-schema ``title`` as its own comment line
     (``// Recipients`` then ``// The list with...``); HF emits only ``description``.
  2. assistant tool call: HF appends a ``json`` constrain marker to the header
     (``<|channel|>commentary json<|message|>``); Harmony does not.
  3. tool result: HF adds ``to=assistant`` to the author header AND passes the
     content through ``|tojson``, so it arrives JSON-quoted with escaped newlines;
     Harmony emits the raw text.

(3) is the load-bearing one: an injected payload lives in a tool result, and HF
delivers it as a quoted JSON string rather than as text. Measured consequence on the
same 559 workspace pairs: attack success 10.2% through vLLM vs 3.7% through
`server.py` on HF rendering, same weights and tasks.

That gap matters beyond tidiness. Evaluating a defense against a base model that
looks 2x more robust than it is compresses the effect we are trying to measure.

Harmony's encoding is `o200k_harmony`, the same vocabulary HF's tokenizer uses, and
the ids agree exactly on the same text -- so this renders to *text* and lets the
caller tokenize. That keeps offset-based work (per-token channel assignment for the
gated arm) possible, which rendering straight to ids would not.
"""

from __future__ import annotations

import datetime
import json
import os
from functools import lru_cache
from typing import Any

# Pinned conversation start date, used by EVERY renderer (training and serving) so
# prompts are reproducible and identical across the two. Left unpinned, both paths
# default to `today()`: training data embeds its prep date, serving embeds the run
# date, they disagree, and re-tokenising the same dataset tomorrow yields different
# prompts. vLLM flags the same non-determinism in its own source.
#
# 2024-06 is the model's knowledge cutoff, and common agent benchmarks' tasks are set in
# May 2024, so this sits inside the task universe rather than two years past it.
from heads_of_security.attention_gating.gpt_oss.gated_tokenizer import CONVERSATION_START_DATE  # noqa: E402


@lru_cache(maxsize=1)
def _encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _tool_descriptions(tools: list[dict]) -> list:
    from openai_harmony import ToolDescription

    out = []
    for t in tools or []:
        fn = t.get("function", t)
        out.append(ToolDescription.new(
            fn["name"], fn.get("description", ""), fn.get("parameters")))
    return out


def render_harmony_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    *,
    reasoning_effort: str | None = None,
    start_date: str | None = None,
    model_identity: str | None = None,
) -> str:
    """OpenAI-format messages -> a Harmony prompt string ending in the reply header.

    Mirrors how a Harmony-native server assembles a conversation:
      * ``messages[0]`` if system/developer becomes the developer *instructions*, and
        tools become the developer ``functions`` namespace. Harmony's own system
        block (identity, reasoning effort, channel list) is separate and generated,
        never taken from the caller.
      * an assistant turn with tool calls becomes one commentary message per call,
        addressed ``to=functions.<name>``.
      * a ``tool`` message is authored *by* ``functions.<name>`` -- which requires
        remembering the most recent call, since the OpenAI wire format identifies
        the tool only by ``tool_call_id``.
    """
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
    )

    sys_content = SystemContent.new()
    if model_identity is not None:
        sys_content = sys_content.with_model_identity(model_identity)
    if reasoning_effort:
        sys_content = sys_content.with_reasoning_effort(
            getattr(ReasoningEffort, reasoning_effort.upper()))
    # vLLM always stamps a conversation start date, defaulting to today -- and notes
    # in its own source that this makes its prompts non-deterministic across days.
    # Match it, but let callers pin the date so an eval is reproducible.
    if start_date is None:
        start_date = os.environ.get("VLLM_SYSTEM_START_DATE") or CONVERSATION_START_DATE
    sys_content = sys_content.with_conversation_start_date(start_date)

    msgs = list(messages)
    instructions = ""
    if msgs and msgs[0].get("role") in ("system", "developer"):
        instructions = _text(msgs[0].get("content")) or ""
        msgs = msgs[1:]

    dev = DeveloperContent.new()
    if instructions:
        dev = dev.with_instructions(instructions)
    if tools:
        dev = dev.with_function_tools(_tool_descriptions(tools))

    out = [Message.from_role_and_content(Role.SYSTEM, sys_content)]
    if instructions or tools:
        out.append(Message.from_role_and_content(Role.DEVELOPER, dev))

    last_call: str | None = None
    # tool_call_id -> function name, so a tool result can be attributed even when
    # the client interleaves calls and results out of order.
    call_names: dict[str, str] = {}

    for m in msgs:
        role = m.get("role")
        if role in ("system", "developer"):
            raise ValueError(
                f"{role!r} message at position >0; Harmony has one instructions slot. "
                "Merge it into messages[0]."
            )
        if role == "user":
            out.append(Message.from_role_and_content(Role.USER, _text(m.get("content"))))
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            text = _text(m.get("content"))
            # Chain-of-thought, when the caller supplies it, goes in its own analysis
            # message ahead of the answer. Needed for teacher-forced training targets,
            # where the trace is part of what the model must learn to produce; a
            # serving client replaying history normally has no CoT to replay, since
            # agent harnesses (and the OpenAI wire format) never store it.
            thinking = _text(m.get("thinking"))
            if thinking:
                out.append(Message.from_role_and_content(Role.ASSISTANT, thinking)
                           .with_channel("analysis"))
            if text:
                out.append(Message.from_role_and_content(Role.ASSISTANT, text)
                           .with_channel("final"))
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name")
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args if args is not None else {})
                last_call = name
                if call.get("id"):
                    call_names[call["id"]] = name
                # content_type="json" is what puts the constrain marker in the
                # header. Harmony's library renders it as a bare ` json`
                # (`<|channel|>commentary json<|message|>`) which is exactly what
                # HF's template emits; the cookbook shows `<|constrain|>json` with
                # the recipient after the channel, and that token does exist
                # (id 200003), but neither OpenAI's own library nor HF writes it
                # when replaying history. Follow the library, since that is what
                # production serving feeds the model.
                out.append(
                    Message.from_role_and_content(Role.ASSISTANT, args)
                    .with_channel("commentary")
                    .with_recipient(f"functions.{name}")
                    .with_content_type("json")
                )
        elif role == "tool":
            name = call_names.get(m.get("tool_call_id") or "", None) or last_call
            if name is None:
                raise ValueError("tool message with no preceding assistant tool call")
            # `to=assistant` is part of the documented tool-response format
            # (developers.openai.com/cookbook/articles/openai-harmony) and vLLM sets
            # it too. Omitting it produced a header HF's template got right and this
            # renderer got wrong.
            out.append(
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{name}"), _text(m.get("content"))
                ).with_channel("commentary").with_recipient("assistant")
            )
        else:
            raise ValueError(f"unsupported role {role!r}")

    conv = Conversation.from_messages(out)
    ids = _encoding().render_conversation_for_completion(conv, Role.ASSISTANT)
    return _encoding().decode(ids)


def _today() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _text(content) -> str:
    """Flatten OpenAI content (string, typed blocks, or None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text") or block.get("content") or "")
    return "".join(parts)
