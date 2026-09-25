"""Channel-aware Harmony tokenizer for the gated gpt-oss models.

Mirrors gpt-oss's stock HuggingFace Jinja chat template **byte for byte**, so a
gated run and a vLLM baseline run see identical prompts. We mirror HF's adapter
rather than the richer native Harmony format because HF's template exposes only one
instructions slot and builds the system turn from kwargs, not from ``messages``.

How this differs from :class:`~heads_of_security.attention_gating.llama.gated_tokenizer.LlamaGatedTokenizer`
-------------------------------------------------------------------------------
The Llama tokenizer builds ``input_ids`` piecewise, encoding scaffold and
content separately and concatenating. Harmony interleaves plain-text boilerplate
with content (``# Instructions\\n\\n`` immediately precedes the developer text),
so piecewise encoding would risk BPE merges across a boundary and drift from the
stock template. Instead we:

1. render the whole Harmony string exactly as the Jinja does, recording a
   ``(start, end, channel)`` span for every fragment;
2. tokenize once with ``return_offsets_mapping=True``;
3. label each token by the span its characters fall in.

That makes byte-parity structural rather than something to be verified per case.
A token whose characters straddle two spans takes the most-untrusted channel
present (``U`` > ``T`` > ``S``), so untrusted bytes can never be labelled trusted.

Channel policy (see the docs and ``ChannelTokenizerMixin``)
-----------------------------------------------------------
- The auto-generated system block's prose is **trusted content** (Harmony ranks
  the system message at the top of its ``system > developer > user`` hierarchy).
  Its delimiters, like all template structure, are scaffold.
- ``tool`` responses are untrusted; ``developer`` / ``user`` / ``assistant``
  (including analysis/CoT) are trusted. Mirrors the Llama defaults so results stay
  comparable across families.
- Assistant reasoning content is trusted like any other generated token; the
  ``<|channel|>analysis<|message|>`` marker is scaffold.

Not implemented (raise rather than silently diverge): the ``tools`` function
namespace and ``builtin_tools`` (browser/python) blocks, which need Harmony's
TypeScript-ish namespace renderer.
"""

from __future__ import annotations

import datetime
import functools
import json
from typing import Optional, Union

from ..channel_tokenizer import (  # re-exported for symmetry with the Llama module
    S_CHANNEL,
    T_CHANNEL,
    U_CHANNEL,
    ChannelTokenizerMixin,
)

# Strings copied verbatim from gpt-oss's stock chat_template.jinja
# (``build_system_message`` and the message loop).
DEFAULT_MODEL_IDENTITY = "You are ChatGPT, a large language model trained by OpenAI."
# The ``Current date:`` rendered when the caller passes none. Pinned, not the wall clock:
# a date that moves makes prompts non-reproducible and puts a different date in front of
# the model at train and serve time. 2024-06 is the model's knowledge cutoff.
CONVERSATION_START_DATE = "2024-06-01"
KNOWLEDGE_CUTOFF = "2024-06"
DEFAULT_REASONING_EFFORT = "medium"
VALID_CHANNELS_LINE = (
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)
TOOLS_CHANNEL_LINE = "\nCalls to these tools must go to the commentary channel: 'functions'."
INSTRUCTIONS_HEADING = "# Instructions\n\n"
TOOLS_HEADING = "# Tools\n\n"

# Roles that HF's template reads from ``messages[0]`` for the developer slot.
_INSTRUCTION_ROLES = ("system", "developer")

# Most-untrusted-wins ordering for tokens that straddle two spans.
_CHANNEL_PRIORITY = {U_CHANNEL: 2, T_CHANNEL: 1, S_CHANNEL: 0}


def _tojson(value) -> str:
    """Match this Jinja's ``|tojson``: no HTML escaping, non-ASCII left raw."""
    return json.dumps(value, ensure_ascii=False)


def _as_tool_output(value) -> str:
    """A tool result as Harmony delivers it: a string verbatim, anything else JSON.

    Agent harnesses hand us strings (YAML-ish tool output), and Harmony puts those in the
    message body unchanged. Only a non-string payload needs serializing, and then
    JSON is the reasonable choice -- but that is a fallback for structured input, not
    the normal path.
    """
    return value if isinstance(value, str) else _tojson(value)


@functools.lru_cache(maxsize=4)
def _tool_namespace_template(chat_template_src: str):
    """A Jinja template exposing gpt-oss's own ``render_tool_namespace`` macro.

    The tool namespace is ~150 lines of recursive TypeScript-type rendering (arrays,
    oneOf, enums, nullability, nested objects). Hand-porting it would be a standing
    parity risk for no benefit, so we reuse the model's own macro definitions
    verbatim: take the template source up to the last ``endmacro`` (definitions only,
    no body) and append our own call.

    Reading ``template.module.render_tool_namespace`` directly does not work — Jinja
    executes the template body to build the module, and the body needs ``messages``.
    """
    import jinja2

    lines = chat_template_src.splitlines(keepends=True)
    last_macro = max(
        (i for i, line in enumerate(lines) if "endmacro" in line), default=None
    )
    if last_macro is None:
        raise ValueError("chat template defines no macros; cannot render tool namespace")
    macros = "".join(lines[: last_macro + 1])

    env = jinja2.Environment(loader=jinja2.BaseLoader())
    # Mirror transformers' own chat-template environment.
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env.from_string(macros + '{{- render_tool_namespace("functions", tools) }}')


class _SpanBuilder:
    """Accumulates text fragments and the channel of each character span."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._spans: list[tuple[int, int, int]] = []
        self._pos = 0

    def add(self, text: str, channel: int) -> None:
        if not text:
            return
        self._parts.append(text)
        self._spans.append((self._pos, self._pos + len(text), channel))
        self._pos += len(text)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def spans(self) -> list[tuple[int, int, int]]:
        return list(self._spans)


class GptOssGatedTokenizer(ChannelTokenizerMixin):
    # Harmony uses literal role names in headers; nothing to remap.
    DEFAULT_ROLE_REMAP: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Harmony scaffold primitives
    #
    # Kept for symmetry with the other gated tokenizers and used by the
    # renderer below; the chat walk itself is overridden (see module docstring).
    # ------------------------------------------------------------------

    def _bos_ids(self) -> list[int]:
        return []  # Harmony has no BOS.

    def _header_ids(self, role: str) -> list[int]:
        return self.base.encode(self._header_text(role), add_special_tokens=False)

    def _turn_terminator_ids(self) -> list[int]:
        return self.base.encode("<|end|>", add_special_tokens=False)

    def _generation_prompt_ids(self) -> list[int]:
        return self.base.encode("<|start|>assistant", add_special_tokens=False)

    def assistant_response_prefix_ids(self) -> list[int]:
        """``<|channel|>final<|message|>`` — Harmony requires a channel on every
        message, so a bare answer used as a training target needs this preamble.

        Not needed when the recorded response already carries its own markers (e.g.
        a captured ``analysis``/``final`` trace); see the preference dataset's
        ``include_final_channel_prefix``.
        """
        return self.base.encode("<|channel|>final<|message|>", add_special_tokens=False)

    @staticmethod
    def _header_text(role: str) -> str:
        return f"<|start|>{role}<|message|>"

    # ------------------------------------------------------------------
    # System block
    # ------------------------------------------------------------------

    @staticmethod
    def _format_date(date_string: Optional[Union[str, datetime.date, datetime.datetime]]) -> str:
        """Render the ``Current date:`` value.

        ``None`` → today, matching the stock template's ``strftime_now`` (needed
        for byte-parity with a vLLM baseline). Tests pass a fixed value to stay
        hermetic. Note this differs deliberately from
        ``LlamaGatedTokenizer._make_date_header``, where ``None`` means *omit*.
        """
        if date_string is None:
            # Pinned, not today: see CONVERSATION_START_DATE. A date that moves with
            # the wall clock makes training non-reproducible and puts a different
            # `Current date:` in front of the model at train and serve time.
            return CONVERSATION_START_DATE
        if isinstance(date_string, (datetime.date, datetime.datetime)):
            return date_string.strftime("%Y-%m-%d")
        if isinstance(date_string, str):
            return date_string
        raise ValueError(
            f"date_string must be None, a str, or a date/datetime; got {date_string!r}"
        )

    @classmethod
    def build_system_block(
        cls,
        *,
        model_identity: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        date_string: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        has_tools: bool = False,
    ) -> str:
        """The prose of the ``system`` turn, verbatim per ``build_system_message()``."""
        identity = DEFAULT_MODEL_IDENTITY if model_identity is None else model_identity
        block = (
            f"{identity}\n"
            f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}\n"
            f"Current date: {cls._format_date(date_string)}\n\n"
            f"Reasoning: {reasoning_effort}\n\n"
            f"{VALID_CHANNELS_LINE}"
        )
        if has_tools:
            block += TOOLS_CHANNEL_LINE
        return block

    # ------------------------------------------------------------------
    # Harmony rendering
    # ------------------------------------------------------------------

    def _render_tool_namespace(self, tools) -> str:
        """The ``## functions`` block, rendered by Harmony's own renderer.

        Not HF's Jinja macro: it drops each parameter's JSON-schema ``title`` (which
        Pydantic auto-generates, so most harnesses' schemas have one) and puts the
        field separator *after* the trailing comment, where it lands inside the
        comment and the declaration is not actually comma-separated.

        The namespace contains no caller-supplied content -- it is entirely scaffold
        -- so taking Harmony's bytes verbatim is safe and avoids re-deriving ~150
        lines of recursive TypeScript-type rendering. Rendered tools-only and then
        stripped of the developer frame, because the instructions that share this
        turn are a *trusted* span that has to be emitted separately.
        """
        from openai_harmony import (
            Conversation,
            DeveloperContent,
            HarmonyEncodingName,
            Message,
            Role,
            ToolDescription,
            load_harmony_encoding,
        )

        def _encoding():
            return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        descs = []
        for t in tools:
            fn = t.get("function", t)
            descs.append(ToolDescription.new(
                fn["name"], fn.get("description", ""), fn.get("parameters")))
        dev = DeveloperContent.new().with_function_tools(descs)
        conv = Conversation.from_messages(
            [Message.from_role_and_content(Role.DEVELOPER, dev)])
        text = _encoding().decode(_encoding().render_conversation(conv))
        body = text.split("<|message|>", 1)[1].rsplit("<|end|>", 1)[0]
        # The caller emits TOOLS_HEADING itself (as scaffold), so drop Harmony's copy.
        return body[len(TOOLS_HEADING):] if body.startswith(TOOLS_HEADING) else body

    def render_with_spans(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        model_identity: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        date_string: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        tools=None,
        builtin_tools=None,
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Render the Harmony prompt and the channel span of every fragment."""
        if builtin_tools is not None:
            raise NotImplementedError(
                "builtin_tools (browser / python) rendering is not implemented for the "
                "gated Harmony tokenizer; it needs a separate system-turn block."
            )

        messages = self._preprocess_messages(list(messages))
        scaffold = S_CHANNEL if self.chat_template_channel else T_CHANNEL
        b = _SpanBuilder()

        # -- system turn (always emitted; content built from kwargs, not messages)
        b.add(self._header_text("system"), scaffold)
        b.add(
            self.build_system_block(
                model_identity=model_identity,
                reasoning_effort=reasoning_effort,
                date_string=date_string,
                has_tools=bool(tools),
            ),
            T_CHANNEL,
        )
        b.add("<|end|>", scaffold)

        # -- developer turn, taken from messages[0] only (HF reads no further)
        developer_message = ""
        developer_role = "developer"
        loop_messages = messages
        if messages and messages[0].get("role") in _INSTRUCTION_ROLES:
            developer_message = messages[0].get("content") or ""
            developer_role = messages[0]["role"]
            loop_messages = messages[1:]

        for extra in loop_messages:
            if extra.get("role") in _INSTRUCTION_ROLES:
                # HF's template silently drops these; refuse instead, so a
                # trusted instruction can never vanish from a security run.
                raise ValueError(
                    f"{extra['role']!r} message found at a position other than messages[0]; "
                    "Harmony has exactly one instructions slot and HF's template would "
                    "silently drop this message. Merge it into messages[0]."
                )

        # HF opens the developer turn for a developer message OR for tools, and emits
        # the two sections in that order inside one turn.
        if developer_message or tools:
            b.add(self._header_text("developer"), scaffold)
            if developer_message:
                b.add(INSTRUCTIONS_HEADING, scaffold)
                b.add(developer_message, self._channel_for_role(developer_role))
            if tools:
                # `\n\n` is a *separator* between the two developer sections, not a
                # suffix on the instructions. HF's template appends it either way,
                # leaving a trailing blank line when there are no tools; Harmony does
                # not, so emitting it unconditionally broke parity on every
                # instructions-without-tools conversation.
                if developer_message:
                    b.add("\n\n", scaffold)
                # Tool *definitions* are part of the harness, not of any untrusted
                # payload, so the whole namespace is scaffold — one span, no
                # per-fragment interleaving. (Tool *results* are a different matter:
                # role="tool" is untrusted via DEFAULT_UNTRUSTED_HEADER_ROLES.)
                b.add(TOOLS_HEADING, scaffold)
                b.add(self._render_tool_namespace(tools), scaffold)
            b.add("<|end|>", scaffold)

        # -- remaining turns
        #
        # Every Harmony branch is self-terminating, but with different terminators
        # (``<|end|>`` / ``<|call|>`` / ``<|return|>``). We hold the terminator back
        # so that an untrusted-inline role can append into the open turn before it
        # closes; otherwise it is flushed as soon as the next turn starts.
        sep = "\n\n"
        last_tool_call_name: Optional[str] = None
        pending_terminator: Optional[str] = None

        def flush_terminator() -> None:
            nonlocal pending_terminator
            if pending_terminator is not None:
                b.add(pending_terminator, scaffold)
                pending_terminator = None

        for i, msg in enumerate(loop_messages):
            # role_remap is applied at DISPATCH, not just to the header text: remapping
            # `tool` -> `input` has to turn the message into a genuine input turn,
            # skipping the tool branch's pairing requirement. The mixin's own chat walk
            # (channel_tokenizer) has always done this; this renderer overrides
            # render_with_spans wholesale and silently ignored the setting.
            role = self.role_remap.get(msg["role"], msg["role"])
            content = msg.get("content")

            if role in self.untrusted_inline_roles:
                # Not a Harmony role: inline into the open turn, exactly as if the
                # caller had concatenated it onto that message's content (which is
                # what the Llama tokenizer does, and keeps byte-parity with HF on
                # the equivalent pre-concatenated input).
                if pending_terminator is None:
                    raise ValueError(
                        f"first message has untrusted-inline role {role!r}; there's no "
                        "turn to inline into. Inline roles must follow a header-emitting role."
                    )
                if content:
                    b.add(sep, scaffold)
                    b.add(content, self._channel_for_role(role))
                continue

            flush_terminator()

            if role == "user":
                b.add(self._header_text("user"), scaffold)
                b.add(content or "", self._channel_for_role(role))
                pending_terminator = "<|end|>"

            elif role == "assistant":
                last_tool_call_name, pending_terminator = self._render_assistant(
                    b,
                    msg,
                    scaffold=scaffold,
                    is_last=(i == len(loop_messages) - 1),
                    add_generation_prompt=add_generation_prompt,
                    later_final_message=self._has_later_final_message(loop_messages, i),
                    last_tool_call_name=last_tool_call_name,
                )

            elif role in self.untrusted_header_roles and role not in ("tool", "ipython"):
                # Delimited data region (as in SecAlign): the untrusted span gets its own
                # turn with its own header, mirroring the Llama form of that defense
                # (<|start_header_id|>input<|end_header_id|>). Selected purely by
                # config -- move a role from untrusted_inline_roles to
                # untrusted_header_roles -- so the gated arms, which mark the same span
                # out-of-band with token_channels and were trained with `input` inlined,
                # keep their format untouched.
                #
                # NOT forgery-proof, deliberately: content is tokenized with special
                # tokens live, so data containing the literal text "<|end|>" produces a
                # real control token and can close this region. That is an inherent
                # weakness of in-band delimiters and belongs in a faithful baseline
                # rather than being papered over. The channel signal has no such hole.
                b.add(self._header_text(role), scaffold)
                b.add(content or "", self._channel_for_role(role))
                pending_terminator = "<|end|>"

            elif role == "tool":
                if last_tool_call_name is None:
                    raise ValueError(
                        "message has tool role, but there was no previous assistant "
                        "message with a tool call"
                    )
                b.add(
                    f"<|start|>functions.{last_tool_call_name}"
                    " to=assistant<|channel|>commentary<|message|>",
                    scaffold,
                )
                # RAW, not JSON-encoded. HF's Jinja template applies `|tojson` here,
                # delivering the tool result as a quoted string with escaped
                # newlines; Harmony -- and therefore every Harmony-native server --
                # emits it verbatim. This is not cosmetic: a prompt injection lives
                # in a tool result, and quoting it makes it read as inert data.
                # Measured on the full workspace attacked pass, same weights: attack
                # success 3.6% with `|tojson` vs 8.6% raw (vLLM reference 10.2%). The
                # JSON-quoted form was accidentally acting as a weak defense, which
                # is the last thing that belongs in a prompt-injection baseline.
                b.add(_as_tool_output(content), self._channel_for_role(role))
                pending_terminator = "<|end|>"

            else:
                raise ValueError(
                    f"unsupported role {role!r} for the Harmony template; expected one of "
                    f"system/developer/user/assistant/tool or an inline role "
                    f"({self.untrusted_inline_roles})"
                )

        flush_terminator()

        if add_generation_prompt:
            b.add("<|start|>assistant", scaffold)

        return b.text, b.spans

    @staticmethod
    def _has_later_final_message(loop_messages: list[dict], idx: int) -> bool:
        """Does a later assistant message carry a final answer (no tool_calls)?"""
        return any(
            m.get("role") == "assistant" and "tool_calls" not in m
            for m in loop_messages[idx + 1 :]
        )

    def _render_assistant(
        self,
        b: _SpanBuilder,
        msg: dict,
        *,
        scaffold: int,
        is_last: bool,
        add_generation_prompt: bool,
        later_final_message: bool,
        last_tool_call_name: Optional[str],
    ) -> tuple[Optional[str], str]:
        """Render one assistant turn.

        Returns ``(last_tool_call_name, pending_terminator)`` — the caller flushes
        the terminator once the turn can no longer receive inlined content.

        Mirrors the stock template's branches, including its checks that reject
        ``<|channel|>`` markers embedded in ``content`` / ``thinking``.
        """
        content = msg.get("content")
        thinking = msg.get("thinking")
        assistant_channel = self._channel_for_role("assistant")

        for field, value in (("content", content), ("thinking", thinking)):
            if isinstance(value, str) and (
                "<|channel|>analysis<|message|>" in value or "<|channel|>final<|message|>" in value
            ):
                raise ValueError(
                    f"assistant message {field!r} contains <|channel|> tags. Pass the "
                    "analysis text in 'thinking' and the final answer in 'content'."
                )

        if "tool_calls" in msg:
            if content and thinking:
                raise ValueError(
                    "cannot pass both content and thinking in an assistant message with "
                    "tool calls; put the analysis text in one or the other"
                )
            analysis = content if content else thinking
            if analysis and not later_final_message:
                b.add("<|start|>assistant<|channel|>analysis<|message|>", scaffold)
                b.add(analysis, assistant_channel)
                b.add("<|end|>", scaffold)

            tool_call = msg["tool_calls"][0]
            tool_call = tool_call.get("function", tool_call)
            name = tool_call["name"]
            content_type = tool_call.get("content_type", "json")
            b.add(
                f"<|start|>assistant to=functions.{name}"
                f"<|channel|>commentary {content_type}<|message|>",
                scaffold,
            )
            b.add(_tojson(tool_call.get("arguments")), assistant_channel)
            return name, "<|call|>"

        if is_last and not add_generation_prompt:
            # Training-shaped: keep the CoT and close with <|return|>.
            if thinking:
                b.add("<|start|>assistant<|channel|>analysis<|message|>", scaffold)
                b.add(thinking, assistant_channel)
                b.add("<|end|>", scaffold)
            b.add("<|start|>assistant<|channel|>final<|message|>", scaffold)
            b.add(content or "", assistant_channel)
            return last_tool_call_name, "<|return|>"

        b.add("<|start|>assistant<|channel|>final<|message|>", scaffold)
        b.add(content or "", assistant_channel)
        return None, "<|end|>"

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _channels_for_offsets(
        self, offsets: list[tuple[int, int]], spans: list[tuple[int, int, int]], scaffold: int
    ) -> list[int]:
        """Label each token by the span(s) its characters overlap.

        Straddling tokens take the most-untrusted channel present, so untrusted
        bytes are never labelled trusted.
        """
        channels: list[int] = []
        for start, end in offsets:
            if end <= start:  # zero-width (e.g. some special tokens)
                end = start + 1
            best: Optional[int] = None
            for s_start, s_end, channel in spans:
                if s_start < end and start < s_end:
                    if best is None or _CHANNEL_PRIORITY[channel] > _CHANNEL_PRIORITY[best]:
                        best = channel
            channels.append(scaffold if best is None else best)
        return channels

    def apply_chat_template_with_channels(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        return_tensors: Optional[str] = "pt",
        **render_kwargs,
    ) -> dict:
        """Tokenize a chat → ``{input_ids, attention_mask, token_channels}``.

        ``input_ids`` are byte-identical to
        ``AutoTokenizer.apply_chat_template(messages, tokenize=True, ...)`` on the
        stock gpt-oss template. Extra kwargs (``model_identity``,
        ``reasoning_effort``, ``date_string``) are forwarded to
        :meth:`render_with_spans`.
        """
        import torch

        text, spans = self.render_with_spans(
            messages, add_generation_prompt=add_generation_prompt, **render_kwargs
        )
        scaffold = S_CHANNEL if self.chat_template_channel else T_CHANNEL

        encoded = self.base(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = [tuple(o) for o in encoded["offset_mapping"]]
        channels = self._channels_for_offsets(offsets, spans, scaffold)

        assert len(ids) == len(channels)

        if return_tensors is None:
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "token_channels": channels,
            }
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "token_channels": torch.tensor([channels], dtype=torch.long),
            }
        raise ValueError(f"Unsupported return_tensors={return_tensors!r}; use 'pt' or None")
