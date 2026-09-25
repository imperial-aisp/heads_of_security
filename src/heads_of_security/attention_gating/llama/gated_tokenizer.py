"""Channel-aware Llama-3 tokenizer for ``LlamaGatedForCausalLM``.

Thin subclass of :class:`~heads_of_security.attention_gating.channel_tokenizer.ChannelTokenizerMixin`
that supplies Llama-3's chat scaffolding (``<|begin_of_text|>``,
``<|start_header_id|>...<|end_header_id|>``, ``<|eot_id|>``) and the Llama-3.1
tools-block / date-header preprocessing. The generic channel policy, sidecar
persistence and chat walk live in the mixin.
"""

from __future__ import annotations

import datetime
import json
from typing import Optional, Union

from ..channel_tokenizer import ChannelTokenizerMixin

# Llama-3's hardcoded "today" string from its stock chat template — matches
# the default that ``apply_chat_template(date_string=None)`` substitutes in.
_LLAMA3_DEFAULT_DATE_STRING = "26 Jul 2024"
# Knowledge cutoff baked into Llama-3's chat template.
_LLAMA3_KNOWLEDGE_CUTOFF = "December 2023"

# Tool-block intro strings copied verbatim from Llama-3.1's stock HF chat
# template. The lack of space between ``...value}.`` and ``Do not use variables.``
# is intentional — the Jinja concatenates them without a separator.
_TOOLS_USER_MESSAGE_INTRO = (
    "Given the following functions, please respond with a JSON for a "
    "function call with its proper arguments that best answers the "
    "given prompt.\n\n"
    'Respond in the format {"name": function name, "parameters": '
    "dictionary of argument name and its value}.Do not use variables.\n\n"
)
_TOOLS_SYSTEM_MESSAGE_INTRO = (
    "You have access to the following functions. To call a function, "
    "please respond with JSON for a function call."
    'Respond in the format {"name": function name, "parameters": '
    "dictionary of argument name and its value}.Do not use variables.\n\n"
)

# Llama-3.1's Jinja chat template silently maps "tool" -> "ipython" in the
# rendered header. We replicate that here so the gated tokenizer's output
# matches what the base model saw during pretraining. Channel lookup still uses
# the *original* (pre-remap) role so that ``untrusted_header_roles`` can list
# "tool" and it will be found.
DEFAULT_ROLE_REMAP: dict[str, str] = {"tool": "ipython"}


class LlamaGatedTokenizer(ChannelTokenizerMixin):
    DEFAULT_ROLE_REMAP = DEFAULT_ROLE_REMAP

    # ------------------------------------------------------------------
    # Llama-3 scaffold primitives
    # ------------------------------------------------------------------

    def _special_ids(self) -> dict[str, int]:
        if not hasattr(self, "_special_ids_cache"):
            b = self.base
            object.__setattr__(
                self,
                "_special_ids_cache",
                {
                    "bos": b.convert_tokens_to_ids("<|begin_of_text|>"),
                    "eot": b.convert_tokens_to_ids("<|eot_id|>"),
                    "sh": b.convert_tokens_to_ids("<|start_header_id|>"),
                    "eh": b.convert_tokens_to_ids("<|end_header_id|>"),
                },
            )
        return self._special_ids_cache

    def _bos_ids(self) -> list[int]:
        return [self._special_ids()["bos"]]

    def _header_ids(self, role: str) -> list[int]:
        sp = self._special_ids()
        nn_ids = self.base.encode("\n\n", add_special_tokens=False)
        return [sp["sh"]] + self.base.encode(role, add_special_tokens=False) + [sp["eh"]] + nn_ids

    def _turn_terminator_ids(self) -> list[int]:
        return [self._special_ids()["eot"]]

    # ------------------------------------------------------------------
    # Llama-3.1 tools / date preprocessing
    # ------------------------------------------------------------------

    def _preprocess_messages(
        self,
        messages: list[dict],
        *,
        date_string: Optional[Union[str, datetime.date, datetime.datetime]] = None,
        tools: Optional[list[dict]] = None,
        tools_in_user_message: bool = True,
    ) -> list[dict]:
        # Match the stock Llama template's system-message `| trim`
        # (`{%- set system_message = messages[0]['content']|trim %}`), which runs
        # BEFORE the tool / date / environment blocks are prepended. The
        # per-message `| trim` for the remaining turns is applied in the
        # rendering pass at the end.
        if (messages and messages[0].get("role") == "system"
                and isinstance(messages[0].get("content"), str)):
            messages[0] = {**messages[0], "content": messages[0]["content"].strip()}

        if tools is not None:
            tool_block = self._format_tools_block(tools, in_user_message=tools_in_user_message)
            if tools_in_user_message:
                for i, m in enumerate(messages):
                    if m.get("role") == "user":
                        messages[i] = {**m, "content": tool_block + m["content"]}
                        break
                else:
                    raise ValueError(
                        "tools_in_user_message=True requires at least one 'user' "
                        "message in `messages` to host the tool block."
                    )
            else:
                if messages and messages[0].get("role") == "system":
                    messages[0] = {**messages[0], "content": tool_block + messages[0]["content"]}
                else:
                    messages.insert(0, {"role": "system", "content": tool_block})

        date_prefix = self._make_date_header(date_string)
        if date_prefix:
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": date_prefix + messages[0]["content"]}
            else:
                messages.insert(0, {"role": "system", "content": date_prefix})

        if tools is not None:
            env_prefix = "Environment: ipython\n"
            if messages and messages[0].get("role") == "system":
                messages[0] = {**messages[0], "content": env_prefix + messages[0]["content"]}
            else:
                messages.insert(0, {"role": "system", "content": env_prefix})

        # Render each message's content exactly as the stock Llama-3 chat
        # template does, so tool-calling evals can pass structured messages
        # instead of pre-encoding JSON themselves:
        #   * an assistant `tool_calls` message -> {"name": .., "parameters": <args>}
        #     (the stock template supports a single call per message);
        #   * a `tool`/`ipython` message -> `content | tojson` (JSON-encoded);
        #   * every other role -> `content | trim` (system already trimmed above).
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if "tool_calls" in msg:
                calls = msg["tool_calls"]
                if len(calls) != 1:
                    raise ValueError(
                        f"gated tokenizer supports exactly one tool_call per message (got {len(calls)})"
                    )
                fn = calls[0].get("function", calls[0])
                messages[i] = {
                    "role": role,
                    "content": json.dumps(
                        {"name": fn["name"], "parameters": fn["arguments"]}, ensure_ascii=False
                    ),
                }
            elif role in ("tool", "ipython"):
                content = msg["content"]
                # Stock Llama tool/ipython branch:
                #   {% if content is mapping or content is iterable %}
                #       {{ content | tojson }}{% else %}{{ content }}{% endif %}
                # A str is iterable, so JSON strings are (re-)encoded; only a
                # non-iterable scalar (int/float/bool/None) is rendered raw, as
                # Jinja's `{{ content }}` would stringify it.
                if hasattr(content, "__iter__"):
                    content = json.dumps(content, ensure_ascii=False)
                else:
                    content = str(content)
                messages[i] = {**msg, "content": content}
            elif role != "system" and msg.get("content"):
                messages[i] = {**msg, "content": msg["content"].strip()}

        return messages

    @staticmethod
    def _format_tools_block(tools: list[dict], *, in_user_message: bool = True) -> str:
        """Render a tools block to prepend to a message's content.

        Strings copied verbatim from Llama-3.1's stock HF chat template. Returned
        text ends with ``\\n\\n`` so it concatenates cleanly in front of the host
        message's content.
        """
        intro = _TOOLS_USER_MESSAGE_INTRO if in_user_message else _TOOLS_SYSTEM_MESSAGE_INTRO
        tool_dumps = "".join(json.dumps(t, indent=4) + "\n\n" for t in tools)
        return intro + tool_dumps

    @staticmethod
    def _make_date_header(
        date_string: Optional[Union[str, datetime.date, datetime.datetime]],
    ) -> str:
        """Build the ``Cutting Knowledge Date / Today Date`` block Llama-3's stock
        template injects into the system turn.

        ``None`` → ``""`` (disabled); ``"default"`` → Llama-3's ``"26 Jul 2024"``;
        ``"today"`` → today; a ``datetime`` → ``"DD Mon YYYY"``. Ends with ``\\n\\n``.
        """
        if date_string is None:
            return ""
        if isinstance(date_string, (datetime.date, datetime.datetime)):
            date_str = date_string.strftime("%d %b %Y")
        elif date_string == "default":
            date_str = _LLAMA3_DEFAULT_DATE_STRING
        elif date_string == "today":
            date_str = datetime.date.today().strftime("%d %b %Y")
        else:
            raise ValueError(
                f"date_string must be None, 'default', 'today', or a datetime; got {date_string!r}"
            )
        return f"Cutting Knowledge Date: {_LLAMA3_KNOWLEDGE_CUTOFF}\nToday Date: {date_str}\n\n"
