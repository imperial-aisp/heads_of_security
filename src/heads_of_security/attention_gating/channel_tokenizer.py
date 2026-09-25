"""Model-agnostic channel-aware tokenizer mixin.

Wraps any HF tokenizer (via composition, not inheritance — see note below) and
adds ``apply_chat_template_with_channels``, which walks a chat and emits per-token
channel labels (trusted / untrusted / scaffold) alongside ``input_ids``.

The *channel policy* (which roles are untrusted, the sidecar persistence, the
message walk) is architecture-independent and lives here. The *scaffold token
sequences* (BOS, role headers, turn terminators, generation prompt) are
architecture-specific and supplied by subclasses via the ``_*_ids`` primitives —
e.g. Llama-3 headers vs Harmony.

Why composition over inheritance: subclassing ``TokenizersBackend`` (or a concrete
HF tokenizer) breaks transformers' loader — a subclass with a custom ``__init__``
triggers a slow→fast rebuild path that drops the byte-level BPE merges, producing
character-by-character tokenization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

T_CHANNEL = 0
U_CHANNEL = 1
S_CHANNEL = 2

DEFAULT_UNTRUSTED_INLINE_ROLES = ["input"]
DEFAULT_UNTRUSTED_HEADER_ROLES = ["ipython", "tool"]

SIDECAR_FILE = "gated_tokenizer_config.json"

# Attribute names that live on the wrapper itself (not the underlying base
# tokenizer). Writes to these go to ``self``; everything else delegates to
# ``self.base``.
_WRAPPER_ATTRS = frozenset(
    {
        "base",
        "untrusted_inline_roles",
        "untrusted_header_roles",
        "role_remap",
        "chat_template_channel",
    }
)


class ChannelTokenizerMixin:
    """Channel policy + chat walk shared by the gated tokenizers.

    Subclasses set ``DEFAULT_ROLE_REMAP`` and implement the scaffold primitives
    (``_bos_ids``, ``_header_ids``, ``_turn_terminator_ids``,
    ``_generation_prompt_ids``); optionally override ``_preprocess_messages`` for
    arch-specific tools/date handling.
    """

    DEFAULT_ROLE_REMAP: dict[str, str] = {}

    def __init__(
        self,
        base: PreTrainedTokenizerBase,
        *,
        untrusted_inline_roles: Optional[list[str]] = None,
        untrusted_header_roles: Optional[list[str]] = None,
        chat_template_channel: bool = False,
        role_remap: Optional[dict[str, str]] = None,
    ):
        self.base = base
        self.untrusted_inline_roles = (
            list(untrusted_inline_roles) if untrusted_inline_roles is not None else DEFAULT_UNTRUSTED_INLINE_ROLES
        )
        self.untrusted_header_roles = (
            list(untrusted_header_roles) if untrusted_header_roles is not None else DEFAULT_UNTRUSTED_HEADER_ROLES
        )
        self.chat_template_channel = chat_template_channel
        self.role_remap = dict(role_remap) if role_remap is not None else dict(self.DEFAULT_ROLE_REMAP)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Only reached for attributes not found on the instance/class.
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name in _WRAPPER_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base={type(self.base).__name__}, "
            f"untrusted_inline_roles={self.untrusted_inline_roles}, "
            f"untrusted_header_roles={self.untrusted_header_roles}, "
            f"chat_template_channel={self.chat_template_channel})"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        untrusted_inline_roles: Optional[list[str]] = None,
        untrusted_header_roles: Optional[list[str]] = None,
        chat_template_channel: Optional[bool] = None,
        role_remap: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        """Load via ``AutoTokenizer.from_pretrained`` and wrap.

        Explicit policy kwargs override anything in the sidecar; ``None`` falls
        back to the sidecar (if present in a local dir) and then to the defaults.
        """
        base = AutoTokenizer.from_pretrained(model_id, **kwargs)

        sidecar = {}
        if os.path.isdir(model_id):
            sidecar_path = Path(model_id) / SIDECAR_FILE
            if sidecar_path.is_file():
                with open(sidecar_path) as f:
                    sidecar = json.load(f)

        return cls(
            base,
            untrusted_inline_roles=(
                untrusted_inline_roles if untrusted_inline_roles is not None else sidecar.get("untrusted_inline_roles")
            ),
            untrusted_header_roles=(
                untrusted_header_roles if untrusted_header_roles is not None else sidecar.get("untrusted_header_roles")
            ),
            chat_template_channel=(
                chat_template_channel if chat_template_channel is not None else sidecar.get("chat_template_channel", False)  # noqa: E501
            ),
            role_remap=(role_remap if role_remap is not None else sidecar.get("role_remap")),
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the base tokenizer + a sidecar JSON with policy fields."""
        self.base.save_pretrained(save_directory, **kwargs)
        sidecar = {
            "untrusted_inline_roles": self.untrusted_inline_roles,
            "untrusted_header_roles": self.untrusted_header_roles,
            "chat_template_channel": self.chat_template_channel,
            "role_remap": self.role_remap,
        }
        with open(Path(save_directory) / SIDECAR_FILE, "w") as f:
            json.dump(sidecar, f, indent=2)

    # ------------------------------------------------------------------
    # Channel policy
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        return 3 if self.chat_template_channel else 2

    def _channel_for_role(self, role: str) -> int:
        if role in self.untrusted_inline_roles or role in self.untrusted_header_roles:
            return U_CHANNEL
        return T_CHANNEL

    # ------------------------------------------------------------------
    # Scaffold primitives — overridden per architecture
    # ------------------------------------------------------------------

    def _bos_ids(self) -> list[int]:
        """Leading token ids (e.g. ``<|begin_of_text|>``); empty for ChatML."""
        return []

    def _header_ids(self, role: str) -> list[int]:
        """Token ids that open a turn for ``role`` (already role-remapped)."""
        raise NotImplementedError

    def _turn_terminator_ids(self) -> list[int]:
        """Token ids that close a turn (e.g. ``<|eot_id|>`` / ``<|im_end|>\\n``)."""
        raise NotImplementedError

    def _inline_sep_ids(self) -> list[int]:
        """Separator inserted before inlined untrusted content."""
        return self.base.encode("\n\n", add_special_tokens=False)

    def _generation_prompt_ids(self) -> list[int]:
        """Trailing assistant header so the next decoded token starts the reply."""
        return self._header_ids("assistant")

    def assistant_response_prefix_ids(self) -> list[int]:
        """Scaffold that must precede a *response* body, after the generation prompt.

        Empty for Llama-3 and ChatML, where the assistant header is the whole
        preamble. Harmony overrides it: every message needs a ``<|channel|>``
        marker, so a training target that is a bare answer has to be prefixed with
        ``<|channel|>final<|message|>``. Used by the preference dataset when the
        recorded response does not already carry its own markers.
        """
        return []

    def _preprocess_messages(self, messages: list[dict], **kwargs) -> list[dict]:
        """Hook for arch-specific message rewriting (tools/date). Default no-op."""
        return messages

    # ------------------------------------------------------------------
    # Chat templating
    # ------------------------------------------------------------------

    def apply_chat_template_with_channels(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        return_tensors: Optional[str] = "pt",
        **preprocess_kwargs,
    ) -> dict:
        """Tokenize a chat → ``{input_ids, attention_mask, token_channels}``.

        Walks ``messages`` left-to-right. Roles in ``untrusted_inline_roles`` append
        their content to the currently-open turn (separated by ``_inline_sep_ids``).
        Every other role opens a new turn with its own header. Scaffold tokens
        (BOS, headers, terminators, separators) land on the scaffold channel
        (``S_CHANNEL`` if ``chat_template_channel`` else ``T_CHANNEL``); message
        content lands on the channel for its role.

        ``return_tensors="pt"`` (default) → ``LongTensor[1, L]`` per key; ``None`` →
        plain Python lists. Extra kwargs are forwarded to ``_preprocess_messages``.
        """
        messages = self._preprocess_messages(list(messages), **preprocess_kwargs)

        base = self.base
        scaffold = S_CHANNEL if self.chat_template_channel else T_CHANNEL

        _header_cache: dict[str, list[int]] = {}

        def header(role: str) -> list[int]:
            if role not in _header_cache:
                _header_cache[role] = self._header_ids(role)
            return _header_cache[role]

        def emit_scaffold(token_ids: list[int]) -> None:
            ids.extend(token_ids)
            channels.extend([scaffold] * len(token_ids))

        def emit_content(content: str, channel: int) -> None:
            cids = base.encode(content, add_special_tokens=False)
            ids.extend(cids)
            channels.extend([channel] * len(cids))

        ids: list[int] = []
        channels: list[int] = []
        emit_scaffold(self._bos_ids())

        sep_ids = self._inline_sep_ids()
        terminator_ids = self._turn_terminator_ids()

        turn_open = False
        last_turn_role: Optional[str] = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            rendered_role = self.role_remap.get(role, role)
            if role in self.untrusted_inline_roles:
                if not turn_open:
                    raise ValueError(
                        f"first message has untrusted-inline role {role!r}; there's "
                        f"no turn to inline into. Inline roles must follow a "
                        f"header-emitting role."
                    )
                if content:
                    emit_scaffold(sep_ids)
                    emit_content(content, self._channel_for_role(role))
            else:
                if turn_open:
                    emit_scaffold(terminator_ids)
                emit_scaffold(header(rendered_role))
                if content:
                    emit_content(content, self._channel_for_role(role))
                turn_open = True
                last_turn_role = rendered_role

        if turn_open:
            emit_scaffold(terminator_ids)

        if add_generation_prompt and last_turn_role != "assistant":
            emit_scaffold(self._generation_prompt_ids())

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
