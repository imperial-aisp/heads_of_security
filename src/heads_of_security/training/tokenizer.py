"""The channel tokenizers plus the two aliases the training recipe needs.

The recipe expects torchtune's ``ModelTokenizer`` surface: ``.pad_id`` for the collator and
``.eot_id`` for response encoding. The gated tokenizers are framework-agnostic (HF surface),
so this module bolts the two aliases on and dispatches by ``config.model_type``.
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoConfig

from heads_of_security.attention_gating.gpt_oss import GptOssGatedTokenizer
from heads_of_security.attention_gating.llama import LlamaGatedTokenizer


class LlamaTrainingTokenizer(LlamaGatedTokenizer):
    """``LlamaGatedTokenizer`` + ``pad_id`` / ``eot_id``. Pad is ``<|finetune_right_pad_id|>``
    (as in torchtune's own Llama-3 tokenizer); end of turn is ``<|eot_id|>``."""

    @property
    def pad_id(self) -> int:
        return self.base.pad_token_id

    @property
    def eot_id(self) -> int:
        return self.base.convert_tokens_to_ids("<|eot_id|>")


class GptOssTrainingTokenizer(GptOssGatedTokenizer):
    """``GptOssGatedTokenizer`` + ``pad_id`` / ``eot_id``.

    Harmony's end-of-turn for a *training* target is ``<|return|>``: the stock template closes
    the final assistant turn with it (and the model is meant to learn to emit it), whereas
    ``<|end|>`` closes earlier turns. ``pad_token`` is already ``<|endoftext|>``.
    """

    @property
    def pad_id(self) -> int:
        return self.base.pad_token_id

    @property
    def eot_id(self) -> int:
        return self.base.convert_tokens_to_ids("<|return|>")


def training_tokenizer(
    model_id: str,
    *,
    untrusted_inline_roles: Optional[list[str]] = None,
    untrusted_header_roles: Optional[list[str]] = None,
    chat_template_channel: Optional[bool] = None,
    **kwargs,
):
    """Config-instantiate factory: the channel tokenizer for ``model_id``'s family.

    YAML usage::

        tokenizer:
          _component_: heads_of_security.training.tokenizer.training_tokenizer
          model_id: openai/gpt-oss-20b
          untrusted_inline_roles: ["input"]
          chat_template_channel: true
    """
    model_type = AutoConfig.from_pretrained(model_id).model_type
    policy = dict(
        untrusted_inline_roles=untrusted_inline_roles,
        untrusted_header_roles=untrusted_header_roles,
        chat_template_channel=chat_template_channel,
    )
    if model_type == "llama":
        tok = LlamaTrainingTokenizer.from_pretrained(model_id, **policy, **kwargs)
        tok.pad_token = "<|finetune_right_pad_id|>"
        return tok
    if model_type == "gpt_oss":
        return GptOssTrainingTokenizer.from_pretrained(model_id, **policy, **kwargs)
    raise ValueError(f"training_tokenizer: unsupported model_type {model_type!r} (llama, gpt_oss)")
