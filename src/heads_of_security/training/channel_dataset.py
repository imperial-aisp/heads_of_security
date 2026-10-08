"""Channel-labelled preference dataset for gate DPO training.

Reads a preference file in the messages format (a JSON list, optionally gzipped, loaded
through ``datasets``):

    {"messages": [{"role": "user", "content": ...}, {"role": "input", "content": ...}],
     "chosen": "...", "rejected": "..."}

``input`` is the untrusted data region carrying the injection. It is an untrusted *inline*
role by default, so its content folds into the preceding ``user`` turn (channel U = 1) and the
prompt is one user turn, as in SecAlign's training layout. The prompt is rendered with the
gated tokenizer's ``apply_chat_template_with_channels``, which emits per-token channel labels
(0 trusted, 1 untrusted, 2 template); the two responses get channel 0.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import PreferenceDataset

from heads_of_security.attention_gating.channel_tokenizer import T_CHANNEL

T = T_CHANNEL


class ChannelPreferenceDataset(PreferenceDataset):
    """Channel-labelled DPO dataset.

    Expects samples with ``messages`` (``user`` / ``input`` roles), ``chosen`` and ``rejected``
    (response text, no trailing end-of-turn token).
    """

    def __init__(
        self,
        *,
        source: str,
        tokenizer,
        filter_fn: Optional[Callable] = None,
        max_seq_len: Optional[int] = None,
        include_final_channel_prefix: bool = True,
        **load_dataset_kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._message_transform = None
        self._max_seq_len = max_seq_len
        self._include_final_channel_prefix = include_final_channel_prefix
        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def _tokenize_messages_with_channels(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        out = self._tokenizer.apply_chat_template_with_channels(
            messages, add_generation_prompt=True, return_tensors=None
        )
        ids, channels = out["input_ids"], out["token_channels"]

        # Harmony needs a <|channel|>final<|message|> preamble before a bare answer. Set
        # include_final_channel_prefix=False when the recorded responses already carry their
        # own channel markers (analysis + final traces, as in the released gpt-oss data).
        if self._include_final_channel_prefix:
            prefix = self._tokenizer.assistant_response_prefix_ids()
            if prefix:
                ids = ids + prefix
                channels = channels + [channels[-1]] * len(prefix)
        return ids, channels

    def _encode_response(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        ids.append(self._tokenizer.eot_id)
        return ids

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict:
        prompt_ids, prompt_channels = self._tokenize_messages_with_channels(sample["messages"])
        chosen_ids = self._encode_response(sample["chosen"])
        rejected_ids = self._encode_response(sample["rejected"])

        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        chosen_channels = prompt_channels + [T] * len(chosen_ids)
        rejected_channels = prompt_channels + [T] * len(rejected_ids)

        # Labels: the prompt and the final end-of-turn token are masked out.
        prompt_len = len(prompt_ids)
        chosen_mask = [True] * prompt_len + [False] * (len(chosen_ids) - 1) + [True]
        rejected_mask = [True] * prompt_len + [False] * (len(rejected_ids) - 1) + [True]

        if self._max_seq_len is not None:
            chosen_input_ids = chosen_input_ids[: self._max_seq_len]
            chosen_channels = chosen_channels[: self._max_seq_len]
            chosen_mask = chosen_mask[: self._max_seq_len]
            rejected_input_ids = rejected_input_ids[: self._max_seq_len]
            rejected_channels = rejected_channels[: self._max_seq_len]
            rejected_mask = rejected_mask[: self._max_seq_len]

        chosen_labels = list(np.where(chosen_mask, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids))
        rejected_labels = list(np.where(rejected_mask, CROSS_ENTROPY_IGNORE_IDX, rejected_input_ids))

        return dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            chosen_token_channels=chosen_channels,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
            rejected_token_channels=rejected_channels,
        )


def channel_preference_dataset(
    tokenizer,
    *,
    source: str,
    filter_fn: Optional[Callable] = None,
    max_seq_len: Optional[int] = None,
    include_final_channel_prefix: bool = True,
    split: str = "train",
    **load_dataset_kwargs: Any,
) -> ChannelPreferenceDataset:
    """Config-instantiate builder.

    ``source`` and ``load_dataset_kwargs`` go to ``datasets.load_dataset``; ``source: json`` with
    ``data_files`` pointing at a local path or an ``hf://datasets/<org>/<repo>/<file>`` URL.
    ``include_final_channel_prefix`` only affects tokenizers that define
    :meth:`assistant_response_prefix_ids` (Harmony / gpt-oss).
    """
    return ChannelPreferenceDataset(
        source=source,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        max_seq_len=max_seq_len,
        include_final_channel_prefix=include_final_channel_prefix,
        split=split,
        **load_dataset_kwargs,
    )
