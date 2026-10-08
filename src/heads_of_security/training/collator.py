"""DPO collator extended with ``token_channels`` padding."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo


def padded_collate_dpo_with_channels(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    channel_padding_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a DPO batch including ``token_channels``.

    Delegates ``input_ids`` and ``labels`` to torchtune's :func:`padded_collate_dpo`, then pads
    and stacks the channel tensors with the same chosen-then-rejected layout.

    Returns ``(input_ids, labels, token_channels)``, all of shape ``(2 * batch_size, max_len)``.
    """
    input_ids, labels = padded_collate_dpo(batch, padding_idx=padding_idx, ignore_idx=ignore_idx)

    chosen_ch = [torch.tensor(ex["chosen_token_channels"], dtype=torch.long) for ex in batch]
    rejected_ch = [torch.tensor(ex["rejected_token_channels"], dtype=torch.long) for ex in batch]
    token_channels = pad_sequence(chosen_ch + rejected_ch, batch_first=True, padding_value=channel_padding_idx)

    return input_ids, labels, token_channels
