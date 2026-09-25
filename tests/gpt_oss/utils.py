"""Shared helpers for the gated gpt-oss tests.

Mirrors ``tests/utils.py`` where it can and adds the two gpt-oss-specific bits:

- **RoPE shape.** gpt-oss uses the chunked-half rotary form
  (``torch.chunk(x, 2, dim=-1)``), so ``cos``/``sin`` are ``head_dim // 2`` wide,
  not ``head_dim`` as in Llama. ``cos=1, sin=0`` is still a pass-through.
- **Sinks need initializing.** ``GptOssAttention.sinks`` is created with
  ``torch.empty``, so a module built directly (rather than through
  ``from_pretrained`` / ``post_init``) holds uninitialized memory. Every helper
  here seeds it deterministically.
"""

from __future__ import annotations

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from heads_of_security.attention_gating import GptOssAttentionGated
from tests.utils import tiny_gpt_oss_config

# Layer indices in the tiny 4-layer config: [sliding, full, sliding, full].
FULL_LAYER_IDX = 1
SLIDING_LAYER_IDX = 0

__all__ = [
    "FULL_LAYER_IDX",
    "SLIDING_LAYER_IDX",
    "attention_pair",
    "causal_mask",
    "position_embeddings",
    "sliding_causal_mask",
    "tiny_gpt_oss_config",
]


def position_embeddings(
    batch_size: int, seq_len: int, head_dim: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) making RoPE a pass-through, in gpt-oss's half-width convention."""
    half = head_dim // 2
    cos = torch.ones(batch_size, seq_len, half, dtype=dtype)
    sin = torch.zeros(batch_size, seq_len, half, dtype=dtype)
    return cos, sin


def causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive causal mask ``[B, 1, Q, Q]`` with -inf above the diagonal."""
    neg_inf = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    causal = torch.triu(neg_inf, diagonal=1)
    return causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()


def sliding_causal_mask(
    batch_size: int, seq_len: int, window: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Additive causal mask restricted to a ``window``-key lookback.

    Matches HF's sliding semantics: query ``q`` may attend to keys ``k`` with
    ``q - window < k <= q``.
    """
    mask = causal_mask(batch_size, seq_len, dtype)
    idx = torch.arange(seq_len)
    too_old = (idx.view(seq_len, 1) - idx.view(1, seq_len)) >= window
    mask[:, :, too_old] = float("-inf")
    return mask


def _init_sinks(module: GptOssAttention, seed: int = 11) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.sinks.copy_(torch.randn(module.sinks.shape, generator=generator))


def attention_pair(
    *,
    init_logit: float = 4.0,
    layer_idx: int = FULL_LAYER_IDX,
    sliding_window: int = 3,
    gating_num_channels: int = 2,
):
    """Build a weight-identical ``(base, gated, config)`` attention triple."""
    torch.manual_seed(0)
    config = tiny_gpt_oss_config(
        init_logit=init_logit,
        gating_num_channels=gating_num_channels,
        sliding_window=sliding_window,
    )
    base = GptOssAttention(config, layer_idx=layer_idx).eval()
    gated = GptOssAttentionGated(config, layer_idx=layer_idx).eval()

    _init_sinks(base)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(gated, name).load_state_dict(getattr(base, name).state_dict())
    with torch.no_grad():
        gated.sinks.copy_(base.sinks)

    return base, gated, config


def equivalence_inputs(config, *, batch_size: int = 2, seq_len: int = 5, seed: int = 1):
    """(hidden_states, position_embeddings, causal mask, token_channels)."""
    torch.manual_seed(seed)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    channels = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 0, 1, 0]], dtype=torch.long)
    return hidden_states, pos, mask, channels[:batch_size, :seq_len]
