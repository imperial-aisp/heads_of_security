"""Channel-gated Llama attention.

The gate operator (``_apply_qk_gate``) and the channel bookkeeping (``_qk_gate_mask``)
live in :mod:`.gating_core` and are shared with the gpt-oss gated attention; this module
keeps only the Llama-specific ``forward``.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .gating_core import GatedAttentionMixin


class LlamaAttentionGated(LlamaAttention, GatedAttentionMixin):
    """Llama attention with learned per-head channel gates.

    ``qk_gating``: add ``logsigmoid(head_gates[q_channel, k_channel, head])`` to
    ``attention_mask`` (a pre-softmax log-bias) before dispatching to the standard
    ``eager_attention_forward``. ``head_gates`` has shape ``[C, C, num_attention_heads]``.

    Gating is a no-op when ``token_channels is None`` (bit-exact identity to the base
    ``LlamaAttention``).

    Config fields expected on the passed ``LlamaConfig``:
      - ``gating_num_channels`` (int, required)
      - ``gating_variant`` (must be ``"qk_gating"``)
      - ``gating_init_logit`` (float, optional, default 4.0)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        if config._attn_implementation != "eager":
            raise ValueError(
                f"LlamaAttentionGated requires _attn_implementation='eager'; got {config._attn_implementation!r}"
            )
        self._init_qk_gates(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        token_channels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            raise ValueError(
                "LlamaAttentionGated expects a non-None attention_mask "
                "(the gate is added to it; build one upstream)"
            )

        if token_channels is not None:
            token_channels = token_channels.to(hidden_states.device)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        new_len = value_states.shape[-2]

        if past_key_values is not None:
            cache_kwargs = {"cache_position": kwargs.get("cache_position")}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if token_channels is not None:
            k_total = key_states.shape[-2]
            if token_channels.shape[-1] > k_total:
                raise ValueError(
                    f"token_channels length {token_channels.shape[-1]} exceeds key length {k_total}"
                )
            attention_mask = self._qk_gate_mask(attention_mask, token_channels, new_len, k_total)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
