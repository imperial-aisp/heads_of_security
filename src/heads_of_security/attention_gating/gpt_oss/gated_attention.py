"""Channel-gated gpt-oss attention (``qk_gating``).

The gate math and channel bookkeeping are shared with the Llama gated attention and
live in :mod:`heads_of_security.attention_gating.gating_core`; this module keeps only
the gpt-oss-specific ``forward``.

Two gpt-oss features matter here:

**Attention sinks.** ``eager_attention_forward`` adds the mask, then concatenates
a learned per-head sink logit onto the score matrix, softmaxes over ``K + 1``,
and finally *drops* the sink column (``scores = probs[..., :-1]``) -- so attention
rows do not sum to 1, the missing mass being ``p(sink)``. The content log-bias is
applied to the content logits; with ``gating_sink`` a second table ``sink_gates``
``[C, H]`` adds a per-(query channel, head) log-bias to the sink logit as well, so a
head can suppress untrusted content *and* the sink together and keep its trusted
attention intact (see ``GatedAttentionMixin._init_qk_gates``). Without ``gating_sink``
the sink is left ungated and closing a gate makes the head read less overall.

**Per-layer masks.** Half of gpt-oss's layers are sliding-window;
``GptOssModel.forward`` builds one mask per layer type and hands each layer the
right one, so the gate site here is uniform across layer types. Out-of-window
keys arrive as ``finfo.min`` and stay masked once a (negative) log-gate is added.
In a sliding layer the gate can therefore only act on in-window pairs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssConfig,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from ..gating_core import _QK_GATE_HEAD_CHUNK, GatedAttentionMixin


def gated_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, H, Q, D]
    key: torch.Tensor,  # [B, H_kv, K, D]
    value: torch.Tensor,  # [B, H_kv, K, D]
    attention_mask: Optional[torch.Tensor],  # [B, 1, Q, K]
    scaling: float,
    dropout: float = 0.0,
    *,
    log_gates: torch.Tensor,  # [C, C, H], already logsigmoid'd
    q_channels: torch.Tensor,  # [B, Q] long
    k_channels: torch.Tensor,  # [B, K] long
    sink_bias: Optional[torch.Tensor] = None,  # [B, H, Q, 1], already logsigmoid'd
    head_chunk: int = _QK_GATE_HEAD_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``eager_attention_forward`` with the channel log-bias added inside, per head.

    Arithmetically identical to adding the bias to the mask first and calling HF's
    ``eager_attention_forward``, but it never builds a ``[B, H, Q, K]`` *mask*.

    That matters because the gate bias varies per head while the base model's mask
    does not, and the mask is the only hook HF's eager path offers. Going through it
    forces the mask up to full per-head size, where it stays live for the whole of
    attention — one extra full-size tensor on top of the three eager already needs.
    Measured on gpt-oss-20b, that is 4.03 vs 3.03 full-size tensors of peak, so
    gating cost a third more memory than the base model for a bias with only C x C
    distinct values per head. At a 12288-token cap one such tensor is 18 GiB, which
    is what pushed 2-GPU serving off an 80 GiB card while base fit.

    Chunking over heads instead makes the score matrix a per-chunk transient, so
    peak drops to the returned ``attn_weights`` plus ``head_chunk / H`` of the
    scratch. Heads are independent in attention -- separate scores, separate softmax
    (the sink is per-head too) -- so this is a loop over the head axis and nothing
    more; no rescaling, no accumulation across chunks.

    ``attn_weights`` is still returned whole, since callers and tests expect it.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    batch, num_heads, q_len, _ = query.shape
    k_len = key_states.shape[-2]
    head_dim = value_states.shape[-1]

    attn_output = query.new_empty(batch, num_heads, q_len, head_dim)
    attn_weights = query.new_empty(batch, num_heads, q_len, k_len, dtype=value_states.dtype)

    # Cast the tiny [C, C, H] table once, so each gather runs in the score dtype
    # rather than fp32 (values unchanged -- logsigmoid already ran in fp32).
    gates = log_gates.to(query.dtype)
    q_idx = q_channels.unsqueeze(2)
    k_idx = k_channels.unsqueeze(1)

    chunk = max(1, min(head_chunk, num_heads))
    for start in range(0, num_heads, chunk):
        stop = min(start + chunk, num_heads)
        # In-place from here on: every op below is elementwise, and each temporary
        # avoided is head_chunk/H of a full score matrix.
        scores = torch.matmul(query[:, start:stop], key_states[:, start:stop].transpose(2, 3))
        scores *= scaling
        if attention_mask is not None:
            scores += attention_mask  # [B, 1, Q, K] broadcasts over the head axis
        scores += gates[..., start:stop][q_idx, k_idx].permute(0, 3, 1, 2)

        sinks = module.sinks[start:stop].reshape(1, -1, 1, 1).expand(batch, -1, q_len, -1)
        if sink_bias is not None:
            # [B, H, Q, 1] -- per (query channel, head), so it varies down the sequence
            # while module.sinks does not. Added, not multiplied: same log-space
            # convention as the content gate.
            sinks = sinks + sink_bias[:, start:stop].to(sinks.dtype)
        combined = torch.cat([scores, sinks.to(scores.dtype)], dim=-1)
        combined -= combined.max(dim=-1, keepdim=True).values  # bf16 overflow guard, as HF
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        head_scores = probs[..., :-1]  # drop the sink column; rows sum to 1 - p(sink)
        head_scores = F.dropout(
            head_scores, p=dropout, training=module.training
        ).to(value_states.dtype)

        attn_weights[:, start:stop] = head_scores
        attn_output[:, start:stop] = torch.matmul(head_scores, value_states[:, start:stop])

    return attn_output.transpose(1, 2).contiguous(), attn_weights


def sink_biased_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    *,
    sink_bias: torch.Tensor,  # [B, H, Q, 1]
    **_ignored,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF's ``eager_attention_forward`` with a per-position bias on the sink logit.

    Needed because the stock function reads ``module.sinks`` off the module rather than
    taking it as an argument -- it accepts ``s_aux`` and ignores it -- so the sink cannot
    be gated by passing anything in. This is a line-for-line copy of the upstream body
    (including the bf16 overflow guard) with one addition: ``sink_bias``.

    Used on the TRAINING path only. There the content gate arrives already folded into
    ``attention_mask`` by ``_qk_gate_mask``, whose one-hot einsum feeds dense matmul
    kernels on the backward; the inference path instead gathers per head inside
    ``gated_eager_attention_forward`` to keep the score matrix a transient. Both add the
    same numbers, in the same log space, to the same logits.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    sinks = sinks + sink_bias.to(sinks.dtype)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # drop the sink column, as upstream does
    attn_weights = F.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GptOssAttentionGated(GptOssAttention, GatedAttentionMixin):
    """gpt-oss attention with learned per-head channel gates.

    ``qk_gating`` adds ``logsigmoid(head_gates[q_channel, k_channel, head])`` to the
    content logits as a pre-softmax log-bias; ``head_gates`` has shape
    ``[C, C, num_attention_heads]``. With ``gating_sink``, ``sink_gates`` ``[C, H]``
    adds ``logsigmoid(sink_gates[q_channel, head])`` to the sink logit.

    Gating is a no-op when ``token_channels is None`` (bit-exact identity to the
    base ``GptOssAttention``).

    Config fields expected on the passed config:
      - ``gating_num_channels`` (int, required)
      - ``gating_variant`` (must be ``"qk_gating"``)
      - ``gating_init_logit`` (float, optional, default 4.0)
      - ``gating_sink`` (bool, optional, default False)
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        if config._attn_implementation != "eager":
            raise ValueError(
                f"GptOssAttentionGated requires _attn_implementation='eager'; "
                f"got {config._attn_implementation!r}"
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
                "GptOssAttentionGated expects a non-None attention_mask "
                "(the qk_gating log-bias is added to it; build one upstream)"
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

        # Number of *new* query positions, captured before the cache grows K.
        new_len = value_states.shape[-2]

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        channels = None
        sink_bias = None
        if token_channels is not None:
            k_total = key_states.shape[-2]
            if self.sliding_window is not None and token_channels.shape[-1] > k_total:
                # Sliding layers keep only the most recent `sliding_window` keys, so
                # the cache is shorter than the full channel sequence. Key column k
                # is absolute position (total_len - k_total + k), hence the tail
                # slice. Taking the tail also keeps q_channels correct, since
                # _align_channels reads the last `new_len` entries.
                token_channels = token_channels[:, -k_total:]
            elif token_channels.shape[-1] > k_total:
                raise ValueError(
                    f"token_channels length {token_channels.shape[-1]} exceeds key length {k_total}"
                )
            if self.training:
                # Training keeps the mask route: its one-hot einsum feeds dense matmul
                # kernels on the backward instead of a scatter-add into the tiny
                # [C, C, H] gradient, and the trained arms were fitted through it.
                # Memory is not the constraint there (sequences are short and fixed).
                attention_mask = self._qk_gate_mask(attention_mask, token_channels, new_len, k_total)
            else:
                channels = self._align_channels(token_channels, new_len, k_total)

            if getattr(self, "gating_sink", False):
                # [B, Q] -> [B, H, Q, 1]. The query positions are the LAST new_len
                # entries, matching _align_channels; taking the head is a slice, not a
                # gather, because sink_gates has no key axis.
                q_ch = token_channels[:, -new_len:]
                sink_bias = F.logsigmoid(self.sink_gates)[q_ch]  # [B, Q, H]
                sink_bias = sink_bias.permute(0, 2, 1).unsqueeze(-1)

        if channels is None and sink_bias is None:
            # Ungated (token_channels=None -> bit-exact identity to base) or training
            # without a sink gate.
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,
                **kwargs,
            )
        elif channels is None:
            # Training with a sink gate: content bias is already in the mask, so only
            # the sink term is left to add.
            attn_output, attn_weights = sink_biased_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                sink_bias=sink_bias,
            )
        else:
            q_channels, k_channels = channels
            attn_output, attn_weights = gated_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                dropout=0.0,
                log_gates=F.logsigmoid(self.head_gates),
                q_channels=q_channels,
                k_channels=k_channels,
                sink_bias=sink_bias,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
