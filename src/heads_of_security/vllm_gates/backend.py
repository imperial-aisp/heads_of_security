"""Channel-gated build of vLLM's Triton attention backend.

Same anchored-patch approach as ``kernel.py``: the stock
``vllm.v1.attention.backends.triton_attn`` source is rebuilt with

1. the gated kernel from :mod:`heads_of_security.vllm_gates.kernel` in place of the
   stock ``unified_attention``;
2. a plumbing block in ``TritonAttentionImpl.forward`` that, when the layer
   carries a ``gate_log_table`` ([C, C, H_local] logsigmoid'd, set by the
   gated model after weight load), fetches this step's query channels,
   scatters them into the layer's physical channel cache via
   ``slot_mapping`` (the exact addressing the KV write uses), and passes all
   three tensors to the kernel.

Layers without ``gate_log_table`` — every layer of a stock model — take the
``gates=None`` path, which is byte-for-byte the stock kernel behaviour.
"""

from __future__ import annotations

from heads_of_security.vllm_gates._patching import assert_vllm_version, patched_module

assert_vllm_version()

_PATCHES: list[tuple[str, str]] = [
    # -- use the gated kernel -------------------------------------------------
    (
        "from vllm.v1.attention.ops.triton_unified_attention import unified_attention",
        "from heads_of_security.vllm_gates.kernel import unified_attention_gated as unified_attention",
    ),
    # -- channel plumbing just before the kernel call -------------------------
    (
        """        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(""",
        """        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        _gate_table = getattr(layer, "gate_log_table", None)
        _sink_gate_table = getattr(layer, "sink_gate_log_table", None)
        _gate_q_ch = _gate_k_ch = None
        if _gate_table is not None:
            from heads_of_security.vllm_gates import channels as _ch

            _gate_q_ch = _ch.step_channels(num_actual_tokens)
            _gate_k_ch = _ch.kv_channel_cache(
                layer,
                key_cache.shape[0] * key_cache.shape[1],
                _gate_q_ch.device,
            )
            # Same slots the KV write used this step; idempotent across layers.
            # PAD_SLOT_ID is -1 and torch would write it into the LAST slot,
            # corrupting whichever request owns it, so drop negatives.
            _slots = attn_metadata.slot_mapping[:num_actual_tokens].long()
            _keep = _slots >= 0
            _gate_k_ch[_slots[_keep]] = _gate_q_ch[:num_actual_tokens][_keep]

        unified_attention(""",
    ),
    # -- thread the tensors into the call -------------------------------------
    (
        """            sinks=self.sinks,
            output_scale=output_scale,""",
        """            sinks=self.sinks,
            gates=_gate_table,
            sink_gates=_sink_gate_table,
            q_channels=_gate_q_ch,
            k_channels=_gate_k_ch,
            output_scale=output_scale,""",
    ),
]

_mod = patched_module(
    "vllm.v1.attention.backends.triton_attn",
    "heads_of_security.vllm_gates._backend_patched",
    _PATCHES,
)

GatedTritonAttentionBackend = _mod.TritonAttentionBackend
GatedTritonAttentionImpl = _mod.TritonAttentionImpl
