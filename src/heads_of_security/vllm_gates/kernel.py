"""Channel-gated build of vLLM's unified triton attention kernel.

Adds the attention-gate log-bias to the score tile, exactly as the HF gated
path does (``heads_of_security.attention_gating.gpt_oss.gated_attention``):

    S[q, k] += logsigmoid(head_gates)[q_channel(q), k_channel(k), head]

applied to real keys only -- by default the sink logit joins the softmax
*without* a bias, which is what makes a uniform gate drain probability mass to
the sink rather than cancel. The sink-gated checkpoints (``gating_sink: true``)
additionally bias the sink logit itself, per (query channel, head):

    sink[h] += logsigmoid(sink_gates)[q_channel(q), head]

implemented by adding the bias to the online-softmax's initial ``M`` (which
``init_softmax_M`` sets to the sink logit, with ``L = 1`` standing for exactly
that one lane).

New ``unified_attention`` kwargs (all default ``None`` == stock behaviour):

- ``gates``      -- fp32 ``[C, C, H_local]``, already ``logsigmoid``'d.
- ``sink_gates`` -- fp32 ``[C, H_local]``, already ``logsigmoid``'d (the padded
  ungated row is zeros, like ``gates``).
- ``q_channels`` -- int32 ``[num_tokens]``: channel of each *query* token in the
  flattened step batch (same layout as ``q``'s row axis).
- ``k_channels`` -- int32 ``[num_blocks * block_size]``: channel of every cached
  token, addressed physically like the K cache (block table indirection).
"""

from __future__ import annotations

import os

from heads_of_security.vllm_gates._patching import assert_vllm_version, patched_module

assert_vllm_version()

_PATCHES: list[tuple[str, str]] = [
    # -- kernel signature: pointers ------------------------------------------
    (
        """    qq_bias_ptr,
    # Scalars""",
        """    qq_bias_ptr,
    gates_ptr,
    sink_gates_ptr,
    q_channels_ptr,
    k_channels_ptr,
    # Scalars""",
    ),
    # -- kernel signature: constexpr toggles ---------------------------------
    (
        """    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool""",
        """    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_GATES: tl.constexpr,  # bool
    USE_SINK_GATES: tl.constexpr,  # bool
    NUM_CHANNELS: tl.constexpr,  # int
    USE_SOFTCAP: tl.constexpr,  # bool""",
    ),
    # -- q-channel load + sink bias, at the softmax-state init ----------------
    # The q-channel load must precede M's init so the sink bias can use it
    # (the tile-loop content bias below reuses the same gate_q_ch).
    (
        """    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)""",
        """    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    if USE_GATES:
        # Channel of each query row; masked rows read 0 (their scores are
        # already -inf, the bias is inert there).
        gate_q_ch = tl.load(
            q_channels_ptr + query_offset_0, mask=query_mask_0, other=0
        ).to(tl.int32)
    if USE_SINK_GATES:
        # Bias the sink logit per (query channel, head). M was initialised to
        # the sink logit with L = 1 representing exactly that one lane, so
        # adding here IS the gated sink: exp(sink + bias - M_final) in the
        # denominator. Rows where M is -inf (3D segments past the first, or
        # masked rows) absorb the add and stay -inf, so no segment logic is
        # mirrored.
        M += tl.load(
            sink_gates_ptr + gate_q_ch * num_query_heads + query_offset_1,
            mask=query_mask_0 & query_mask_1,
            other=0.0,
        )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)""",
    ),
    # -- the bias itself, per KV tile, after every other score modifier ------
    (
        """        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)""",
        """        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        if USE_GATES:
            # Channel of each cached key in this tile, via the same physical
            # block addressing as the K load above.
            gate_k_ch = tl.load(
                k_channels_ptr
                + physical_block_idx * BLOCK_SIZE
                + seq_offset % BLOCK_SIZE,
                mask=tile_mask,
                other=0,
            ).to(tl.int32)
        if USE_GATES:
            # gates is [C, C, H_local] contiguous fp32:
            # flat = (q_ch * C + k_ch) * H + head
            gate_flat = (
                gate_q_ch[:, None] * NUM_CHANNELS + gate_k_ch[None, :]
            ) * num_query_heads + query_offset_1[:, None]
            # -inf survives the add on masked lanes, so no extra masking.
            S += tl.load(gates_ptr + gate_flat)

        M, L, P, alpha = softmax_step(S, M, L)""",
    ),
    # -- wrapper signature ----------------------------------------------------
    (
        """    qq_bias=None,
    # Optional tensor for sinks""",
        """    qq_bias=None,
    # Channel gates (heads_of_security): logsigmoid tables + per-token channels.
    gates=None,
    sink_gates=None,
    q_channels=None,
    k_channels=None,
    # Optional tensor for sinks""",
    ),
    # -- refuse the INT4 special-case kernel ----------------------------------
    (
        """    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        assert use_causal and not use_per_seq_causal, (
            "INT4_PER_TOKEN_HEAD only supports causal attention"
        )""",
        """    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        assert gates is None, (
            "channel gates are not implemented for the INT4 packed-KV kernel"
        )
        assert use_causal and not use_per_seq_causal, (
            "INT4_PER_TOKEN_HEAD only supports causal attention"
        )""",
    ),
    # -- wrapper: resolve + validate the gate inputs --------------------------
    (
        """    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None""",
        """    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    use_gates = gates is not None
    if use_gates:
        assert q_channels is not None and k_channels is not None, (
            "gates require q_channels and k_channels"
        )
        assert gates.dtype == torch.float32 and gates.is_contiguous()
        assert gates.shape[0] == gates.shape[1], gates.shape
        assert gates.shape[2] == q.shape[1], (
            f"gates head dim {gates.shape[2]} != local query heads {q.shape[1]}"
        )
        assert q_channels.shape[0] >= q.shape[0], (
            f"q_channels covers {q_channels.shape[0]} tokens < {q.shape[0]} queries"
        )
        # The gather is unmasked by design (masked lanes are already -inf), so an
        # out-of-range channel id would read past the table and return a garbage
        # bias with no fault at all. The tape is range-checked on the host when
        # it is ingested and built (runner_patch: 0 <= id < ungated index), so
        # this device-side check is a debug aid: it costs four GPU->host syncs
        # per attention layer per step, one of them a reduction over the whole
        # physical channel cache. HEADS_OF_SECURITY_VLLM_GATES_CHECK_CHANNELS=1 enables it.
        if HOS_CHECK_CHANNELS:
            num_ch = gates.shape[0]
            assert int(q_channels.max()) < num_ch and int(k_channels.max()) < num_ch, (
                f"channel id >= {num_ch} would index outside the gate table"
            )
            assert int(q_channels.min()) >= 0 and int(k_channels.min()) >= 0, (
                "negative channel id"
            )
    num_gate_channels = gates.shape[0] if use_gates else 1

    use_sink_gates = sink_gates is not None
    if use_sink_gates:
        assert use_gates, "sink_gates require the content gate table"
        assert sinks is not None, (
            "sink_gates on a model without attention sinks"
        )
        assert sink_gates.dtype == torch.float32 and sink_gates.is_contiguous()
        assert sink_gates.shape == (num_gate_channels, q.shape[1]), (
            f"sink_gates {tuple(sink_gates.shape)} != "
            f"({num_gate_channels}, {q.shape[1]})"
        )
""",
    ),
    # -- wrapper: thread the args into the (single) launch --------------------
    (
        """        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,""",
        """        qq_bias_ptr=qq_bias,
        gates_ptr=gates,
        sink_gates_ptr=sink_gates,
        q_channels_ptr=q_channels,
        k_channels_ptr=k_channels,
        k_scale_cache_ptr=k_scale_ptr,""",
    ),
    (
        """        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),""",
        """        USE_QQ_BIAS=use_qq_bias,
        USE_GATES=use_gates,
        USE_SINK_GATES=use_sink_gates,
        NUM_CHANNELS=num_gate_channels,
        USE_SOFTCAP=(softcap > 0),""",
    ),
]

_mod = patched_module(
    "vllm.v1.attention.ops.triton_unified_attention",
    "heads_of_security.vllm_gates._kernel_patched",
    _PATCHES,
)
# Read by the patched wrapper (a module global of the patched source). Off by
# default: the host-side checks in runner_patch make it redundant, and it syncs.
_mod.HOS_CHECK_CHANNELS = (
    os.environ.get("HEADS_OF_SECURITY_VLLM_GATES_CHECK_CHANNELS", "0") == "1"
)

unified_attention_gated = _mod.unified_attention
