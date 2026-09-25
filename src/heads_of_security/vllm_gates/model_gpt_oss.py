"""Channel-gated gpt-oss model for vLLM.

Three thin subclasses down the stock composition hooks
(``TransformerBlock.attention_cls`` / ``GptOssModel.block_cls``), plus:

- ``head_gates`` — a per-layer ``[C, C, H]`` parameter (raw logits, matching
  the training checkpoint layout ``model.layers.N.self_attn.head_gates``; the
  stock ``hf_to_vllm_mapper`` rewrites ``.self_attn.`` → ``.attn.``). Kept
  full-width on every rank; the TP shard is taken when the table is finalized.
- ``sink_gates`` — ``[C, H]``, present when the snapshot's ``gating_sink`` is true.
- ``finalize_gates`` — after weight load, precomputes the logsigmoid'd fp32
  table(s), slices this rank's heads, and hangs them on the ``Attention`` layer as
  ``gate_log_table`` / ``sink_gate_log_table``, where the gated backend picks them up.

The attention layers are built with ``attn_backend=GatedTritonAttentionBackend``
so gating works regardless of the global backend selection.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    OAIAttention,
    TransformerBlock,
)
from vllm.v1.attention.backend import AttentionType

from heads_of_security.vllm_gates.backend import GatedTritonAttentionBackend
from heads_of_security.vllm_gates.gate_tables import build_gate_log_table, build_sink_gate_log_table, check_variant


class OAIAttentionGated(OAIAttention):
    def __init__(self, config, quant_config=None, cache_config=None, prefix=""):
        super().__init__(
            config, quant_config=quant_config, cache_config=cache_config, prefix=prefix
        )
        num_channels = getattr(config, "gating_num_channels", 3)
        check_variant(config)
        # Full head width on every rank (the checkpoint tensor is unsharded);
        # finalize_gates() slices this rank's heads into the runtime table.
        # dtype pinned: model init runs under set_default_torch_dtype(bfloat16),
        # and default_weight_loader would then downcast the fp32 gate logits on
        # copy -- silently, and by ~8% of the trained T->U contrast.
        self.head_gates = nn.Parameter(
            torch.zeros(num_channels, num_channels, config.num_attention_heads,
                        dtype=torch.float32),
            requires_grad=False,
        )
        if getattr(config, "gating_sink", False):
            # Sink-gated checkpoints (gates_sink_*): one extra gate per
            # (query channel, head) on the attention-sink logit, so channel
            # discrimination stops paying its content bias to the ungated sink.
            self.sink_gates = nn.Parameter(
                torch.zeros(num_channels, config.num_attention_heads,
                            dtype=torch.float32),
                requires_grad=False,
            )

    def _build_attention(self, config, cache_config, quant_config, prefix) -> Attention:
        # Stock body with the gated backend forced per-layer.
        sliding_window = config.sliding_window if self.layer_idx % 2 == 0 else None
        return Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
            attn_backend=GatedTritonAttentionBackend,
        )

    def finalize_gates(self) -> None:
        # Plain attribute on the Attention module: the gated backend reads it
        # via getattr; absent (stock models) means the ungated kernel path.
        self.attn.gate_log_table = build_gate_log_table(
            self.head_gates, self.num_local_attention_heads
        )
        if hasattr(self, "sink_gates"):
            self.attn.sink_gate_log_table = build_sink_gate_log_table(
                self.sink_gates, self.num_local_attention_heads
            )


class TransformerBlockGated(TransformerBlock):
    attention_cls = OAIAttentionGated


class GptOssModelGated(GptOssModel):
    block_cls = TransformerBlockGated


class GptOssGatedForCausalLM(GptOssForCausalLM):
    def __init__(self, vllm_config, prefix: str = ""):
        # nn.Module init only; then mirror the stock ctor with the gated model.
        # (Calling super().__init__ would build the full ungated 20B first.)
        nn.Module.__init__(self)
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.models.utils import maybe_prefix

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        # runner_patch labels generated tokens channel 0 unconditionally.
        assert getattr(self.config, "gating_generated_channel", 0) == 0, (
            f"gating_generated_channel={self.config.gating_generated_channel} is "
            f"not supported by the vLLM runner (generated tokens are channel 0)")
        self.model = GptOssModelGated(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights) -> set[str]:
        loaded = super().load_weights(weights)
        gated = 0
        for module in self.modules():
            if isinstance(module, OAIAttentionGated):
                module.finalize_gates()
                gated += 1
        assert gated == self.config.num_hidden_layers, (
            f"finalized gates on {gated} layers, expected "
            f"{self.config.num_hidden_layers}"
        )
        # The mxfp4 loader drops checkpoint keys it does not recognise without error
        # (`if name not in params_dict: continue`), so a naming drift would leave the
        # tables at their zeros init and produce an arm that looks perfectly plausible
        # and means nothing.
        #
        # Detected by value, because the loader's returned manifest does NOT list the
        # gate params -- measured: 0 of 24 even on a trained arm that loads correctly,
        # so a manifest check rejects everything.
        #
        # The catch a value check has: an all-zero table is a LEGITIMATE configuration
        # (the uniform-gate ablation pins trusted/scaffold/sink at logit 0.0) and is
        # byte-identical to the zeros init. build_snapshot therefore records whether
        # the tables it wrote were non-zero, and the check is skipped when they were
        # not. Defaults to True, so every pre-existing snapshot -- and every trained
        # arm, whose gates are never all-zero -- is guarded exactly as before.
        #
        # Nothing is lost by skipping it: if a deliberately all-zero table fails to
        # load, the zeros init leaves the model BIT-IDENTICAL to the intended one, so
        # there is no wrong result to catch.
        if getattr(self.config, "heads_of_security_head_gates_nonzero", True):
            n_zero = sum(1 for m in self.modules()
                         if isinstance(m, OAIAttentionGated) and not m.head_gates.any())
            assert n_zero == 0, (
                f"{n_zero} layers have all-zero head_gates -- the gate tables did not "
                f"load from the checkpoint (expected keys like "
                f"model.layers.N.self_attn.head_gates)"
            )
        if getattr(self.config, "heads_of_security_sink_gates_nonzero", True):
            n_zero_sink = sum(
                1 for m in self.modules()
                if isinstance(m, OAIAttentionGated) and hasattr(m, "sink_gates")
                and not m.sink_gates.any()
            )
            assert n_zero_sink == 0, (
                f"{n_zero_sink} layers have all-zero sink_gates -- gating_sink is set "
                f"but the sink tables did not load (expected keys like "
                f"model.layers.N.self_attn.sink_gates)"
            )
        return loaded
