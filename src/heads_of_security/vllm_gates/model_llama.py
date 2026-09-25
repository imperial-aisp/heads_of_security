"""Channel-gated Llama for vLLM (Llama-3.1-8B / Llama-3.3-70B checkpoints).

Same composition as :mod:`model` (gpt-oss), through llama.py's stock hooks
(``LlamaForCausalLM.__init__(layer_type=)`` / ``LlamaDecoderLayer(attn_layer_type=)``),
with one difference: ``LlamaAttention`` has no ``_build_attention`` factory —
its ``Attention`` is constructed inline — so the gated subclass evicts the
stock layer from ``static_forward_context`` (``Attention.__init__`` raises
``ValueError`` on a duplicate prefix) and rebuilds it with the gated backend.

Two simplifications vs gpt-oss: no attention sinks, no alternating sliding
window. And no weight-name mapper is needed: the vLLM module path of the
attention module is literally ``model.layers.N.self_attn``, so the trained
checkpoint key ``model.layers.N.self_attn.head_gates`` matches as-is.

The channel count comes from the snapshot's ``gating_num_channels`` (3 for the
released gates: trusted / untrusted / scaffold); the ungated identity index is
therefore ``C``, appended by ``build_gate_log_table``.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)
from vllm.v1.attention.backend import AttentionType

from heads_of_security.vllm_gates.backend import GatedTritonAttentionBackend
from heads_of_security.vllm_gates.model import build_gate_log_table, check_variant


class LlamaAttentionGated(LlamaAttention):
    def __init__(
        self,
        config,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_position_embeddings=8192,
        quant_config=None,
        bias=False,
        bias_o_proj=False,
        cache_config=None,
        prefix="",
        attn_type=AttentionType.DECODER,
    ):
        assert attn_type == AttentionType.DECODER, attn_type
        # The rebuild below hardcodes per_layer_sliding_window=None; refuse
        # configs where that would silently change the attention pattern.
        assert getattr(config, "layer_types", None) is None, (
            f"sliding-window llama variants unsupported: {config.layer_types}"
        )
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type,
        )
        num_channels = getattr(config, "gating_num_channels", 3)
        check_variant(config)
        # Full head width on every rank (the checkpoint tensor is unsharded);
        # finalize_gates() slices this rank's heads. dtype pinned: model init
        # runs under set_default_torch_dtype(bfloat16) and the loader would
        # silently round the fp32 gate logits on copy.
        self.head_gates = nn.Parameter(
            torch.zeros(num_channels, num_channels, config.num_attention_heads,
                        dtype=torch.float32),
            requires_grad=False,
        )
        attn_prefix = f"{prefix}.attn"
        ctx = get_current_vllm_config().compilation_config.static_forward_context
        evicted = ctx.pop(attn_prefix, None)
        assert evicted is self.attn, attn_prefix
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=None,
            attn_type=attn_type,
            prefix=attn_prefix,
            attn_backend=GatedTritonAttentionBackend,
        )

    def finalize_gates(self) -> None:
        # self.num_heads is already TP-local in LlamaAttention.
        self.attn.gate_log_table = build_gate_log_table(self.head_gates, self.num_heads)


class LlamaDecoderLayerGated(LlamaDecoderLayer):
    def __init__(self, vllm_config, prefix="", config=None):
        super().__init__(vllm_config, prefix, config,
                         attn_layer_type=LlamaAttentionGated)


class LlamaGatedForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        cfg = vllm_config.model_config.hf_config
        # runner_patch labels generated tokens channel 0 unconditionally.
        assert getattr(cfg, "gating_generated_channel", 0) == 0, (
            f"gating_generated_channel={cfg.gating_generated_channel} is not "
            f"supported by the vLLM runner (generated tokens are channel 0)")
        super().__init__(vllm_config=vllm_config, prefix=prefix,
                         layer_type=LlamaDecoderLayerGated)

    def load_weights(self, weights) -> set[str]:
        loaded = super().load_weights(weights)
        gated = 0
        for module in self.modules():
            if isinstance(module, LlamaAttentionGated):
                module.finalize_gates()
                gated += 1
        # Counts every layer, so PP (which would shard layers) is implicitly
        # rejected here; only TP is supported.
        assert gated == self.config.num_hidden_layers, (
            f"finalized gates on {gated} layers, expected "
            f"{self.config.num_hidden_layers}"
        )
        # The loader drops checkpoint keys it does not recognise without error, so a
        # naming drift would leave head_gates at its zeros init and produce an arm that
        # looks plausible and means nothing. Detected by value: the loader's returned
        # manifest does not list the gate params (measured on gpt-oss: 0 of 24 even
        # when they load correctly), so a manifest check would reject everything.
        #
        # Skipped when build_snapshot recorded that the table it wrote was itself
        # all-zero -- a legitimate configuration that is byte-identical to the zeros
        # init, and therefore has no wrong result to catch. Defaults to True, so every
        # pre-existing snapshot and every trained arm is guarded exactly as before.
        if getattr(self.config, "heads_of_security_head_gates_nonzero", True):
            n_zero = sum(1 for m in self.modules()
                         if isinstance(m, LlamaAttentionGated) and not m.head_gates.any())
            assert n_zero == 0, (
                f"{n_zero} layers have all-zero head_gates -- the gate tables did not "
                f"load from the checkpoint (expected keys like "
                f"model.layers.N.self_attn.head_gates)"
            )
        return loaded
