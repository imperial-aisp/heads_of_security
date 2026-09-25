"""Channel-gated Llama causal LM: the stock HF Llama with :class:`LlamaAttentionGated` in
every layer and the :class:`GatedCausalLMMixin` loader surface."""

from __future__ import annotations

import logging

import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)

from .gated_attention import LlamaAttentionGated
from .gating_core import GatedCausalLMMixin

logger = logging.getLogger(__name__)


# Layer-count → HF model id for ``from_gates`` auto-inference. The four
# supported Llamas all have distinct ``num_hidden_layers``, so the snapshot's
# per-layer count alone identifies the architecture.
_LLAMA_BASE_MODEL_REGISTRY: dict[int, str] = {
    80: "meta-llama/Llama-3.3-70B-Instruct",
    32: "meta-llama/Llama-3.1-8B-Instruct",
    28: "meta-llama/Llama-3.2-3B-Instruct",
    16: "meta-llama/Llama-3.2-1B-Instruct",
}


class LlamaGatedConfig(LlamaConfig):
    def __init__(
        self,
        gating_variant: str = "qk_gating",
        gating_num_channels: int = 2,
        gating_init_logit: float = 4.0,
        gating_generated_channel: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gating_variant = gating_variant
        self.gating_num_channels = gating_num_channels
        self.gating_init_logit = gating_init_logit
        self.gating_generated_channel = gating_generated_channel
        self._attn_implementation = "eager"


class LlamaGatedDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttentionGated(config, layer_idx=layer_idx)


class LlamaGatedModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            LlamaGatedDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        )


class LlamaGatedForCausalLM(GatedCausalLMMixin, LlamaForCausalLM):
    config_class = LlamaGatedConfig
    _BASE_MODEL_REGISTRY = _LLAMA_BASE_MODEL_REGISTRY

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaGatedModel(config)
        self.post_init()
        self._init_generated_channel(config)

    def forward(self, *, token_channels=None, position_ids=None, **kwargs):
        return super().forward(token_channels=token_channels, position_ids=position_ids, **kwargs)
