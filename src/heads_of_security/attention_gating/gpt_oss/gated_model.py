"""Channel-gated gpt-oss model (``openai/gpt-oss-20b`` / ``-120b``).

Thin wrapper over ``GptOssForCausalLM``: every decoder layer swaps its
``self_attn`` for :class:`~.gated_attention.GptOssAttentionGated`. The MoE MLP,
router and sinks are reused unchanged — gating only touches the attention scores.

``token_channels`` reaches the attention modules via ``**kwargs`` threaded by
``GptOssModel.forward`` → ``GptOssDecoderLayer.forward`` → ``self_attn``
(verified for transformers 5.9.0); the only override needed on the LM head is to
surface ``token_channels`` / ``position_ids`` in the ``forward`` signature so HF
``generate()`` forwards them.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssModel,
)

from ..gating_core import GatedCausalLMMixin
from .gated_attention import GptOssAttentionGated

logger = logging.getLogger(__name__)

# Layer-count → HF model id for ``from_gates`` auto-inference. The two released
# gpt-oss sizes have distinct ``num_hidden_layers``, so the per-layer count in a
# gates snapshot identifies the architecture on its own.
_GPT_OSS_BASE_MODEL_REGISTRY: dict[int, str] = {
    24: "openai/gpt-oss-20b",
    36: "openai/gpt-oss-120b",
}


class GptOssGatedConfig(GptOssConfig):
    """``GptOssConfig`` plus the gating fields, with eager attention forced.

    ``GptOssConfig`` is a ``@strict`` dataclass; the gating fields are set after
    ``super().__init__`` on this (undecorated) subclass.
    """

    def __init__(
        self,
        gating_variant: str = "qk_gating",
        gating_num_channels: int = 2,
        gating_init_logit: float = 4.0,
        gating_generated_channel: int = 0,
        gating_sink: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gating_variant = gating_variant
        self.gating_num_channels = gating_num_channels
        self.gating_init_logit = gating_init_logit
        self.gating_generated_channel = gating_generated_channel
        # Gate the attention sink as well as the content channels. Off by default so
        # every existing checkpoint loads and runs exactly as before.
        self.gating_sink = gating_sink
        # qk-gating adds a float log-bias to the mask; under sdpa the mask is a
        # bool tensor (or None), which would silently produce garbage.
        self._attn_implementation = "eager"


class GptOssGatedDecoderLayer(GptOssDecoderLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GptOssAttentionGated(config, layer_idx=layer_idx)


class GptOssGatedModel(GptOssModel):
    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            GptOssGatedDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        )


class GptOssGatedForCausalLM(GatedCausalLMMixin, GptOssForCausalLM):
    config_class = GptOssGatedConfig
    _BASE_MODEL_REGISTRY = _GPT_OSS_BASE_MODEL_REGISTRY

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.model = GptOssGatedModel(config)
        self.post_init()
        self._init_generated_channel(config)

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """Let transformers pick the fast MoE kernel for this subclass.

        transformers decides whether a model can have its experts implementation set by
        *opening the defining module's source file and grepping it* for
        ``"@use_experts_implementation"`` (``modeling_utils._can_set_experts_implementation``).
        That string lives in ``modeling_gpt_oss.py``; this subclass lives here, so the
        heuristic returns False, ``from_pretrained`` never resolves an implementation, and
        the model silently falls back to ``eager`` -- the Python loop over all 32 experts --
        while an identical stock load gets ``grouped_mm``.

        The consequences were not cosmetic. Measured on a 133-token prompt, one forward:

            eager       206 ms   logits differ from stock base (max 1.77, 196k/201k)
            grouped_mm   32 ms   logits BIT-IDENTICAL to stock base

        6.4x slower, and not bit-comparable to the base arm it is measured against -- the
        two kernels sum the expert outputs in different orders, which is ~1 ULP in bf16 at
        layer 0 and compounds over 24 layers into a greedy-decoding fork.

        We subclass the attention only; ``GptOssExperts`` is reused unmodified, so the
        capability is real and the heuristic is simply looking at the wrong file. Defer to
        the base class, which is where the experts are actually defined.
        """
        return GptOssForCausalLM._can_set_experts_implementation()

    def forward(self, *, token_channels=None, position_ids=None, **kwargs):
        return super().forward(token_channels=token_channels, position_ids=position_ids, **kwargs)

