"""Shared helpers for attention-gating tests."""

from __future__ import annotations

from typing import Optional

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from heads_of_security.attention_gating import (
    GptOssGatedConfig,
    GptOssGatedForCausalLM,
    LlamaGatedForCausalLM,
)

# Architectures the model-agnostic suite parametrizes over.
ARCHS = ["llama", "gpt_oss"]
# Archs whose attention adds a learned per-head sink logit after the mask, so
# attention rows do not sum to 1 and a uniform gate is not exactly the identity.
ARCHS_WITH_SINKS = ["gpt_oss"]

# arch -> (base ForCausalLM, gated ForCausalLM)
_LM_CLASSES = {
    "llama": (LlamaForCausalLM, LlamaGatedForCausalLM),
    "gpt_oss": (GptOssForCausalLM, GptOssGatedForCausalLM),
}


def gated_lm_class(arch: str):
    return _LM_CLASSES[arch][1]


def base_lm_class(arch: str):
    return _LM_CLASSES[arch][0]


def tiny_gpt_oss_config(
    *,
    gating_variant: str = "qk_gating",
    init_logit: float = 4.0,
    generated_channel: int = 0,
    gating_num_channels: int = 2,
    sliding_window: int = 3,
):
    """Tiny CPU-friendly gated gpt-oss config.

    4 layers alternating ``sliding_attention`` / ``full_attention`` (the released
    checkpoints' pattern), so both layer types are exercised. ``sliding_window=3``
    is deliberately smaller than the sequence lengths used in tests, so the window
    actually bites. ``rope_parameters`` is pinned to keep YaRN from warning about
    an implicit-vs-explicit factor mismatch at these tiny dimensions.
    """
    return GptOssGatedConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=sliding_window,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
        gating_variant=gating_variant,
        gating_num_channels=gating_num_channels,
        gating_init_logit=init_logit,
        gating_generated_channel=generated_channel,
    )


def tiny_config(
    arch: str,
    *,
    gating_variant: str = "qk_gating",
    init_logit: float = 4.0,
    generated_channel: int = 0,
    gating_num_channels: int = 2,
):
    """Tiny CPU-friendly gated config for ``arch`` (``llama`` / ``gpt_oss``), with the
    gating fields set for the gated ForCausalLM."""
    if arch == "llama":
        return tiny_llama_config(
            gating_variant=gating_variant,
            init_logit=init_logit,
            generated_channel=generated_channel,
            gating_num_channels=gating_num_channels,
        )
    if arch == "gpt_oss":
        return tiny_gpt_oss_config(
            gating_variant=gating_variant,
            init_logit=init_logit,
            generated_channel=generated_channel,
            gating_num_channels=gating_num_channels,
        )
    raise ValueError(f"unknown arch {arch!r}")


def base_like_gated(arch: str, gated):
    """Build a non-gated base ForCausalLM sharing ``gated``'s weights (gates dropped),
    for gated-open == base equivalence checks."""
    base = base_lm_class(arch)(gated.config).eval()
    sd = {
        k: v
        for k, v in gated.state_dict().items()
        if not k.endswith(("head_gates", "sink_gates"))
    }
    missing, unexpected = base.load_state_dict(sd, strict=False)
    assert unexpected == [], f"unexpected keys loading base: {unexpected}"
    return base


def tiny_llama_config(
    gating_variant: Optional[str] = None,
    init_logit: float = 4.0,
    generated_channel: int = 0,
    gating_num_channels: int = 2,
) -> LlamaConfig:
    """Tiny CPU-friendly `LlamaConfig`.

    If ``gating_variant`` is given, also sets the four gating fields
    (``gating_num_channels``, ``gating_variant``, ``gating_init_logit``,
    ``gating_generated_channel``) expected by `LlamaAttentionGated` and
    `LlamaGatedForCausalLM`.

    Fields irrelevant for attention-only tests (``vocab_size``,
    ``intermediate_size``, extra ``num_hidden_layers``) are harmless — the
    attention module ignores them — and let the same config be reused by
    the full-model tests.
    """
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        max_position_embeddings=16,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    if gating_variant is not None:
        config.gating_num_channels = gating_num_channels
        config.gating_variant = gating_variant
        config.gating_init_logit = init_logit
        config.gating_generated_channel = generated_channel
    return config


def copy_base_to_gated(base: LlamaForCausalLM, gated: LlamaGatedForCausalLM) -> None:
    """Load base LM state_dict into gated LM. `head_gates` keeps its init."""
    missing, unexpected = gated.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [], f"unexpected keys: {unexpected}"
    assert all(k.endswith((".head_gates", ".sink_gates")) for k in missing), f"unexpected missing: {missing}"


def position_embeddings(
    batch_size: int, seq_len: int, head_dim: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) that make RoPE a pass-through: cos=1, sin=0."""
    cos = torch.ones(batch_size, seq_len, head_dim, dtype=dtype)
    sin = torch.zeros(batch_size, seq_len, head_dim, dtype=dtype)
    return cos, sin


def causal_mask(batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standard additive causal mask: [B, 1, Q, Q] with -inf above diagonal."""
    neg_inf = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    causal = torch.triu(neg_inf, diagonal=1)
    return causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
