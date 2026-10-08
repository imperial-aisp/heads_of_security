"""Model builders for the training recipe (``_component_`` targets).

Both return a gated model with pretrained base weights and **freshly initialised** gates:
every ``head_gates`` (and ``sink_gates``) entry at ``gating_init_logit``. Filling them matters:
no base checkpoint carries gate tensors, so a bare ``from_pretrained`` would leave them at 0,
i.e. ``sigmoid(0) = 0.5`` on every pair instead of the identity.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig

from heads_of_security.attention_gating import GptOssGatedForCausalLM, LlamaGatedForCausalLM


def fresh_gates_state_dict(
    num_hidden_layers: int,
    num_attention_heads: int,
    *,
    gating_num_channels: int,
    gating_init_logit: float,
    gating_sink: bool = False,
) -> dict:
    """``head_gates [C, C, H]`` (and ``sink_gates [C, H]`` under ``gating_sink``) for every layer,
    keyed like a trained gates file. The presence of ``sink_gates`` is what makes the built
    model sink-gated (``infer_gating_from_state_dict`` reads it off the keys)."""
    shape = (gating_num_channels, gating_num_channels, num_attention_heads)
    sd = {f"model.layers.{i}.self_attn.head_gates": torch.full(shape, gating_init_logit)
          for i in range(num_hidden_layers)}
    if gating_sink:
        sink_shape = (gating_num_channels, num_attention_heads)
        sd.update({f"model.layers.{i}.self_attn.sink_gates": torch.full(sink_shape, gating_init_logit)
                   for i in range(num_hidden_layers)})
    return sd


def build_gated_llama(
    checkpoint_dir: str,
    gating_num_channels: int = 3,
    gating_init_logit: float = 1.0,
    gating_generated_channel: int = 0,
    **_ignored,
) -> LlamaGatedForCausalLM:
    """``LlamaGatedForCausalLM`` with pretrained weights and gates at ``gating_init_logit``."""
    from transformers import LlamaConfig

    hf_config = LlamaConfig.from_pretrained(checkpoint_dir)
    hf_config._attn_implementation = "eager"
    hf_config.gating_variant = "qk_gating"
    hf_config.gating_num_channels = gating_num_channels
    hf_config.gating_init_logit = gating_init_logit
    hf_config.gating_generated_channel = gating_generated_channel

    model = LlamaGatedForCausalLM.from_pretrained(checkpoint_dir, config=hf_config)
    # from_pretrained initialises missing params on the meta device; re-fill the gates.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.head_gates.fill_(gating_init_logit)
    return model


def build_gated(
    checkpoint_dir: str,
    gating_num_channels: int = 3,
    gating_init_logit: float = 1.0,
    gating_generated_channel: int = 0,
    gating_sink: bool = False,
    **_ignored,
):
    """Build a gated model with pretrained base weights + fresh gates, dispatching on
    ``config.model_type`` (``llama`` or ``gpt_oss``). ``gating_sink`` is gpt-oss only."""
    cfg = AutoConfig.from_pretrained(checkpoint_dir)
    if cfg.model_type == "llama":
        if gating_sink:
            raise ValueError("gating_sink needs an attention sink; Llama has none")
        return build_gated_llama(
            checkpoint_dir,
            gating_num_channels=gating_num_channels,
            gating_init_logit=gating_init_logit,
            gating_generated_channel=gating_generated_channel,
        )
    if cfg.model_type != "gpt_oss":
        raise ValueError(f"build_gated: unsupported model_type {cfg.model_type!r} (llama, gpt_oss)")

    fresh = fresh_gates_state_dict(
        cfg.num_hidden_layers, cfg.num_attention_heads,
        gating_num_channels=gating_num_channels, gating_init_logit=gating_init_logit, gating_sink=gating_sink,
    )
    model = GptOssGatedForCausalLM.from_gates(fresh, base_model=checkpoint_dir, dtype=torch.bfloat16)
    model.config.gating_generated_channel = gating_generated_channel
    model.gating_generated_channel = gating_generated_channel
    return model
