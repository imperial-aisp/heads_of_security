"""Model-loading helpers: build a gated model + channel tokenizer from a gates file."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoConfig

from heads_of_security.attention_gating import (
    GptOssGatedForCausalLM,
    GptOssGatedTokenizer,
    LlamaGatedForCausalLM,
    LlamaGatedTokenizer,
)

# config.model_type -> (gated ForCausalLM, gated tokenizer).
_GATED_BY_TYPE = {
    "llama": (LlamaGatedForCausalLM, LlamaGatedTokenizer),
    "gpt_oss": (GptOssGatedForCausalLM, GptOssGatedTokenizer),
}


def gated_classes_for(model_path: str):
    """Return ``(GatedForCausalLM, GatedTokenizer)`` for ``model_path`` by model_type."""
    model_type = AutoConfig.from_pretrained(model_path).model_type
    if model_type not in _GATED_BY_TYPE:
        raise ValueError(
            f"No gated classes registered for model_type {model_type!r}; "
            f"supported: {sorted(_GATED_BY_TYPE)}"
        )
    return _GATED_BY_TYPE[model_type]


def _layer_device_map(num_layers: int, num_gpus: int) -> dict[str, int]:
    """Build a device_map that keeps each decoder layer on one GPU."""
    device_map: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": num_gpus - 1,
        "lm_head": num_gpus - 1,
    }
    layers_per_gpu = (num_layers + num_gpus - 1) // num_gpus
    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = i // layers_per_gpu
    return device_map


def load_gated_model(
    gates: str,
    model: str,
    device: str = "cuda:0",
    *,
    untrusted_inline_roles: Optional[list[str]] = None,
    untrusted_header_roles: Optional[list[str]] = None,
    chat_template_channel: Optional[bool] = None,
    role_remap: Optional[dict[str, str]] = None,
):
    """Load a gated model from a trained gates safetensors file.

    Dispatches to the gated class for ``model``'s ``model_type`` (Llama or gpt-oss).
    Policy kwargs (role lists, ``chat_template_channel``, ``role_remap``) default to
    the tokenizer's defaults when ``None``; see ``from_pretrained`` for the sidecar
    fallback chain.

    ``device="auto"`` shards decoder layers across all visible GPUs. The model is
    loaded on CPU first and then dispatched with a per-layer device map: accelerate's
    own ``device_map="auto"`` splits *within* a gated decoder layer (the extra gate
    parameters change its memory profile), which breaks the attention forward.
    """
    gated_cls, tok_cls = gated_classes_for(model)

    if device == "auto":
        from accelerate import dispatch_model

        num_gpus = torch.cuda.device_count()
        device_map = _layer_device_map(AutoConfig.from_pretrained(model).num_hidden_layers, num_gpus)
        gated_model = gated_cls.from_gates(gates=gates, base_model=model, dtype=torch.bfloat16, device_map="cpu")
        gated_model = dispatch_model(gated_model, device_map=device_map)
    elif device == "cpu":
        gated_model = gated_cls.from_gates(gates=gates, base_model=model, dtype=torch.bfloat16)
    else:
        gated_model = gated_cls.from_gates(gates=gates, base_model=model, dtype=torch.bfloat16, device_map=device)

    if chat_template_channel is None:
        # The tokenizer must emit exactly the channels the gates were trained on: a
        # 3-channel table means the chat-template scaffold had its own channel.
        chat_template_channel = int(gated_model.config.gating_num_channels) == 3
    tok = tok_cls.from_pretrained(
        model,
        untrusted_inline_roles=untrusted_inline_roles,
        untrusted_header_roles=untrusted_header_roles,
        chat_template_channel=chat_template_channel,
        role_remap=role_remap,
    )
    if tok.num_channels != int(gated_model.config.gating_num_channels):
        raise ValueError(
            f"tokenizer renders {tok.num_channels} channels but the gates have "
            f"{gated_model.config.gating_num_channels}; the channel ids would not mean what the gates were trained on"
        )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    gated_model.generation_config.pad_token_id = tok.pad_token_id
    return gated_model, tok
