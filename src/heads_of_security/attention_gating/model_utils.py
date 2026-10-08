"""Model-loading helpers: build a gated model + channel tokenizer from released gates.

``load_gated_model`` takes a Hub repo id (the repo names its own base model);
``load_gated_model_local`` takes a gates file plus an explicit base model.
"""

from __future__ import annotations

import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
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


GATES_FILE = "gates.safetensors"
GATES_CONFIG_FILE = "gates_config.json"


def load_gated_model(
    repo_id: str,
    device: str = "cuda:0",
    *,
    untrusted_roles: Optional[list[str]] = None,
    untrusted_inline_roles: Optional[list[str]] = None,
):
    """Load released gates from the Hub, e.g. ``load_gated_model("imperial-cpg/hos-gpt-oss-20b")``.

    The repo holds ``gates.safetensors`` and a ``gates_config.json`` naming the base
    model, which is fetched and gated. ``untrusted_roles`` are the roles whose whole
    message is untrusted (default ``tool``/``ipython``, as in training);
    ``untrusted_inline_roles`` are roles inlined into the open turn as untrusted
    content (default ``input``). For a local gates file, a different base model, or
    the full tokenizer policy, use ``load_gated_model_local``.
    """
    try:
        config_path = hf_hub_download(repo_id, GATES_CONFIG_FILE)
    except EntryNotFoundError as e:
        raise ValueError(
            f"{repo_id!r} has no {GATES_CONFIG_FILE}: not a heads-of-security gates repo "
            "(use load_gated_model_local for a bare gates file)"
        ) from e
    with open(config_path) as f:
        base_model = json.load(f)["base_model"]
    return load_gated_model_local(
        hf_hub_download(repo_id, GATES_FILE),
        base_model,
        device,
        untrusted_header_roles=untrusted_roles,
        untrusted_inline_roles=untrusted_inline_roles,
    )


def load_gated_model_local(
    gates: str,
    model: str,
    device: str = "cuda:0",
    *,
    untrusted_inline_roles: Optional[list[str]] = None,
    untrusted_header_roles: Optional[list[str]] = None,
    chat_template_channel: Optional[bool] = None,
    role_remap: Optional[dict[str, str]] = None,
):
    """Load a gated model from a local gates safetensors file on top of ``model``.

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
