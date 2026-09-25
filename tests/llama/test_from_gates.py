"""Tests for ``LlamaGatedConfig`` and ``LlamaGatedForCausalLM.from_gates``.

CPU, tiny config, no real Llama download. The "infer real Llama from gate
shape" path is documented in the verification notebook and exercised
manually; we don't test it here to keep the suite GPU-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from heads_of_security.attention_gating import (
    LlamaAttentionGated,
    LlamaGatedConfig,
    LlamaGatedForCausalLM,
)
from heads_of_security.attention_gating.gating_core import infer_gating_from_state_dict
from tests.utils import tiny_llama_config


def _fake_gate_state_dict(shape: tuple[int, ...], n_layers: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        f"model.layers.{i}.self_attn.head_gates": torch.randn(*shape)
        for i in range(n_layers)
    }


# ---------------------------------------------------------------------------
# LlamaGatedConfig
# ---------------------------------------------------------------------------


def test_llama_gated_config_defaults():
    cfg = LlamaGatedConfig()
    assert cfg.gating_variant == "qk_gating"
    assert cfg.gating_num_channels == 2
    assert cfg.gating_init_logit == 4.0
    assert cfg.gating_generated_channel == 0
    assert cfg._attn_implementation == "eager"


def test_llama_gated_config_coerces_non_eager_to_eager():
    cfg = LlamaGatedConfig(attn_implementation="sdpa")
    assert cfg._attn_implementation == "eager"


def test_llama_gated_config_accepts_explicit_eager():
    cfg = LlamaGatedConfig(attn_implementation="eager")
    assert cfg._attn_implementation == "eager"


def test_llama_gated_config_overrides_persist():
    cfg = LlamaGatedConfig(
        gating_variant="qk_gating",
        gating_num_channels=3,
        gating_init_logit=1.5,
        gating_generated_channel=1,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
        head_dim=4,
        vocab_size=32,
    )
    assert cfg.gating_variant == "qk_gating"
    assert cfg.gating_num_channels == 3
    assert cfg.gating_init_logit == 1.5
    assert cfg.gating_generated_channel == 1
    assert cfg.num_hidden_layers == 4
    assert cfg._attn_implementation == "eager"


# ---------------------------------------------------------------------------
# infer_gating_from_state_dict
# ---------------------------------------------------------------------------


def test_infer_qk_gating_from_shapes():
    sd = _fake_gate_state_dict(shape=(2, 2, 3), n_layers=4)
    info = infer_gating_from_state_dict(sd)
    assert info == {
        "gating_variant": "qk_gating",
        "gating_num_channels": 2,
        "num_hidden_layers": 4,
        "num_attention_heads": 3,
    }


def test_infer_rejects_non_square_head_gates():
    sd = _fake_gate_state_dict(shape=(2, 3), n_layers=4)
    with pytest.raises(ValueError, match=r"\[C, C, H\]"):
        infer_gating_from_state_dict(sd)


# ---------------------------------------------------------------------------
# from_gates end-to-end (tiny base saved to tmpdir, no real Llama download)
# ---------------------------------------------------------------------------


def _save_tiny_base(tmp_path: Path, gating_variant: str = "qk_gating") -> tuple[Path, LlamaConfig]:
    """Build & save a tiny vanilla Llama matching ``tiny_llama_config``.

    Returns (checkpoint_dir, config). ``gating_variant`` only affects which
    extra dims need to match; the base model itself is non-gated.
    """
    base_config = tiny_llama_config()
    torch.manual_seed(42)
    base = LlamaForCausalLM(base_config).eval()
    ckpt_dir = tmp_path / "tiny_base"
    base.save_pretrained(str(ckpt_dir))
    return ckpt_dir, base_config


def _gates_for_tiny(gating_variant: str) -> dict[str, torch.Tensor]:
    """Random gate state_dict matching tiny_llama_config (2 layers, 2 heads/kv_heads)."""
    assert gating_variant == "qk_gating", gating_variant
    shape = (2, 2, 2)  # [C, C, num_attention_heads]
    torch.manual_seed(7)
    return {
        f"model.layers.{i}.self_attn.head_gates": torch.randn(*shape)
        for i in range(2)
    }


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_from_gates_dict_with_tiny_base(tmp_path: Path, gating_variant: str):
    ckpt_dir, _ = _save_tiny_base(tmp_path)
    gates = _gates_for_tiny(gating_variant)

    model = LlamaGatedForCausalLM.from_gates(gates, base_model=str(ckpt_dir))

    assert isinstance(model, LlamaGatedForCausalLM)
    assert len(model.model.layers) == 2
    assert model.config.gating_variant == gating_variant
    assert model.config.gating_num_channels == 2

    for i, layer in enumerate(model.model.layers):
        expected = gates[f"model.layers.{i}.self_attn.head_gates"]
        assert isinstance(layer.self_attn, LlamaAttentionGated)
        assert torch.equal(layer.self_attn.head_gates.detach().to(expected.dtype), expected)

    # Forward smoke check.
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    token_channels = torch.tensor([[0, 1, 0, 1]])
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_channels=token_channels)
    assert not torch.isnan(out.logits).any()


def test_from_gates_path_input(tmp_path: Path):
    ckpt_dir, _ = _save_tiny_base(tmp_path)
    gates = _gates_for_tiny("qk_gating")
    gate_path = tmp_path / "gates.safetensors"
    save_file(gates, str(gate_path))

    model = LlamaGatedForCausalLM.from_gates(gate_path, base_model=str(ckpt_dir))

    for i, layer in enumerate(model.model.layers):
        expected = gates[f"model.layers.{i}.self_attn.head_gates"]
        assert torch.equal(layer.self_attn.head_gates.detach().to(expected.dtype), expected)


def test_from_gates_accepts_str_path(tmp_path: Path):
    ckpt_dir, _ = _save_tiny_base(tmp_path)
    gates = _gates_for_tiny("qk_gating")
    gate_path = tmp_path / "gates.safetensors"
    save_file(gates, str(gate_path))

    model = LlamaGatedForCausalLM.from_gates(str(gate_path), base_model=str(ckpt_dir))
    assert model.config.gating_variant == "qk_gating"


def test_from_pretrained_base_llama_works_with_default_gating(tmp_path: Path):
    """``config_class = LlamaGatedConfig`` means a bare ``from_pretrained`` on
    a non-gated base loads cleanly with default gating fields. head_gates land
    on the model uninitialized (HF doesn't know to init them) — caller is
    expected to either train them or overlay via ``from_gates``."""
    ckpt_dir, _ = _save_tiny_base(tmp_path)
    model = LlamaGatedForCausalLM.from_pretrained(str(ckpt_dir))

    assert isinstance(model.config, LlamaGatedConfig)
    assert model.config.gating_variant == "qk_gating"
    assert model.config.gating_num_channels == 2
    assert model.config._attn_implementation == "eager"
    head_gates = model.model.layers[0].self_attn.head_gates
    assert head_gates.shape == (2, 2, 2)  # [C, C, num_attention_heads]


def test_from_gates_unknown_layer_count_raises():
    """No base_model + no config + unknown layer count → registry miss."""
    gates = _fake_gate_state_dict(shape=(2, 2, 3), n_layers=5)  # not in registry
    with pytest.raises(ValueError, match="Cannot infer base_model"):
        LlamaGatedForCausalLM.from_gates(gates)
