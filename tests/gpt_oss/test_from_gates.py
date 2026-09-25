"""Tests for ``GptOssGatedConfig`` and ``GptOssGatedForCausalLM.from_gates``.

Mirrors ``tests/test_from_gates.py``. CPU, tiny config, no real gpt-oss download —
the "infer the real checkpoint from the gate shape" path is covered by the
registry test plus the demonstration notebook.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from heads_of_security.attention_gating import (
    GptOssAttentionGated,
    GptOssGatedConfig,
    GptOssGatedForCausalLM,
)
from heads_of_security.attention_gating.gating_core import infer_gating_from_state_dict
from heads_of_security.attention_gating.gpt_oss.gated_model import _GPT_OSS_BASE_MODEL_REGISTRY
from tests.utils import tiny_gpt_oss_config

# tiny config: 4 layers, 4 attention heads, 2 channels
_TINY_GATE_SHAPE = (2, 2, 4)
_TINY_LAYERS = 4


def _fake_gate_state_dict(shape: tuple[int, ...], n_layers: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        f"model.layers.{i}.self_attn.head_gates": torch.randn(*shape) for i in range(n_layers)
    }


# ---------------------------------------------------------------------------
# GptOssGatedConfig
# ---------------------------------------------------------------------------


def test_gpt_oss_gated_config_defaults():
    config = GptOssGatedConfig()
    assert config.gating_variant == "qk_gating"
    assert config.gating_num_channels == 2
    assert config.gating_init_logit == 4.0
    assert config.gating_generated_channel == 0
    assert config._attn_implementation == "eager"


def test_gpt_oss_gated_config_coerces_non_eager_to_eager():
    config = GptOssGatedConfig(attn_implementation="sdpa")
    assert config._attn_implementation == "eager"


def test_gpt_oss_gated_config_accepts_explicit_eager():
    config = GptOssGatedConfig(attn_implementation="eager")
    assert config._attn_implementation == "eager"


def test_gpt_oss_gated_config_overrides_persist():
    config = GptOssGatedConfig(
        gating_variant="qk_gating",
        gating_num_channels=3,
        gating_init_logit=1.5,
        gating_generated_channel=2,
    )
    assert config.gating_num_channels == 3
    assert config.gating_init_logit == 1.5
    assert config.gating_generated_channel == 2


def test_registry_covers_both_released_sizes():
    assert _GPT_OSS_BASE_MODEL_REGISTRY == {
        24: "openai/gpt-oss-20b",
        36: "openai/gpt-oss-120b",
    }


# ---------------------------------------------------------------------------
# Gate-shape inference
# ---------------------------------------------------------------------------


def test_infer_qk_gating_from_shapes():
    gates = _fake_gate_state_dict(shape=(2, 2, 64), n_layers=24)
    inferred = infer_gating_from_state_dict(gates)
    assert inferred["gating_variant"] == "qk_gating"
    assert inferred["gating_num_channels"] == 2
    assert inferred["num_attention_heads"] == 64
    assert inferred["num_hidden_layers"] == 24


def test_infer_rejects_inconsistent_shapes():
    gates = _fake_gate_state_dict(shape=(2, 2, 4), n_layers=2)
    gates["model.layers.2.self_attn.head_gates"] = torch.randn(2, 2, 8)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        infer_gating_from_state_dict(gates)


# ---------------------------------------------------------------------------
# from_gates end-to-end (tiny base saved to tmpdir)
# ---------------------------------------------------------------------------


def _save_tiny_base(tmp_path: Path) -> Path:
    """Build and save a tiny vanilla gpt-oss matching ``tiny_gpt_oss_config``."""
    torch.manual_seed(42)
    base = GptOssForCausalLM(tiny_gpt_oss_config()).eval()
    ckpt_dir = tmp_path / "tiny_base"
    base.save_pretrained(str(ckpt_dir))
    return ckpt_dir


def _gates_for_tiny() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        f"model.layers.{i}.self_attn.head_gates": torch.randn(*_TINY_GATE_SHAPE)
        for i in range(_TINY_LAYERS)
    }


def test_from_gates_dict_with_tiny_base(tmp_path: Path):
    ckpt_dir = _save_tiny_base(tmp_path)
    gates = _gates_for_tiny()

    model = GptOssGatedForCausalLM.from_gates(gates, base_model=str(ckpt_dir))

    assert isinstance(model, GptOssGatedForCausalLM)
    assert len(model.model.layers) == _TINY_LAYERS
    assert model.config.gating_variant == "qk_gating"
    assert model.config.gating_num_channels == 2

    for i, layer in enumerate(model.model.layers):
        expected = gates[f"model.layers.{i}.self_attn.head_gates"]
        assert isinstance(layer.self_attn, GptOssAttentionGated)
        assert torch.equal(layer.self_attn.head_gates.detach().to(expected.dtype), expected)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            token_channels=torch.tensor([[0, 1, 0, 1]]),
        )
    assert not torch.isnan(out.logits).any()


def test_from_gates_path_input(tmp_path: Path):
    ckpt_dir = _save_tiny_base(tmp_path)
    gates = _gates_for_tiny()
    gate_path = tmp_path / "gates.safetensors"
    save_file(gates, str(gate_path))

    model = GptOssGatedForCausalLM.from_gates(gate_path, base_model=str(ckpt_dir))

    for i, layer in enumerate(model.model.layers):
        expected = gates[f"model.layers.{i}.self_attn.head_gates"]
        assert torch.equal(layer.self_attn.head_gates.detach().to(expected.dtype), expected)


def test_from_gates_accepts_str_path(tmp_path: Path):
    ckpt_dir = _save_tiny_base(tmp_path)
    gates = _gates_for_tiny()
    gate_path = tmp_path / "gates.safetensors"
    save_file(gates, str(gate_path))

    model = GptOssGatedForCausalLM.from_gates(str(gate_path), base_model=str(ckpt_dir))
    assert model.config.gating_variant == "qk_gating"


def test_from_pretrained_base_works_with_default_gating(tmp_path: Path):
    """A bare ``from_pretrained`` on a non-gated base loads with default gating.

    ``head_gates`` arrive uninitialized (HF doesn't know to init them); the caller
    either trains them or overlays a snapshot via ``from_gates``.
    """
    ckpt_dir = _save_tiny_base(tmp_path)
    model = GptOssGatedForCausalLM.from_pretrained(str(ckpt_dir))

    assert isinstance(model.config, GptOssGatedConfig)
    assert model.config.gating_variant == "qk_gating"
    assert model.config._attn_implementation == "eager"
    assert model.model.layers[0].self_attn.head_gates.shape == _TINY_GATE_SHAPE


def test_from_gates_unknown_layer_count_raises():
    gates = _fake_gate_state_dict(shape=(2, 2, 4), n_layers=5)  # 5 not in the registry
    with pytest.raises(ValueError, match="Cannot infer base_model"):
        GptOssGatedForCausalLM.from_gates(gates)
