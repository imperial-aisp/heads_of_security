"""End-to-end plumbing tests for `LlamaGatedForCausalLM`.

CPU, tiny config.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from heads_of_security.attention_gating import (
    LlamaAttentionGated,
    LlamaGatedDecoderLayer,
    LlamaGatedForCausalLM,
    LlamaGatedModel,
)

from .utils import copy_base_to_gated, tiny_llama_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_batch(config: LlamaConfig, batch_size: int = 2, seq_len: int = 5):
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_channels = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 0, 1, 0]], dtype=torch.long)
    return input_ids, attention_mask, token_channels


# ---------------------------------------------------------------------------
# Test 1 — class hierarchy matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_class_hierarchy(gating_variant):
    config = tiny_llama_config(gating_variant=gating_variant)
    model = LlamaGatedForCausalLM(config).eval()

    assert isinstance(model.model, LlamaGatedModel)
    assert len(model.model.layers) == config.num_hidden_layers
    for layer in model.model.layers:
        assert isinstance(layer, LlamaGatedDecoderLayer)
        assert isinstance(layer.self_attn, LlamaAttentionGated)


# ---------------------------------------------------------------------------
# Test 2 — forward runs, shape matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_forward_returns_logits(gating_variant):
    config = tiny_llama_config(gating_variant=gating_variant)
    model = LlamaGatedForCausalLM(config).eval()
    input_ids, attention_mask, token_channels = _random_batch(config)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_channels=token_channels,
    )

    assert out.logits.shape == (input_ids.shape[0], input_ids.shape[1], config.vocab_size)


# ---------------------------------------------------------------------------
# Test 3 — gated ≡ base when gates are open (proves token_channels reaches attention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_gated_matches_base_when_gates_open(gating_variant):
    # Build base and gated with matching base weights; gated has high
    # init_logit so the gating path is near-identity.
    torch.manual_seed(1)
    base_config = tiny_llama_config()
    base = LlamaForCausalLM(base_config).eval()

    gated_config = tiny_llama_config(gating_variant=gating_variant, init_logit=20.0)
    gated = LlamaGatedForCausalLM(gated_config).eval()
    copy_base_to_gated(base, gated)

    input_ids, attention_mask, token_channels = _random_batch(base_config)

    # (a) token_channels=None -> gating path skipped -> bit-exact.
    base_out = base(input_ids=input_ids, attention_mask=attention_mask)
    gated_out = gated(input_ids=input_ids, attention_mask=attention_mask, token_channels=None)
    assert torch.equal(gated_out.logits, base_out.logits)

    # (b) token_channels provided + gates ≈ 1 -> allclose.
    gated_out = gated(input_ids=input_ids, attention_mask=attention_mask, token_channels=token_channels)
    assert torch.allclose(gated_out.logits, base_out.logits, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4 — closing a gate changes logits (kwarg actually reaches attention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_closing_gate_changes_logits(gating_variant):
    torch.manual_seed(1)
    base_config = tiny_llama_config()
    base = LlamaForCausalLM(base_config).eval()

    gated_config = tiny_llama_config(gating_variant=gating_variant, init_logit=20.0)
    gated = LlamaGatedForCausalLM(gated_config).eval()
    copy_base_to_gated(base, gated)

    # Close one asymmetric gate on the first layer.
    with torch.no_grad():
        gated.model.layers[0].self_attn.head_gates.flatten()[0] = -20.0

    input_ids, attention_mask, token_channels = _random_batch(base_config)
    base_out = base(input_ids=input_ids, attention_mask=attention_mask)
    gated_out = gated(input_ids=input_ids, attention_mask=attention_mask, token_channels=token_channels)

    # If token_channels didn't reach attention, gates wouldn't be consulted
    # and logits would match base. They must differ.
    assert not torch.allclose(gated_out.logits, base_out.logits, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5 — loss computes, gradients flow to head_gates
# ---------------------------------------------------------------------------


def test_loss_and_grad_flow_to_head_gates():
    config = tiny_llama_config(gating_variant="qk_gating")
    model = LlamaGatedForCausalLM(config).train()
    input_ids, attention_mask, token_channels = _random_batch(config)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_channels=token_channels,
        labels=input_ids,
    )

    assert out.loss.ndim == 0
    assert out.loss.requires_grad
    out.loss.backward()
    for layer in model.model.layers:
        assert layer.self_attn.head_gates.grad is not None


# ---------------------------------------------------------------------------
# Test 6 — freeze_base freezes everything except head_gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_freeze_base(gating_variant):
    config = tiny_llama_config(gating_variant=gating_variant)
    model = LlamaGatedForCausalLM(config)

    total_before = sum(p.numel() for p in model.parameters())
    n_trainable = model.freeze_base()

    # Only head_gates are trainable.
    assert n_trainable > 0
    assert n_trainable < total_before

    for name, p in model.named_parameters():
        if "head_gates" in name:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"

    # Trainable count matches the sum of head_gates sizes.
    expected = sum(
        layer.self_attn.head_gates.numel() for layer in model.model.layers
    )
    assert n_trainable == expected

