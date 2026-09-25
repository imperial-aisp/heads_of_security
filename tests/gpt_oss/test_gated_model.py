"""Tests for ``GptOssGatedForCausalLM``.

Mirrors ``tests/test_gated_model.py`` (the Llama suite) for the qk-gating-only
gpt-oss stack. CPU, tiny configs.
"""

from __future__ import annotations

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssModel,
)

from heads_of_security.attention_gating import (
    GptOssAttentionGated,
    GptOssGatedDecoderLayer,
    GptOssGatedForCausalLM,
    GptOssGatedModel,
)
from tests.utils import base_like_gated, tiny_gpt_oss_config


def _build(*, init_logit: float = 20.0, generated_channel: int = 0):
    torch.manual_seed(0)
    config = tiny_gpt_oss_config(init_logit=init_logit, generated_channel=generated_channel)
    return GptOssGatedForCausalLM(config).eval(), config


def _prompt(config, batch_size: int = 1, seq_len: int = 6):
    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    channels = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.long)[:, :seq_len]
    if batch_size > 1:
        channels = channels.expand(batch_size, -1).contiguous()
    return input_ids, channels


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------


def test_class_hierarchy():
    model, config = _build()

    assert isinstance(model, GptOssForCausalLM)
    assert isinstance(model.model, GptOssGatedModel)
    assert isinstance(model.model, GptOssModel)
    assert len(model.model.layers) == config.num_hidden_layers

    for layer in model.model.layers:
        assert isinstance(layer, GptOssGatedDecoderLayer)
        assert isinstance(layer, GptOssDecoderLayer)
        assert isinstance(layer.self_attn, GptOssAttentionGated)
        assert isinstance(layer.self_attn, GptOssAttention)
        assert layer.self_attn.head_gates.shape == (
            config.gating_num_channels,
            config.gating_num_channels,
            config.num_attention_heads,
        )


def test_config_forces_eager():
    config = tiny_gpt_oss_config()
    assert config._attn_implementation == "eager"


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_forward_returns_logits():
    model, config = _build()
    input_ids, channels = _prompt(config)
    out = model(input_ids=input_ids, token_channels=channels)
    assert out.logits.shape == (input_ids.shape[0], input_ids.shape[1], config.vocab_size)
    assert not torch.isnan(out.logits).any()


def test_forward_without_channels_works():
    model, config = _build()
    input_ids, _ = _prompt(config)
    out = model(input_ids=input_ids)
    assert not torch.isnan(out.logits).any()


# ---------------------------------------------------------------------------
# Equivalence to the base model
# ---------------------------------------------------------------------------


def test_gated_matches_base_when_gates_open():
    model, config = _build(init_logit=20.0)
    base = base_like_gated("gpt_oss", model)
    input_ids, channels = _prompt(config)

    with torch.no_grad():
        gated_logits = model(input_ids=input_ids, token_channels=channels).logits
        base_logits = base(input_ids=input_ids).logits

    assert torch.allclose(gated_logits, base_logits, atol=1e-4, rtol=1e-4), (
        f"open-gate gated != base, max diff {(gated_logits - base_logits).abs().max().item()}"
    )


def test_channels_none_matches_base_bit_exact():
    model, config = _build(init_logit=4.0)  # non-open gates: still a no-op
    base = base_like_gated("gpt_oss", model)
    input_ids, _ = _prompt(config)

    with torch.no_grad():
        gated_logits = model(input_ids=input_ids).logits
        base_logits = base(input_ids=input_ids).logits

    assert torch.equal(gated_logits, base_logits)


def test_gated_can_set_experts_implementation():
    """The gated subclass must be allowed to use the fast MoE kernel.

    transformers gates this on a heuristic that opens the model class's module file and
    greps it for "@use_experts_implementation" (modeling_utils). That string lives in
    modeling_gpt_oss.py; this subclass lives in gated_model.py, so the heuristic returned
    False and from_pretrained silently left the model on the `eager` experts loop while an
    identical stock load got `grouped_mm`.

    Measured on the real 20b at a 133-token prompt, that was 206 ms vs 32 ms per forward
    (6.4x), and the two kernels accumulate expert outputs in different orders -- ~1 ULP in
    bf16, compounding over 24 layers into different greedy decodes. Silent, so only a test
    that asserts the capability will catch it coming back.
    """
    assert GptOssGatedForCausalLM._can_set_experts_implementation() is True
    assert (
        GptOssGatedForCausalLM._can_set_experts_implementation()
        is GptOssForCausalLM._can_set_experts_implementation()
    ), "gated must resolve the same experts capability as the base class it wraps"


def test_gated_resolves_same_experts_impl_as_base():
    """Whatever the platform supports, gated and stock must agree on the MoE kernel.

    On CPU both land on `eager` (grouped_mm needs a capable GPU), so this asserts equality
    rather than a specific value -- the failure mode being guarded is the two diverging.
    """
    model, _ = _build()
    base = base_like_gated("gpt_oss", model)

    gated_impl = model._check_and_adjust_experts_implementation(None)
    base_impl = base._check_and_adjust_experts_implementation(None)
    assert gated_impl == base_impl, (
        f"gated resolved experts={gated_impl!r} but stock resolved {base_impl!r}; "
        "the gated model would run a different MoE kernel from the arm it is compared to"
    )


# ---------------------------------------------------------------------------
# Closing a gate changes behaviour
# ---------------------------------------------------------------------------


def test_closing_gate_changes_logits():
    model, config = _build(init_logit=20.0)
    input_ids, channels = _prompt(config)

    with torch.no_grad():
        before = model(input_ids=input_ids, token_channels=channels).logits
        model.model.layers[0].self_attn.head_gates[0, 1, :] = -20.0
        after = model(input_ids=input_ids, token_channels=channels).logits

    assert (after - before).abs().max().item() > 1e-4


def test_closing_gate_is_a_no_op_without_channels():
    model, config = _build(init_logit=20.0)
    input_ids, _ = _prompt(config)

    with torch.no_grad():
        before = model(input_ids=input_ids).logits
        model.model.layers[0].self_attn.head_gates[0, 1, :] = -20.0
        after = model(input_ids=input_ids).logits

    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# Gradients / freezing
# ---------------------------------------------------------------------------


def test_loss_and_grad_flow_to_head_gates():
    model, config = _build(init_logit=2.0)
    model.freeze_base()
    input_ids, channels = _prompt(config)

    out = model(input_ids=input_ids, token_channels=channels, labels=input_ids)
    assert out.loss is not None
    out.loss.backward()

    grads = {
        name: p.grad
        for name, p in model.named_parameters()
        if p.requires_grad and name.endswith("head_gates")
    }
    assert grads
    assert any(g is not None and g.abs().sum() > 0 for g in grads.values()), (
        "no gradient reached any gate parameter"
    )


def test_freeze_base_counts_only_gates():
    model, config = _build()
    n_trainable = model.freeze_base()

    expected = sum(
        layer.self_attn.head_gates.numel() for layer in model.model.layers
    )
    assert n_trainable == expected
    assert n_trainable == (
        config.gating_num_channels**2 * config.num_attention_heads * config.num_hidden_layers
    )

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(name.endswith("head_gates") for name in trainable)


def test_generated_channel_recorded_on_model():
    model, _ = _build(generated_channel=1)
    assert model.gating_generated_channel == 1
