"""Unit tests for `LlamaAttentionGated`.

All tests run on CPU with tiny configs.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaAttention

from heads_of_security.attention_gating import LlamaAttentionGated

from .utils import causal_mask, position_embeddings, tiny_llama_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_base_weights(src: LlamaAttention, dst: LlamaAttentionGated) -> None:
    dst.q_proj.load_state_dict(src.q_proj.state_dict())
    dst.k_proj.load_state_dict(src.k_proj.state_dict())
    dst.v_proj.load_state_dict(src.v_proj.state_dict())
    dst.o_proj.load_state_dict(src.o_proj.state_dict())


def _build_pair(gating_variant: str, init_logit: float = 4.0):
    torch.manual_seed(0)
    config = tiny_llama_config(gating_variant=gating_variant, init_logit=init_logit)
    base = LlamaAttention(config, layer_idx=0).eval()
    gated = LlamaAttentionGated(config, layer_idx=0).eval()
    _copy_base_weights(base, gated)
    return base, gated, config


def _equivalence_inputs():
    """Shared fixture-ish helper: (hidden_states, pos, mask, token_channels)."""
    torch.manual_seed(1)
    batch_size, seq_len = 2, 5
    hidden_size, head_dim = 8, 4  # must match tiny_llama_config
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    pos = position_embeddings(batch_size, seq_len, head_dim)
    mask = causal_mask(batch_size, seq_len)
    token_channels = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 0, 1, 0]], dtype=torch.long)
    return hidden_states, pos, mask, token_channels


# ---------------------------------------------------------------------------
# Test 1 — equivalence (or divergence) with respect to base LlamaAttention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_gated_matches_base_when_gates_open(gating_variant):
    """Two scenarios where the gated module should be equivalent to base:

    a) token_channels=None           -> gating path is skipped -> bit-exact match.
    b) gates all open (init_logit=20) + channels -> allclose to base.
    """
    hidden_states, pos, mask, token_channels = _equivalence_inputs()

    # (a) token_channels=None -> gating path skipped -> bit-exact.
    base, gated, _ = _build_pair(gating_variant)
    base_out, base_weights = base(hidden_states, pos, mask)
    gated_out, gated_weights = gated(hidden_states, pos, mask, token_channels=None)
    assert torch.equal(base_out, gated_out)
    assert torch.equal(base_weights, gated_weights)

    # (b) init_logit=20 -> gates ≈ 1 on the gating path -> allclose to base.
    base, gated, _ = _build_pair(gating_variant, init_logit=20.0)
    base_out, base_weights = base(hidden_states, pos, mask)
    gated_out, gated_weights = gated(hidden_states, pos, mask, token_channels=token_channels)
    assert torch.allclose(gated_out, base_out, rtol=1e-5, atol=1e-6)
    assert torch.allclose(gated_weights, base_weights, rtol=1e-5, atol=1e-6)


def test_closing_one_qk_gate_diverges_on_attn_weights_and_output():
    hidden_states, pos, mask, token_channels = _equivalence_inputs()

    base, gated, _ = _build_pair("qk_gating", init_logit=20.0)
    with torch.no_grad():
        gated.head_gates.flatten()[0] = -20.0  # asymmetric closure

    base_out, base_weights = base(hidden_states, pos, mask)
    gated_out, gated_weights = gated(hidden_states, pos, mask, token_channels=token_channels)

    # The bias is added pre-softmax, so both weights and output diverge.
    assert not torch.allclose(gated_out, base_out, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(gated_weights, base_weights, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_token_channels_wrong_shape_raises(gating_variant):
    _, gated, config = _build_pair(gating_variant)
    batch_size, seq_len = 1, 4
    hidden_states = torch.zeros(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    bad_token_channels = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)

    with pytest.raises(ValueError, match="token_channels"):
        gated(hidden_states, pos, mask, token_channels=bad_token_channels)


# ---------------------------------------------------------------------------
# Direct tests of the gate helper
# ---------------------------------------------------------------------------


def test_apply_qk_gate():
    # H0: channel 0 can't read from channel 1.
    # H1: open.

    # head_gates indexed by (q_channel, k_channel, head).
    head_gates = torch.tensor(
        [
            # k_ch = 0        k_ch = 1
            [[10.0, 10.0], [-10.0, 10.0]],  # q_ch=0
            [[10.0, 10.0], [10.0, 10.0]],  # q_ch=1
        ]
    )  # [C=2, C=2, H=2]

    # Different per-batch channels so we exercise batched indexing.
    token_channels = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long)
    mask = causal_mask(batch_size=2, seq_len=3)

    combined = LlamaAttentionGated._apply_qk_gate(head_gates, mask, token_channels, token_channels)
    assert combined.shape == (2, 2, 3, 3)

    rtol, atol = 1e-4, 1e-4
    ninf = float("-inf")

    # Batch 0, token channels [0, 1, 0]. Head 0 blocks (q_ch=0 -> k_ch=1).
    # Only sub-diagonal blocked cell is (q=2, k=1). (q=0, k=1) is causally
    # masked already: -inf + (-10) stays -inf.
    assert torch.allclose(
        combined[0, 0],
        torch.tensor(
            [
                [0.0, ninf, ninf],
                [0.0, 0.0, ninf],
                [0.0, -10.0, 0.0],
            ]
        ),
        rtol=rtol,
        atol=atol,
    )
    # Batch 0, head 1 is open -> causal mask passes through.
    assert torch.allclose(combined[0, 1], mask[0, 0], rtol=rtol, atol=atol)

    # Batch 1, token channels [1, 0, 1]. Head 0 blocks (q_ch=0 -> k_ch=1) which
    # here is at (q=1, k=0) (q_ch=0, k_ch=1) and (q=1, k=2) — but (q=1, k=2) is
    # causally masked, so the only visible sub-diagonal block is (q=1, k=0).
    assert torch.allclose(
        combined[1, 0],
        torch.tensor(
            [
                [0.0, ninf, ninf],
                [-10.0, 0.0, ninf],
                [0.0, 0.0, 0.0],
            ]
        ),
        rtol=rtol,
        atol=atol,
    )
    # Batch 1, head 1 is open -> causal mask passes through.
    assert torch.allclose(combined[1, 1], mask[1, 0], rtol=rtol, atol=atol)


def test_apply_qk_gate_preserves_minus_inf_in_mask():
    # Causal -inf entries must stay -inf after adding the bias.
    head_gates = torch.zeros(2, 2, 2)  # logsigmoid(0) = log(0.5) ≈ -0.693
    token_channels = torch.tensor([[0, 1, 0]], dtype=torch.long)
    mask = causal_mask(1, 3)

    combined = LlamaAttentionGated._apply_qk_gate(head_gates, mask, token_channels, token_channels)

    # Broadcast mask -> [B, H, Q, K] and check every position that started
    # as -inf is still -inf (the upper triangle).
    broadcast_mask = mask.expand(1, 2, 3, 3)
    assert torch.equal(combined[broadcast_mask == float("-inf")], broadcast_mask[broadcast_mask == float("-inf")])


# ---------------------------------------------------------------------------
# Test 2 — qk_gating: zeroed gate -> zeroed attn_weights on matching pairs
# ---------------------------------------------------------------------------


def test_qk_gating_zeroes_attn_weights_on_blocked_pairs():
    base, gated, config = _build_pair("qk_gating", init_logit=20.0)

    # Close the (q_ch=0, k_ch=1, head=0) gate. Leave the rest ≈ 1.
    with torch.no_grad():
        gated.head_gates[0, 1, 0] = -20.0

    torch.manual_seed(3)
    batch_size, seq_len = 1, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    token_channels = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=token_channels)

    ch = token_channels[0]  # [S]
    q_is_0 = (ch == 0).view(seq_len, 1).expand(seq_len, seq_len)
    k_is_1 = (ch == 1).view(1, seq_len).expand(seq_len, seq_len)
    blocked = q_is_0 & k_is_1  # [Q, K]

    # (a) Blocked pairs on head 0 are ~0.
    assert torch.all(gated_weights[0, 0][blocked].abs() < 1e-6)

    # (b) Head != 0 is unchanged — all its gates are ≈ 1.
    assert torch.allclose(gated_weights[0, 1], base_weights[0, 1], rtol=1e-5, atol=1e-6)

    # (c) Head 0 rows with q_channel != 0 (untrusted query) are unchanged.
    q_is_1 = ch == 1
    assert torch.allclose(gated_weights[0, 0][q_is_1], base_weights[0, 0][q_is_1], rtol=1e-5, atol=1e-6)

    # (d) Head 0 rows with q_channel == 0 (trusted query) still sum to 1.
    row_sums = gated_weights[0, 0][q_is_1.logical_not()].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: train vs infer _apply_qk_gate implementations produce same output
# ---------------------------------------------------------------------------


def test_qk_gate_train_vs_infer_equivalence():
    """_apply_qk_gate with training=True (einsum) and training=False (indexing)
    must produce identical results."""
    torch.manual_seed(42)
    C, H = 3, 4
    B, Q, K = 2, 7, 12
    head_gates = torch.randn(C, C, H)
    mask = causal_mask(B, K)[:, :, -Q:, :]  # [B, 1, Q, K] (may be non-square during decode)
    q_channels = torch.randint(0, C, (B, Q))
    k_channels = torch.randint(0, C, (B, K))

    out_train = LlamaAttentionGated._apply_qk_gate(head_gates, mask, q_channels, k_channels, training=True)
    out_infer = LlamaAttentionGated._apply_qk_gate(head_gates, mask, q_channels, k_channels, training=False)

    assert out_train.shape == out_infer.shape
    assert torch.allclose(out_train, out_infer, rtol=1e-5, atol=1e-6), \
        f"max diff: {(out_train - out_infer).abs().max().item()}"


def test_qk_gate_train_vs_infer_single_token_decode():
    """Equivalence during autoregressive decode (Q=1, K=full context)."""
    torch.manual_seed(7)
    C, H = 2, 8
    B, K = 1, 20
    head_gates = torch.randn(C, C, H)
    mask = torch.zeros(B, 1, 1, K)
    q_channels = torch.randint(0, C, (B, 1))
    k_channels = torch.randint(0, C, (B, K))

    out_train = LlamaAttentionGated._apply_qk_gate(head_gates, mask, q_channels, k_channels, training=True)
    out_infer = LlamaAttentionGated._apply_qk_gate(head_gates, mask, q_channels, k_channels, training=False)

    assert torch.allclose(out_train, out_infer, rtol=1e-5, atol=1e-6), \
        f"max diff: {(out_train - out_infer).abs().max().item()}"


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_full_forward_train_vs_eval_match(gating_variant):
    """Full attention forward in train mode vs eval mode produces same output."""
    hidden_states, pos, mask, token_channels = _equivalence_inputs()
    _, gated, _ = _build_pair(gating_variant, init_logit=1.0)

    with torch.no_grad():
        gated.head_gates.normal_()  # randomize gates away from init

    gated.eval()
    with torch.no_grad():
        eval_out, eval_w = gated(hidden_states, pos, mask, token_channels=token_channels)

    gated.train()
    with torch.no_grad():
        train_out, train_w = gated(hidden_states, pos, mask, token_channels=token_channels)

    assert torch.allclose(eval_out, train_out, rtol=1e-5, atol=1e-6), \
        f"output max diff: {(eval_out - train_out).abs().max().item()}"
    assert torch.allclose(eval_w, train_w, rtol=1e-5, atol=1e-6), \
        f"weights max diff: {(eval_w - train_w).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test: static cache does not misalign channel assignments
# ---------------------------------------------------------------------------


def test_qk_gate_channels_aligned_with_static_cache():
    """When static cache makes K longer than token_channels, q_channels must
    stay aligned with the actual Q positions, not shift due to right-padding.

    Regression test for a bug where right-padding token_channels to match
    the static cache buffer, then slicing q_channels = tc[:, -new_len:],
    shifted the channel assignment by (k_total - tc_len) positions.
    """
    _, gated, config = _build_pair("qk_gating", init_logit=1.0)
    gated.eval()

    with torch.no_grad():
        gated.head_gates.normal_()

    batch_size, seq_len = 1, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    token_channels = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.long)

    # Run without cache (baseline)
    with torch.no_grad():
        out_base, w_base = gated(hidden_states, pos, mask, token_channels=token_channels)

    # Run with DynamicCache
    from transformers import DynamicCache
    cache_dyn = DynamicCache()
    cache_position = torch.arange(seq_len)
    with torch.no_grad():
        out_dyn, w_dyn = gated(
            hidden_states, pos, mask,
            token_channels=token_channels,
            past_key_values=cache_dyn,
            cache_position=cache_position,
        )

    # DynamicCache K has length seq_len (no padding), so it must match baseline exactly.
    assert torch.allclose(w_base, w_dyn, rtol=1e-5, atol=1e-6), \
        f"DynamicCache changed attention weights: max diff {(w_base - w_dyn).abs().max().item()}"
    assert torch.allclose(out_base, out_dyn, rtol=1e-5, atol=1e-6), \
        f"DynamicCache changed output: max diff {(out_base - out_dyn).abs().max().item()}"

    # Run with StaticCache — buffer is larger than seq_len
    from transformers import StaticCache
    cache_static = StaticCache(
        config=config, max_batch_size=batch_size,
        max_cache_len=seq_len + 4,  # 4 extra slots
        device="cpu", dtype=torch.float32,
    )
    # Static cache returns full buffer for K, making k_total > tc_len
    mask_static = torch.zeros(batch_size, 1, seq_len, seq_len + 4)
    mask_static[:, :, :, :seq_len] = mask
    mask_static[:, :, :, seq_len:] = float("-inf")

    with torch.no_grad():
        out_static, w_static = gated(
            hidden_states, pos, mask_static,
            token_channels=token_channels,
            past_key_values=cache_static,
            cache_position=cache_position,
        )

    # The attention weights for real positions should match
    # (static has extra K columns for unfilled cache, but those are masked)
    w_static_trimmed = w_static[:, :, :, :seq_len]
    assert torch.allclose(w_base, w_static_trimmed, rtol=1e-5, atol=1e-6), \
        f"Static cache changed attention weights: max diff {(w_base - w_static_trimmed).abs().max().item()}"
    assert torch.allclose(out_base, out_static, rtol=1e-5, atol=1e-6), \
        f"Static cache changed output: max diff {(out_base - out_static).abs().max().item()}"


def test_qk_gate_static_cache_decode_channels():
    """During decode with static cache, the single new token's q_channel
    must match its actual channel, not be shifted by cache padding."""
    _, gated, config = _build_pair("qk_gating", init_logit=1.0)
    gated.eval()

    # Close gate[0,1,h] (trusted Q attending to untrusted K) on all heads
    with torch.no_grad():
        gated.head_gates[0, 1, :] = -20.0

    batch_size, prefill_len = 1, 6
    token_channels = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.long)

    from transformers import StaticCache
    cache = StaticCache(
        config=config, max_batch_size=batch_size,
        max_cache_len=prefill_len + 4,
        device="cpu", dtype=torch.float32,
    )

    # Prefill
    hidden_prefill = torch.randn(batch_size, prefill_len, config.hidden_size)
    pos_prefill = position_embeddings(batch_size, prefill_len, config.head_dim)
    mask_prefill = torch.zeros(batch_size, 1, prefill_len, prefill_len + 4)
    mask_prefill[:, :, :, :prefill_len] = causal_mask(batch_size, prefill_len)
    mask_prefill[:, :, :, prefill_len:] = float("-inf")
    cache_pos_prefill = torch.arange(prefill_len)

    with torch.no_grad():
        gated(hidden_prefill, pos_prefill, mask_prefill,
              token_channels=token_channels,
              past_key_values=cache, cache_position=cache_pos_prefill)

    # Decode one token with channel 0 (trusted)
    hidden_decode = torch.randn(batch_size, 1, config.hidden_size)
    pos_decode = position_embeddings(batch_size, 1, config.head_dim)
    decode_channels = torch.cat([token_channels, torch.tensor([[0]])], dim=-1)
    # Mask: new token can attend to all prefill tokens + itself, not unfilled cache
    mask_decode = torch.zeros(batch_size, 1, 1, prefill_len + 4)
    mask_decode[:, :, :, :prefill_len + 1] = 0
    mask_decode[:, :, :, prefill_len + 1:] = float("-inf")
    cache_pos_decode = torch.tensor([prefill_len])

    with torch.no_grad():
        _, w_decode = gated(
            hidden_decode, pos_decode, mask_decode,
            token_channels=decode_channels,
            past_key_values=cache, cache_position=cache_pos_decode,
        )

    # The new token is trusted (ch=0). With gate[0,1,:] closed,
    # it should NOT attend to untrusted K positions (indices 2,3).
    # Check that attention weights at untrusted K positions are ~0.
    w = w_decode[0, :, 0, :]  # [H, K_total]
    untrusted_k = [2, 3]
    for k_idx in untrusted_k:
        assert w[:, k_idx].abs().max() < 1e-5, \
            f"Trusted Q attends to untrusted K[{k_idx}]: {w[:, k_idx].tolist()}"
