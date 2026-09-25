"""Unit tests for ``GptOssAttentionGated``.

Mirrors ``tests/test_gated_attention.py`` (the Llama suite). gpt-oss-specific
behaviour (attention sinks, sliding windows) lives in ``test_sinks.py`` and
``test_sliding_window.py``.

All tests run on CPU with tiny configs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.gpt_oss.modeling_gpt_oss import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from heads_of_security.attention_gating import GptOssAttentionGated
from heads_of_security.attention_gating.gpt_oss.gated_attention import gated_eager_attention_forward

from .utils import (
    FULL_LAYER_IDX,
    SLIDING_LAYER_IDX,
    attention_pair,
    causal_mask,
    equivalence_inputs,
    position_embeddings,
    sliding_causal_mask,
    tiny_gpt_oss_config,
)

# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_requires_eager_attention():
    config = tiny_gpt_oss_config()
    config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        GptOssAttentionGated(config, layer_idx=FULL_LAYER_IDX)


def test_rejects_non_qk_gating_variant():
    config = tiny_gpt_oss_config(gating_variant="not_a_variant")
    with pytest.raises(ValueError, match="qk_gating"):
        GptOssAttentionGated(config, layer_idx=FULL_LAYER_IDX)


def test_gate_shape_is_channels_channels_heads():
    config = tiny_gpt_oss_config(gating_num_channels=3)
    gated = GptOssAttentionGated(config, layer_idx=FULL_LAYER_IDX)
    assert gated.head_gates.shape == (3, 3, config.num_attention_heads)


def test_missing_attention_mask_raises():
    _, gated, config = attention_pair()
    hidden_states, pos, _, channels = equivalence_inputs(config)
    with pytest.raises(ValueError, match="attention_mask"):
        gated(hidden_states, pos, None, token_channels=channels)


# ---------------------------------------------------------------------------
# Test 1 — equivalence (or divergence) with respect to base GptOssAttention
# ---------------------------------------------------------------------------


def test_gated_matches_base_when_channels_none_bit_exact():
    """``token_channels=None`` skips the gating path entirely → bit-exact."""
    base, gated, config = attention_pair()
    hidden_states, pos, mask, _ = equivalence_inputs(config)

    base_out, base_weights = base(hidden_states, pos, mask)
    gated_out, gated_weights = gated(hidden_states, pos, mask, token_channels=None)

    assert torch.equal(base_out, gated_out)
    assert torch.equal(base_weights, gated_weights)


def test_gated_matches_base_when_gates_open():
    """init_logit=20 → logsigmoid ≈ -2e-9 → allclose to base.

    Note gpt-oss cannot be *bit*-exact here even with a uniform gate: the sink
    logit is not gated, so a uniform bias shifts a sliver of mass onto the sink.
    At init 20 that shift is ~1e-9. See ``test_sinks.py``.
    """
    base, gated, config = attention_pair(init_logit=20.0)
    hidden_states, pos, mask, channels = equivalence_inputs(config)

    base_out, base_weights = base(hidden_states, pos, mask)
    gated_out, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)

    assert torch.allclose(gated_out, base_out, rtol=1e-5, atol=1e-6)
    assert torch.allclose(gated_weights, base_weights, rtol=1e-5, atol=1e-6)


def test_closing_one_qk_gate_diverges_on_attn_weights_and_output():
    base, gated, config = attention_pair(init_logit=20.0)
    hidden_states, pos, mask, channels = equivalence_inputs(config)

    with torch.no_grad():
        gated.head_gates[0, 1, 0] = -20.0

    base_out, base_weights = base(hidden_states, pos, mask)
    gated_out, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)

    # The bias is pre-softmax, so both weights and output move.
    assert not torch.allclose(gated_out, base_out, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(gated_weights, base_weights, rtol=1e-5, atol=1e-6)


def test_token_channels_longer_than_keys_raises():
    _, gated, config = attention_pair()
    batch_size, seq_len = 1, 4
    hidden_states = torch.zeros(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    bad_channels = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)

    with pytest.raises(ValueError, match="token_channels"):
        gated(hidden_states, pos, mask, token_channels=bad_channels)


# ---------------------------------------------------------------------------
# Direct test of the gate helper — manual tensor comparison
# ---------------------------------------------------------------------------


def test_apply_qk_gate_manual_values():
    """Hand-computed expected mask, mirroring the Llama suite's version.

    H0: queries on channel 0 cannot read keys on channel 1. H1: fully open.
    """
    head_gates = torch.tensor(
        [
            # k_ch = 0        k_ch = 1
            [[10.0, 10.0], [-10.0, 10.0]],  # q_ch = 0
            [[10.0, 10.0], [10.0, 10.0]],  # q_ch = 1
        ]
    )  # [C=2, C=2, H=2]

    token_channels = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long)
    mask = causal_mask(batch_size=2, seq_len=3)

    combined = GptOssAttentionGated._apply_qk_gate(
        head_gates, mask, token_channels, token_channels
    )
    assert combined.shape == (2, 2, 3, 3)

    rtol = atol = 1e-4
    ninf = float("-inf")

    # Batch 0, channels [0, 1, 0]. Head 0 blocks (q_ch=0 -> k_ch=1); the only
    # such cell not already causally masked is (q=2, k=1).
    assert torch.allclose(
        combined[0, 0],
        torch.tensor([[0.0, ninf, ninf], [0.0, 0.0, ninf], [0.0, -10.0, 0.0]]),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(combined[0, 1], mask[0, 0], rtol=rtol, atol=atol)

    # Batch 1, channels [1, 0, 1]. The blocked, causally-visible cell is (q=1, k=0).
    assert torch.allclose(
        combined[1, 0],
        torch.tensor([[0.0, ninf, ninf], [-10.0, 0.0, ninf], [0.0, 0.0, 0.0]]),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(combined[1, 1], mask[1, 0], rtol=rtol, atol=atol)


def test_apply_qk_gate_preserves_minus_inf_in_mask():
    """Causal -inf entries must stay -inf after the bias is added."""
    head_gates = torch.zeros(2, 2, 2)  # logsigmoid(0) = log 0.5
    token_channels = torch.tensor([[0, 1, 0]], dtype=torch.long)
    mask = causal_mask(1, 3)

    combined = GptOssAttentionGated._apply_qk_gate(
        head_gates, mask, token_channels, token_channels
    )

    broadcast = mask.expand(1, 2, 3, 3)
    assert torch.equal(
        combined[broadcast == float("-inf")], broadcast[broadcast == float("-inf")]
    )


# ---------------------------------------------------------------------------
# Test 2 — closed gate zeroes the matching attention cells
# ---------------------------------------------------------------------------


def test_qk_gating_zeroes_attn_weights_on_blocked_pairs():
    """Mirrors the Llama test, with one gpt-oss difference.

    Assertion (d) in the Llama version — gated rows still sum to 1 — cannot hold
    here: gpt-oss drops the sink column after softmax, so rows sum to
    ``1 - p(sink)`` and the freed mass goes to the sink rather than to other keys.
    We assert the row sums *drop* instead, and ``test_sinks.py`` verifies that the
    missing mass is exactly what the sink gained.
    """
    base, gated, config = attention_pair(init_logit=20.0)

    with torch.no_grad():
        gated.head_gates[0, 1, 0] = -20.0  # (q_ch=0, k_ch=1, head 0)

    torch.manual_seed(3)
    batch_size, seq_len = 1, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    channels = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)

    ch = channels[0]
    q_is_0 = (ch == 0).view(seq_len, 1).expand(seq_len, seq_len)
    k_is_1 = (ch == 1).view(1, seq_len).expand(seq_len, seq_len)
    blocked = q_is_0 & k_is_1

    # (a) blocked cells on head 0 are ~0
    assert torch.all(gated_weights[0, 0][blocked].abs() < 1e-6)

    # (b) other heads untouched (their gates are still ≈ open)
    assert torch.allclose(gated_weights[0, 1], base_weights[0, 1], rtol=1e-5, atol=1e-6)

    # (c) head-0 rows whose query is on channel 1 are untouched
    q_is_1 = ch == 1
    assert torch.allclose(
        gated_weights[0, 0][q_is_1], base_weights[0, 0][q_is_1], rtol=1e-5, atol=1e-6
    )

    # (d) head-0 rows whose query is on channel 0 lost mass (to the sink), and
    #     never gained any.
    trusted_rows = q_is_1.logical_not()
    gated_sums = gated_weights[0, 0][trusted_rows].sum(dim=-1)
    base_sums = base_weights[0, 0][trusted_rows].sum(dim=-1)
    assert torch.all(gated_sums <= base_sums + 1e-6)
    # at least one row actually had a blocked key to lose
    assert torch.any(gated_sums < base_sums - 1e-6)


# ---------------------------------------------------------------------------
# train (einsum) vs infer (indexing) gate implementations
# ---------------------------------------------------------------------------


def test_qk_gate_train_vs_infer_equivalence():
    torch.manual_seed(42)
    num_channels, num_heads = 3, 4
    batch_size, q_len, k_len = 2, 7, 12
    head_gates = torch.randn(num_channels, num_channels, num_heads)
    mask = causal_mask(batch_size, k_len)[:, :, -q_len:, :]
    q_channels = torch.randint(0, num_channels, (batch_size, q_len))
    k_channels = torch.randint(0, num_channels, (batch_size, k_len))

    out_train = GptOssAttentionGated._apply_qk_gate(
        head_gates, mask, q_channels, k_channels, training=True
    )
    out_infer = GptOssAttentionGated._apply_qk_gate(
        head_gates, mask, q_channels, k_channels, training=False
    )

    assert out_train.shape == out_infer.shape
    assert torch.allclose(out_train, out_infer, rtol=1e-5, atol=1e-6)


def test_qk_gate_train_vs_infer_single_token_decode():
    torch.manual_seed(7)
    num_channels, num_heads = 2, 8
    batch_size, k_len = 1, 20
    head_gates = torch.randn(num_channels, num_channels, num_heads)
    mask = torch.zeros(batch_size, 1, 1, k_len)
    q_channels = torch.randint(0, num_channels, (batch_size, 1))
    k_channels = torch.randint(0, num_channels, (batch_size, k_len))

    out_train = GptOssAttentionGated._apply_qk_gate(
        head_gates, mask, q_channels, k_channels, training=True
    )
    out_infer = GptOssAttentionGated._apply_qk_gate(
        head_gates, mask, q_channels, k_channels, training=False
    )
    assert torch.allclose(out_train, out_infer, rtol=1e-5, atol=1e-6)


def test_full_forward_train_vs_eval_match():
    _, gated, config = attention_pair(init_logit=1.0)
    hidden_states, pos, mask, channels = equivalence_inputs(config)

    with torch.no_grad():
        gated.head_gates.normal_()

    gated.eval()
    with torch.no_grad():
        eval_out, eval_weights = gated(hidden_states, pos, mask, token_channels=channels)

    gated.train()
    with torch.no_grad():
        train_out, train_weights = gated(hidden_states, pos, mask, token_channels=channels)

    assert torch.allclose(eval_out, train_out, rtol=1e-5, atol=1e-6)
    assert torch.allclose(eval_weights, train_weights, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Cache alignment — channels must not shift
# ---------------------------------------------------------------------------


def test_qk_gate_channels_aligned_with_dynamic_cache():
    """A full-attention layer with DynamicCache must match the uncached forward."""
    from transformers import DynamicCache

    _, gated, config = attention_pair(init_logit=1.0, layer_idx=FULL_LAYER_IDX)
    gated.eval()
    with torch.no_grad():
        gated.head_gates.normal_()

    batch_size, seq_len = 1, 6
    torch.manual_seed(5)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    channels = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        out_base, weights_base = gated(hidden_states, pos, mask, token_channels=channels)

    cache = DynamicCache(config=config)
    with torch.no_grad():
        out_cached, weights_cached = gated(
            hidden_states,
            pos,
            mask,
            token_channels=channels,
            past_key_values=cache,
            cache_position=torch.arange(seq_len),
        )

    assert torch.allclose(weights_base, weights_cached, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_base, out_cached, rtol=1e-5, atol=1e-6)


def test_qk_gate_static_cache_decode_channels():
    """Decoding with a static cache: the new token's q_channel must be its own.

    With gate[T, U, :] shut, a trusted query must not attend to untrusted keys.
    """
    from transformers import StaticCache

    _, gated, config = attention_pair(init_logit=1.0, layer_idx=FULL_LAYER_IDX)
    gated.eval()
    with torch.no_grad():
        gated.head_gates[0, 1, :] = -20.0

    batch_size, prefill_len = 1, 6
    channels = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.long)
    cache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=prefill_len + 4,
        device="cpu",
        dtype=torch.float32,
    )

    torch.manual_seed(6)
    hidden_prefill = torch.randn(batch_size, prefill_len, config.hidden_size)
    pos_prefill = position_embeddings(batch_size, prefill_len, config.head_dim)
    mask_prefill = torch.zeros(batch_size, 1, prefill_len, prefill_len + 4)
    mask_prefill[:, :, :, :prefill_len] = causal_mask(batch_size, prefill_len)
    mask_prefill[:, :, :, prefill_len:] = float("-inf")

    with torch.no_grad():
        gated(
            hidden_prefill,
            pos_prefill,
            mask_prefill,
            token_channels=channels,
            past_key_values=cache,
            cache_position=torch.arange(prefill_len),
        )

    hidden_decode = torch.randn(batch_size, 1, config.hidden_size)
    pos_decode = position_embeddings(batch_size, 1, config.head_dim)
    decode_channels = torch.cat([channels, torch.tensor([[0]])], dim=-1)
    mask_decode = torch.zeros(batch_size, 1, 1, prefill_len + 4)
    mask_decode[:, :, :, prefill_len + 1 :] = float("-inf")

    with torch.no_grad():
        _, weights = gated(
            hidden_decode,
            pos_decode,
            mask_decode,
            token_channels=decode_channels,
            past_key_values=cache,
            cache_position=torch.tensor([prefill_len]),
        )

    per_head = weights[0, :, 0, :]  # [H, K_total]
    for untrusted_k in (2, 3):
        assert per_head[:, untrusted_k].abs().max() < 1e-5, (
            f"trusted query attended to untrusted key {untrusted_k}: "
            f"{per_head[:, untrusted_k].tolist()}"
        )


# ---------------------------------------------------------------------------
# Head-chunked inference route vs the [B, H, Q, K]-mask route
# ---------------------------------------------------------------------------
#
# Inference adds the log-bias inside attention, a chunk of heads at a time, instead
# of folding it into a per-head mask and calling HF's eager_attention_forward. The
# two must agree exactly -- the mask route is what the trained arms were validated
# against -- so these compare them directly.
#
# head_chunk is parametrized ON PURPOSE, including values that do not divide the
# head count. The tiny config has 4 heads and the shipped chunk is 8, so
# min(8, 4) = 4 and the production default runs the loop exactly ONCE: without
# these cases a chunk-boundary bug would pass the whole suite.


def _projected(gated, config, hidden_states, pos):
    """q/k/v exactly as ``forward`` builds them, so both routes see identical inputs."""
    batch, seq_len = hidden_states.shape[:2]
    shape = (batch, seq_len, -1, config.head_dim)
    query = gated.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = gated.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = gated.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos, sin = pos
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return query, key, value


def _mask_route(gated, query, key, value, mask, channels, seq_len):
    """The old path: bias folded into a [B, H, Q, K] mask, then HF eager attention."""
    gated_mask = gated._qk_gate_mask(mask, channels, seq_len, seq_len)
    return eager_attention_forward(
        gated, query, key, value, gated_mask,
        scaling=gated.scaling, dropout=0.0,
        sliding_window=gated.sliding_window, s_aux=gated.sinks,
    )


@pytest.mark.parametrize("head_chunk", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("layer_idx", [FULL_LAYER_IDX, SLIDING_LAYER_IDX])
def test_head_chunked_matches_mask_route(head_chunk, layer_idx):
    _, gated, config = attention_pair(init_logit=1.0, layer_idx=layer_idx)
    gated.eval()
    with torch.no_grad():
        gated.head_gates.normal_()

    batch_size, seq_len = 2, 9
    torch.manual_seed(3)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = (
        sliding_causal_mask(batch_size, seq_len, config.sliding_window)
        if layer_idx == SLIDING_LAYER_IDX
        else causal_mask(batch_size, seq_len)
    )
    channels = torch.randint(0, config.gating_num_channels, (batch_size, seq_len))
    query, key, value = _projected(gated, config, hidden_states, pos)

    want_out, want_weights = _mask_route(gated, query, key, value, mask, channels, seq_len)
    got_out, got_weights = gated_eager_attention_forward(
        gated, query, key, value, mask,
        scaling=gated.scaling, dropout=0.0,
        log_gates=F.logsigmoid(gated.head_gates),
        q_channels=channels, k_channels=channels,
        head_chunk=head_chunk,
    )

    assert got_out.shape == want_out.shape
    assert got_weights.shape == want_weights.shape
    # Not bit-exact across chunk sizes: slicing the head axis changes the matmul
    # shape, so BLAS picks a different blocking and reduces in a different order
    # (chunks 2 and 3 drift ~1e-7 on a 64-thread CPU; 1, 4 and 64 land bit-exact).
    # A bare torch.matmul reproduces it with no gating code involved. The gating is
    # a plain per-head loop that cannot depend on chunk size, so allow float32
    # rounding only -- a real discrepancy would be O(gate), not O(eps).
    torch.testing.assert_close(got_out, want_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_weights, want_weights, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("layer_idx", [FULL_LAYER_IDX, SLIDING_LAYER_IDX])
def test_forward_matches_mask_route(layer_idx):
    """End-to-end: ``forward`` in eval must equal the mask route through o_proj.

    Covers the wiring -- that ``forward`` hands the chunked path the same channels
    ``_qk_gate_mask`` would have used -- which the function-level test above cannot.
    """
    _, gated, config = attention_pair(init_logit=1.0, layer_idx=layer_idx)
    gated.eval()
    with torch.no_grad():
        gated.head_gates.normal_()

    batch_size, seq_len = 2, 9
    torch.manual_seed(4)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = (
        sliding_causal_mask(batch_size, seq_len, config.sliding_window)
        if layer_idx == SLIDING_LAYER_IDX
        else causal_mask(batch_size, seq_len)
    )
    channels = torch.randint(0, config.gating_num_channels, (batch_size, seq_len))

    with torch.no_grad():
        got_out, got_weights = gated(hidden_states, pos, mask, token_channels=channels)
        query, key, value = _projected(gated, config, hidden_states, pos)
        ref_out, want_weights = _mask_route(gated, query, key, value, mask, channels, seq_len)
        want_out = gated.o_proj(ref_out.reshape(batch_size, seq_len, -1).contiguous())

    torch.testing.assert_close(got_out, want_out, rtol=0, atol=0)
    torch.testing.assert_close(got_weights, want_weights, rtol=0, atol=0)


@pytest.mark.parametrize("head_chunk", [1, 3, 64])
def test_head_chunked_matches_mask_route_decode(head_chunk):
    """Q=1 decode step against the mask route.

    A separate case because the bias gather broadcasts differently here: q_idx is
    ``[B, 1, 1]`` against k_idx ``[B, 1, K]``, where prefill has ``[B, Q, 1]`` against
    ``[B, 1, K]``. Every other test in this section is a Q=K prefill.
    """
    _, gated, config = attention_pair(init_logit=1.0, layer_idx=FULL_LAYER_IDX)
    gated.eval()
    with torch.no_grad():
        gated.head_gates.normal_()

    batch_size, k_len = 2, 12
    num_heads = config.num_attention_heads
    num_kv, head_dim = config.num_key_value_heads, config.head_dim
    torch.manual_seed(6)
    query = torch.randn(batch_size, num_heads, 1, head_dim)
    key = torch.randn(batch_size, num_kv, k_len, head_dim)
    value = torch.randn(batch_size, num_kv, k_len, head_dim)
    mask = torch.zeros(batch_size, 1, 1, k_len)
    q_channels = torch.randint(0, config.gating_num_channels, (batch_size, 1))
    k_channels = torch.randint(0, config.gating_num_channels, (batch_size, k_len))

    gated_mask = gated._apply_qk_gate(
        gated.head_gates, mask, q_channels, k_channels, training=False
    )
    want_out, want_weights = eager_attention_forward(
        gated, query, key, value, gated_mask,
        scaling=gated.scaling, dropout=0.0,
        sliding_window=gated.sliding_window, s_aux=gated.sinks,
    )
    got_out, got_weights = gated_eager_attention_forward(
        gated, query, key, value, mask,
        scaling=gated.scaling, dropout=0.0,
        log_gates=F.logsigmoid(gated.head_gates),
        q_channels=q_channels, k_channels=k_channels,
        head_chunk=head_chunk,
    )

    torch.testing.assert_close(got_out, want_out, rtol=0, atol=0)
    torch.testing.assert_close(got_weights, want_weights, rtol=0, atol=0)
