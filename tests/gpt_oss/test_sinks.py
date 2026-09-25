"""How the channel gate interacts with gpt-oss's attention sinks.

``eager_attention_forward`` adds the mask, concatenates a learned per-head sink
logit, softmaxes over ``K + 1``, then drops the sink column. So the returned
attention rows sum to ``1 - p(sink)``, and ``p(sink)`` is recoverable as
``1 - row.sum()``.

The gate biases content logits only. That yields an exactly characterisable
behaviour, which is what this module pins down:

1. ratios *among* surviving keys are unchanged (plain softmax property);
2. the sink-to-surviving-key ratio is also unchanged, because the sink logit is
   untouched;
3. the row sum *drops* — the sink keeps a share of the freed mass;
4. surviving keys still rise, just less than a sink-free model would give them.

CPU, tiny configs.
"""

from __future__ import annotations

import math

import torch

from .utils import attention_pair, causal_mask, equivalence_inputs, position_embeddings


def _sink_prob(attn_weights: torch.Tensor) -> torch.Tensor:
    """``p(sink)`` per row, i.e. the mass dropped with the sink column."""
    return 1.0 - attn_weights.sum(dim=-1)


# ---------------------------------------------------------------------------
# Baseline sink behaviour
# ---------------------------------------------------------------------------


def test_base_rows_do_not_sum_to_one():
    """Sanity check on the premise: gpt-oss keeps mass on the dropped sink."""
    base, _, config = attention_pair()
    hidden_states, pos, mask, _ = equivalence_inputs(config)

    _, weights = base(hidden_states, pos, mask)
    sink = _sink_prob(weights)

    assert torch.all(sink > 0), "expected the sink to hold some mass"
    assert torch.all(sink < 1)


def test_channels_none_leaves_sink_mass_bit_exact():
    base, gated, config = attention_pair(init_logit=4.0)
    hidden_states, pos, mask, _ = equivalence_inputs(config)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=None)

    assert torch.equal(_sink_prob(base_weights), _sink_prob(gated_weights))


# ---------------------------------------------------------------------------
# A uniform gate is not exactly the identity
# ---------------------------------------------------------------------------


def test_uniform_gate_shifts_mass_to_sink():
    """At init 4.0 every pair gets the same bias, yet the sink still gains.

    The shift is small but strictly positive, and content mass strictly decreases.
    """
    base, gated, config = attention_pair(init_logit=4.0)
    hidden_states, pos, mask, channels = equivalence_inputs(config)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)

    base_sink = _sink_prob(base_weights)
    gated_sink = _sink_prob(gated_weights)

    assert torch.all(gated_sink > base_sink - 1e-9)
    assert (gated_sink - base_sink).max() > 0, "uniform gate should move some mass"
    # Content mass shrank by exactly what the sink gained.
    assert torch.allclose(
        base_weights.sum(-1) - gated_weights.sum(-1), gated_sink - base_sink, atol=1e-6
    )


def test_uniform_gate_shift_matches_closed_form():
    """The sink's odds against content scale by exp(-bias); check that exactly.

    With a uniform bias ``b`` on every content logit, ``p_sink / p_content`` is
    multiplied by ``exp(-b)``. At init 4.0, ``b = logsigmoid(4.0)``.
    """
    base, gated, config = attention_pair(init_logit=4.0)
    hidden_states, pos, mask, channels = equivalence_inputs(config)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)

    bias = math.log(torch.sigmoid(torch.tensor(4.0)).item())
    expected_ratio = math.exp(-bias)

    base_odds = _sink_prob(base_weights) / base_weights.sum(-1)
    gated_odds = _sink_prob(gated_weights) / gated_weights.sum(-1)

    assert torch.allclose(
        gated_odds, base_odds * expected_ratio, rtol=1e-4, atol=1e-6
    ), f"expected sink odds to scale by {expected_ratio}"


def test_large_init_logit_recovers_base_closely():
    """init 20 → bias ≈ -2e-9, so the sink shift is numerically negligible."""
    base, gated, config = attention_pair(init_logit=20.0)
    hidden_states, pos, mask, channels = equivalence_inputs(config)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)

    assert (_sink_prob(gated_weights) - _sink_prob(base_weights)).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# Closing a gate: where does the mass go?
# ---------------------------------------------------------------------------


def _blocked_row_setup():
    """One head with (q_ch=0 -> k_ch=1) shut; returns base/gated weights + channels."""
    base, gated, config = attention_pair(init_logit=20.0)
    with torch.no_grad():
        gated.head_gates[0, 1, 0] = -30.0

    torch.manual_seed(4)
    batch_size, seq_len = 1, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = causal_mask(batch_size, seq_len)
    channels = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)

    _, base_weights = base(hidden_states, pos, mask)
    _, gated_weights = gated(hidden_states, pos, mask, token_channels=channels)
    return base_weights[0, 0], gated_weights[0, 0], channels[0], seq_len


def test_closed_gate_preserves_ratios_among_surviving_keys():
    base_w, gated_w, ch, seq_len = _blocked_row_setup()

    # Query 4 is on channel 0 and can see keys 0..4; keys 1 and 3 are channel 1
    # (blocked), keys 0, 2, 4 survive.
    q = 4
    survivors = [0, 2, 4]
    base_row, gated_row = base_w[q], gated_w[q]

    for a, b in zip(survivors, survivors[1:]):
        base_ratio = base_row[a] / base_row[b]
        gated_ratio = gated_row[a] / gated_row[b]
        assert torch.allclose(base_ratio, gated_ratio, rtol=1e-4), (
            "softmax must preserve ratios among unblocked keys"
        )


def test_closed_gate_preserves_sink_to_survivor_ratio():
    """The sink logit is ungated, so its odds against a survivor are unchanged."""
    base_w, gated_w, _, _ = _blocked_row_setup()
    q, survivor = 4, 2

    base_ratio = (1.0 - base_w[q].sum()) / base_w[q][survivor]
    gated_ratio = (1.0 - gated_w[q].sum()) / gated_w[q][survivor]

    assert torch.allclose(base_ratio, gated_ratio, rtol=1e-4)


def test_closed_gate_drops_row_sum_and_raises_survivors():
    """The head reads less overall, yet surviving keys individually gain."""
    base_w, gated_w, _, _ = _blocked_row_setup()
    q, blocked, survivors = 4, [1, 3], [0, 2, 4]

    assert gated_w[q].sum() < base_w[q].sum() - 1e-6, "row sum should drop"
    assert gated_w[q][blocked].abs().max() < 1e-6, "blocked keys should be ~0"
    for k in survivors:
        assert gated_w[q][k] > base_w[q][k], (
            "surviving keys are renormalised upward, just not fully"
        )


def test_closed_gate_gives_survivors_less_than_sink_free_renormalisation():
    """Quantifies 'less than a sink-free model': compare to full renormalisation.

    Without a sink, removing the blocked keys would rescale survivors by
    ``1 / (1 - p_blocked)``. With one, the rescale factor is smaller because the
    sink takes a share too.
    """
    base_w, gated_w, _, _ = _blocked_row_setup()
    q, blocked, survivor = 4, [1, 3], 2

    p_blocked = base_w[q][blocked].sum()
    sink_free = base_w[q][survivor] / (1.0 - p_blocked)

    assert gated_w[q][survivor] < sink_free, (
        "with an ungated sink, survivors must gain strictly less than full "
        "sink-free renormalisation"
    )


# ---------------------------------------------------------------------------
# The sink parameter itself is never trained
# ---------------------------------------------------------------------------


def test_freeze_base_leaves_sinks_frozen():
    from heads_of_security.attention_gating import GptOssGatedForCausalLM
    from tests.utils import tiny_gpt_oss_config

    torch.manual_seed(0)
    model = GptOssGatedForCausalLM(tiny_gpt_oss_config()).eval()
    model.freeze_base()

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "expected the gates to be trainable"
    assert all(name.endswith("head_gates") for name in trainable)
    assert not any("sinks" in name for name in trainable), (
        "sinks must stay frozen — they are base-model parameters, not gates"
    )
