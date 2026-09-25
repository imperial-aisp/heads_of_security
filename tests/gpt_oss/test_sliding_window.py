"""Gating on gpt-oss's sliding-window layers.

Half of gpt-oss's layers are sliding-window (128 keys in the released
checkpoints; 3 in the tiny test config so the window actually bites). The window
is expressed as ``finfo.min`` entries in the additive mask, so adding a negative
log-gate keeps those cells masked — the gate composes correctly, but it can only
*act* on pairs that are already in-window.

That asymmetry is the architectural caveat behind the model-selection doc: in a
sliding layer, an instruction/data pair farther apart than the window is not
gateable at all, because it is not attendable in the first place.

CPU, tiny configs.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from heads_of_security.attention_gating import GptOssGatedForCausalLM
from tests.utils import tiny_gpt_oss_config

from .utils import (
    FULL_LAYER_IDX,
    SLIDING_LAYER_IDX,
    attention_pair,
    causal_mask,
    position_embeddings,
    sliding_causal_mask,
)

WINDOW = 3


# ---------------------------------------------------------------------------
# Layer typing
# ---------------------------------------------------------------------------


def test_tiny_config_alternates_layer_types():
    config = tiny_gpt_oss_config(sliding_window=WINDOW)
    assert config.layer_types == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]


def test_sliding_window_set_only_on_sliding_layers():
    _, sliding, _ = attention_pair(layer_idx=SLIDING_LAYER_IDX, sliding_window=WINDOW)
    _, full, _ = attention_pair(layer_idx=FULL_LAYER_IDX, sliding_window=WINDOW)
    assert sliding.sliding_window == WINDOW
    assert full.sliding_window is None


# ---------------------------------------------------------------------------
# The gate cannot un-mask an out-of-window pair
# ---------------------------------------------------------------------------


def test_out_of_window_pairs_stay_masked_with_gates_open():
    _, gated, config = attention_pair(init_logit=20.0, layer_idx=SLIDING_LAYER_IDX)
    batch_size, seq_len = 1, 6

    torch.manual_seed(2)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = sliding_causal_mask(batch_size, seq_len, WINDOW)
    channels = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)

    with torch.no_grad():
        _, weights = gated(hidden_states, pos, mask, token_channels=channels)

    idx = torch.arange(seq_len)
    out_of_window = (idx.view(-1, 1) - idx.view(1, -1)) >= WINDOW
    assert weights[0, :, out_of_window].abs().max() < 1e-9


def test_closing_a_gate_cannot_revive_an_out_of_window_pair():
    """Opening or shutting the gate leaves out-of-window cells at exactly zero."""
    _, gated, config = attention_pair(init_logit=20.0, layer_idx=SLIDING_LAYER_IDX)
    batch_size, seq_len = 1, 6

    torch.manual_seed(2)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = sliding_causal_mask(batch_size, seq_len, WINDOW)
    channels = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)

    idx = torch.arange(seq_len)
    out_of_window = (idx.view(-1, 1) - idx.view(1, -1)) >= WINDOW

    with torch.no_grad():
        _, open_weights = gated(hidden_states, pos, mask, token_channels=channels)
        gated.head_gates[0, 1, :] = -30.0
        _, shut_weights = gated(hidden_states, pos, mask, token_channels=channels)

    assert open_weights[0, :, out_of_window].abs().max() < 1e-9
    assert shut_weights[0, :, out_of_window].abs().max() < 1e-9


def test_gate_still_acts_within_the_window():
    _, gated, config = attention_pair(init_logit=20.0, layer_idx=SLIDING_LAYER_IDX)
    batch_size, seq_len = 1, 6

    torch.manual_seed(2)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos = position_embeddings(batch_size, seq_len, config.head_dim)
    mask = sliding_causal_mask(batch_size, seq_len, WINDOW)
    # key 3 is untrusted and within the window of query 4
    channels = torch.tensor([[0, 0, 0, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        _, before = gated(hidden_states, pos, mask, token_channels=channels)
        gated.head_gates[0, 1, :] = -30.0
        _, after = gated(hidden_states, pos, mask, token_channels=channels)

    assert before[0, :, 4, 3].abs().min() > 1e-6, "in-window pair should start visible"
    assert after[0, :, 4, 3].abs().max() < 1e-6, "gate should shut the in-window pair"


def test_full_layer_gates_a_pair_the_sliding_layer_cannot():
    """The architectural caveat, stated as a test.

    Query 5 / key 1 are 4 apart — outside a 3-key window. A full-attention layer
    gates that pair; a sliding layer cannot, because it is already masked.
    """
    batch_size, seq_len = 1, 6
    torch.manual_seed(2)
    channels = torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.long)  # key 1 untrusted
    q, k = 5, 1

    results = {}
    for label, layer_idx, mask_fn in (
        ("full", FULL_LAYER_IDX, causal_mask),
        ("sliding", SLIDING_LAYER_IDX, lambda b, s: sliding_causal_mask(b, s, WINDOW)),
    ):
        _, gated, config = attention_pair(init_logit=20.0, layer_idx=layer_idx)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        pos = position_embeddings(batch_size, seq_len, config.head_dim)
        mask = mask_fn(batch_size, seq_len)
        with torch.no_grad():
            _, before = gated(hidden_states, pos, mask, token_channels=channels)
            gated.head_gates[0, 1, :] = -30.0
            _, after = gated(hidden_states, pos, mask, token_channels=channels)
        results[label] = (before[0, :, q, k].clone(), after[0, :, q, k].clone())

    full_before, full_after = results["full"]
    sliding_before, sliding_after = results["sliding"]

    # Full attention: the pair was visible and the gate shut it.
    assert full_before.abs().min() > 1e-6
    assert full_after.abs().max() < 1e-6
    # Sliding: never visible, so the gate had nothing to do.
    assert sliding_before.abs().max() < 1e-9
    assert sliding_after.abs().max() < 1e-9


# ---------------------------------------------------------------------------
# Channel alignment when the sliding cache evicts old keys
# ---------------------------------------------------------------------------


def test_decode_matches_full_forward_with_sliding_cache():
    """The sliding cache holds only the last ``window`` keys.

    ``token_channels`` still covers the whole sequence, so the gated attention
    tail-slices it to match the retained keys. If that slice were misaligned the
    per-step logits would diverge from the uncached forward.
    """
    torch.manual_seed(0)
    config = tiny_gpt_oss_config(init_logit=4.0, sliding_window=WINDOW)
    model = GptOssGatedForCausalLM(config).eval()
    with torch.no_grad():
        # A shut T->U gate makes the channel assignment observable in the logits.
        for layer in model.model.layers:
            layer.self_attn.head_gates[0, 1, :] = -6.0

    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (1, 7))
    channels = torch.tensor([[0, 0, 1, 1, 0, 1, 0]], dtype=torch.long)

    with torch.no_grad():
        full = model(input_ids=input_ids, token_channels=channels).logits

    cache = DynamicCache(config=config)
    for t in range(input_ids.shape[1]):
        with torch.no_grad():
            step = model(
                input_ids=input_ids[:, t : t + 1],
                token_channels=channels[:, : t + 1],
                past_key_values=cache,
                use_cache=True,
            ).logits
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-4, rtol=1e-4), (
            f"sliding-cache decode diverged at step {t}: "
            f"max diff {(step[:, 0] - full[:, t]).abs().max().item()}"
        )


def test_sliding_cache_is_shorter_than_channel_sequence():
    """Guards the premise of the tail slice (and documents the eviction)."""
    torch.manual_seed(0)
    config = tiny_gpt_oss_config(sliding_window=WINDOW)
    model = GptOssGatedForCausalLM(config).eval()

    seen: dict[int, int] = {}

    def hook(module, args, kwargs, output):
        seen[module.layer_idx] = kwargs["past_key_values"].layers[module.layer_idx].keys.shape[-2]

    handles = [
        layer.self_attn.register_forward_hook(hook, with_kwargs=True)
        for layer in model.model.layers
    ]

    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    channels = torch.zeros_like(input_ids)
    cache = DynamicCache(config=config)
    try:
        with torch.no_grad():
            for t in range(input_ids.shape[1]):
                model(
                    input_ids=input_ids[:, t : t + 1],
                    token_channels=channels[:, : t + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    assert seen[SLIDING_LAYER_IDX] <= WINDOW, "sliding layer should cap its cache"
    assert seen[FULL_LAYER_IDX] == input_ids.shape[1], "full layer should keep everything"
