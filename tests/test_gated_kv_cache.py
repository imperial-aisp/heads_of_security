"""KV-cache and `.generate()` tests for `LlamaGatedForCausalLM`.

Ground truth throughout is the non-cached full forward (already
validated in ``test_gated_attention.py`` and ``test_gated_model.py``).
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from heads_of_security.attention_gating import LlamaGatedForCausalLM

from .utils import copy_base_to_gated, tiny_llama_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model_with_one_closed_gate(gating_variant: str, generated_channel: int = 0):
    torch.manual_seed(0)
    cfg = tiny_llama_config(
        gating_variant=gating_variant,
        init_logit=4.0,
        generated_channel=generated_channel,
    )
    model = LlamaGatedForCausalLM(cfg).eval()
    # Close one gate on layer 0 so the gating path affects logits.
    with torch.no_grad():
        model.model.layers[0].self_attn.head_gates.flatten()[0] = -4.0
    return model, cfg


def _random_prompt(cfg, batch_size: int = 2, seq_len: int = 6):
    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    # Fixed channels that exercise both values in the {0, 1} alphabet.
    patterns = [[0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1]]
    token_channels = torch.tensor(patterns[:batch_size], dtype=torch.long)[:, :seq_len]
    return input_ids, token_channels


# ---------------------------------------------------------------------------
# Test 1 — step-by-step decode (no prefill) matches non-cached full forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_decode_step_by_step_matches_full_forward(gating_variant):
    model, cfg = _build_model_with_one_closed_gate(gating_variant)
    input_ids, token_channels = _random_prompt(cfg)
    batch_size, seq_len = input_ids.shape

    full = model(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        token_channels=token_channels,
    )

    cache = DynamicCache()
    for t in range(seq_len):
        step = model(
            input_ids=input_ids[:, t : t + 1],
            attention_mask=torch.ones(batch_size, t + 1, dtype=torch.long),
            token_channels=token_channels[:, : t + 1],
            past_key_values=cache,
            use_cache=True,
        )
        assert step.logits.shape == (batch_size, 1, cfg.vocab_size)
        assert torch.allclose(step.logits[:, 0], full.logits[:, t], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2 — prefill then step decode matches non-cached full forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_prefill_then_decode_matches_full_forward(gating_variant):
    model, cfg = _build_model_with_one_closed_gate(gating_variant)
    input_ids, token_channels = _random_prompt(cfg)
    batch_size, seq_len = input_ids.shape
    prefill_len = 3

    full = model(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        token_channels=token_channels,
    )

    cache = DynamicCache()
    prefill = model(
        input_ids=input_ids[:, :prefill_len],
        attention_mask=torch.ones(batch_size, prefill_len, dtype=torch.long),
        token_channels=token_channels[:, :prefill_len],
        past_key_values=cache,
        use_cache=True,
    )
    assert torch.allclose(prefill.logits, full.logits[:, :prefill_len], rtol=1e-5, atol=1e-6)

    for t in range(prefill_len, seq_len):
        step = model(
            input_ids=input_ids[:, t : t + 1],
            attention_mask=torch.ones(batch_size, t + 1, dtype=torch.long),
            token_channels=token_channels[:, : t + 1],
            past_key_values=cache,
            use_cache=True,
        )
        assert torch.allclose(step.logits[:, 0], full.logits[:, t], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3 — cache + token_channels=None matches base (no-gating) LM
# ---------------------------------------------------------------------------


def test_decode_with_channels_none_matches_base_lm():
    torch.manual_seed(0)
    base_cfg = tiny_llama_config()
    base = LlamaForCausalLM(base_cfg).eval()
    gated_cfg = tiny_llama_config(gating_variant="qk_gating", init_logit=4.0)
    gated = LlamaGatedForCausalLM(gated_cfg).eval()
    copy_base_to_gated(base, gated)

    input_ids, _ = _random_prompt(base_cfg)
    batch_size, seq_len = input_ids.shape
    prefill_len = 3

    base_cache = DynamicCache()
    gated_cache = DynamicCache()
    for t_start, t_end in [(0, prefill_len), *[(t, t + 1) for t in range(prefill_len, seq_len)]]:
        mask = torch.ones(batch_size, t_end, dtype=torch.long)
        base_out = base(
            input_ids=input_ids[:, t_start:t_end],
            attention_mask=mask,
            past_key_values=base_cache,
            use_cache=True,
        )
        gated_out = gated(
            input_ids=input_ids[:, t_start:t_end],
            attention_mask=mask,
            token_channels=None,
            past_key_values=gated_cache,
            use_cache=True,
        )
        assert torch.equal(gated_out.logits, base_out.logits)


# ---------------------------------------------------------------------------
# Test 4 — .generate() greedy matches manual greedy over full forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_generate_greedy_matches_manual_full_forward(gating_variant):
    model, cfg = _build_model_with_one_closed_gate(gating_variant, generated_channel=0)

    torch.manual_seed(1)
    batch_size, prompt_len = 2, 4
    num_new = 3
    prompt_ids = torch.randint(0, cfg.vocab_size, (batch_size, prompt_len))
    prompt_channels = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)

    # Manual greedy over non-cached full forward (trusted ground truth).
    ids = prompt_ids.clone()
    channels = prompt_channels.clone()
    for _ in range(num_new):
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            token_channels=channels,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=-1)
        new_channel = torch.full(
            (batch_size, 1), cfg.gating_generated_channel, dtype=torch.long
        )
        channels = torch.cat([channels, new_channel], dim=-1)

    # Native greedy .generate() with cache.
    gen = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        token_channels=prompt_channels,
        max_new_tokens=num_new,
        do_sample=False,
        num_beams=1,
    )

    assert torch.equal(gen, ids)


# ---------------------------------------------------------------------------
# Test 5 — gating_generated_channel config actually affects generate output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gating_variant", ["qk_gating"])
def test_generated_channel_config_affects_output(gating_variant):
    """Build two models with identical weights but different
    `gating_generated_channel`; some generated-step logits must differ.
    Proves the config wires into the running token_channels during generate.

    Compares per-step logits (via `output_scores`) rather than token IDs —
    argmax can coincide on the tiny test model even when the underlying
    logit distributions differ."""
    model_ch0, cfg = _build_model_with_one_closed_gate(gating_variant, generated_channel=0)
    model_ch1, _ = _build_model_with_one_closed_gate(gating_variant, generated_channel=1)
    # _build_model_with_one_closed_gate uses manual_seed(0) before init, so
    # weight init is identical across the two instances.

    torch.manual_seed(1)
    batch_size, prompt_len = 2, 4
    num_new = 4
    prompt_ids = torch.randint(0, cfg.vocab_size, (batch_size, prompt_len))
    prompt_channels = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)

    gen_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        token_channels=prompt_channels,
        max_new_tokens=num_new,
        do_sample=False,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )
    out_ch0 = model_ch0.generate(**gen_kwargs)
    out_ch1 = model_ch1.generate(**gen_kwargs)

    # Step 0's logits are computed from the prompt only — generated_channel
    # hasn't been consulted yet, so they must match bit-exactly.
    assert torch.equal(out_ch0.scores[0], out_ch1.scores[0])

    # Steps >= 1 involve the newly generated token as a query, whose channel
    # IS generated_channel. Logits must differ on at least one such step.
    assert any(
        not torch.equal(s0, s1) for s0, s1 in zip(out_ch0.scores[1:], out_ch1.scores[1:])
    )
