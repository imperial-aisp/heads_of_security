"""KV-cache and ``.generate()`` tests for ``GptOssGatedForCausalLM``.

Mirrors ``tests/llama/test_gated_kv_cache.py``. Ground truth throughout is the
non-cached full forward (validated in ``test_gated_attention.py`` /
``test_gated_model.py``).

Note gpt-oss's sliding layers evict cache entries, so these tests exercise the
channel tail-slice in ``GptOssAttentionGated.forward`` as well; the dedicated
coverage for that lives in ``test_sliding_window.py``.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from heads_of_security.attention_gating import GptOssGatedForCausalLM
from tests.utils import tiny_gpt_oss_config


def _build_model_with_one_closed_gate(generated_channel: int = 0):
    torch.manual_seed(0)
    config = tiny_gpt_oss_config(init_logit=4.0, generated_channel=generated_channel)
    model = GptOssGatedForCausalLM(config).eval()
    with torch.no_grad():
        model.model.layers[0].self_attn.head_gates.flatten()[0] = -4.0
    return model, config


def _random_prompt(config, batch_size: int = 2, seq_len: int = 6):
    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    patterns = [[0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1]]
    channels = torch.tensor(patterns[:batch_size], dtype=torch.long)[:, :seq_len]
    return input_ids, channels


# ---------------------------------------------------------------------------
# Test 1 — step-by-step decode matches the non-cached full forward
# ---------------------------------------------------------------------------


def test_decode_step_by_step_matches_full_forward():
    model, config = _build_model_with_one_closed_gate()
    input_ids, channels = _random_prompt(config)
    batch_size, seq_len = input_ids.shape

    full = model(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        token_channels=channels,
    )

    cache = DynamicCache(config=config)
    for t in range(seq_len):
        step = model(
            input_ids=input_ids[:, t : t + 1],
            attention_mask=torch.ones(batch_size, t + 1, dtype=torch.long),
            token_channels=channels[:, : t + 1],
            past_key_values=cache,
            use_cache=True,
        )
        assert step.logits.shape == (batch_size, 1, config.vocab_size)
        assert torch.allclose(step.logits[:, 0], full.logits[:, t], rtol=1e-4, atol=1e-5), (
            f"decode step {t} diverged: "
            f"{(step.logits[:, 0] - full.logits[:, t]).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test 2 — prefill then decode matches the non-cached full forward
# ---------------------------------------------------------------------------


def test_prefill_then_decode_matches_full_forward():
    model, config = _build_model_with_one_closed_gate()
    input_ids, channels = _random_prompt(config)
    batch_size, seq_len = input_ids.shape
    prefill_len = 3

    full = model(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        token_channels=channels,
    )

    cache = DynamicCache(config=config)
    prefill = model(
        input_ids=input_ids[:, :prefill_len],
        attention_mask=torch.ones(batch_size, prefill_len, dtype=torch.long),
        token_channels=channels[:, :prefill_len],
        past_key_values=cache,
        use_cache=True,
    )
    assert torch.allclose(prefill.logits, full.logits[:, :prefill_len], rtol=1e-4, atol=1e-5)

    for t in range(prefill_len, seq_len):
        step = model(
            input_ids=input_ids[:, t : t + 1],
            attention_mask=torch.ones(batch_size, t + 1, dtype=torch.long),
            token_channels=channels[:, : t + 1],
            past_key_values=cache,
            use_cache=True,
        )
        assert torch.allclose(step.logits[:, 0], full.logits[:, t], rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3 — cache + token_channels=None matches the base (non-gated) LM
# ---------------------------------------------------------------------------


def test_decode_with_channels_none_matches_base_lm():
    torch.manual_seed(0)
    config = tiny_gpt_oss_config(init_logit=4.0)
    gated = GptOssGatedForCausalLM(config).eval()
    base = GptOssForCausalLM(config).eval()
    sd = {k: v for k, v in gated.state_dict().items() if not k.endswith("head_gates")}
    missing, unexpected = base.load_state_dict(sd, strict=False)
    assert unexpected == [], f"unexpected keys: {unexpected}"

    input_ids, _ = _random_prompt(config)
    batch_size, seq_len = input_ids.shape
    prefill_len = 3

    base_cache = DynamicCache(config=config)
    gated_cache = DynamicCache(config=config)
    spans = [(0, prefill_len), *[(t, t + 1) for t in range(prefill_len, seq_len)]]
    for t_start, t_end in spans:
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
# Test 4 — .generate() greedy matches manual greedy over the full forward
# ---------------------------------------------------------------------------


def test_generate_greedy_matches_manual_full_forward():
    model, config = _build_model_with_one_closed_gate(generated_channel=0)

    torch.manual_seed(1)
    batch_size, prompt_len, num_new = 2, 4, 3
    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    prompt_channels = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)

    ids = prompt_ids.clone()
    channels = prompt_channels.clone()
    for _ in range(num_new):
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            token_channels=channels,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=-1)
        channels = torch.cat(
            [channels, torch.full((batch_size, 1), config.gating_generated_channel, dtype=torch.long)],
            dim=-1,
        )

    generated = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        token_channels=prompt_channels,
        max_new_tokens=num_new,
        do_sample=False,
        num_beams=1,
    )

    assert torch.equal(generated, ids)


# ---------------------------------------------------------------------------
# Test 5 — gating_generated_channel actually reaches generate()
# ---------------------------------------------------------------------------


def test_generated_channel_config_affects_output():
    """Two models, identical weights, different ``gating_generated_channel``.

    Step 0's logits come from the prompt alone, so they must match bit-exactly;
    later steps involve the newly generated token as a query, whose channel *is*
    ``gating_generated_channel``, so at least one must differ.
    """
    model_ch0, config = _build_model_with_one_closed_gate(generated_channel=0)
    model_ch1, _ = _build_model_with_one_closed_gate(generated_channel=1)

    torch.manual_seed(1)
    batch_size, prompt_len, num_new = 2, 4, 4
    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
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

    assert torch.equal(out_ch0.scores[0], out_ch1.scores[0])
    assert any(
        not torch.equal(s0, s1) for s0, s1 in zip(out_ch0.scores[1:], out_ch1.scores[1:])
    )
