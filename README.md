# Heads of Security

Trained per-head **trust-channel gates** on attention, as a defense against prompt injection
in tool-using LLM agents. This repository contains the inference implementation for
`openai/gpt-oss-20b` and `meta-llama/Llama-3.3-70B-Instruct` in two backends, HuggingFace
transformers (eager) and vLLM, together with the trained gate tensors used in the paper and
demos showing how to load and run them.

## The method in one paragraph

Every token of a prompt is assigned a *channel*: 0 = **trusted** (system prompt, user turns,
the assistant's own outputs), 1 = **untrusted** (tool results), 2 = **scaffold** (chat-template
tokens). Each attention head `h` in each layer gets a tiny table of logits
`head_gates[q, k, h]` over (query channel, key channel) pairs. At inference,
`logsigmoid(head_gates[q, k, h])` is added to the pre-softmax attention score of every
(query of channel `q`, key of channel `k`) pair, so a head can learn to attend *less* to
untrusted tokens when serving a trusted query while leaving everything else untouched. The
base model is frozen; only the gates (18k parameters on gpt-oss-20b, 46k on Llama-70B) are
trained, by DPO on preference pairs where the rejected response follows an injected
instruction. A uniform table is an exact identity to the base model, so the init is lossless.
gpt-oss additionally gets a `sink_gates[q, h]` bias on its attention-sink logit; without it,
"attend less to untrusted" can only be expressed as "attend less to everything".

Channels are **out of band**: they are not tokens, they ride alongside `input_ids` (as a
`token_channels` tensor in HF, as a per-request side-band in vLLM), and they are derived from
message roles by the serving layer, so a prompt cannot forge its own trust level.

## Repository layout

```
src/heads_of_security/
  attention_gating/          HuggingFace (eager) implementation
    gating_core.py             gate parameters, the gate operator, the from_gates loader
    gated_attention.py         LlamaAttentionGated
    gated_model.py             LlamaGatedConfig / LlamaGatedForCausalLM
    gpt_oss/                   GptOssAttentionGated, GptOssGatedForCausalLM, Harmony channel tokenizer
    channel_tokenizer.py       role -> channel policy, chat walk, sidecar config (shared)
    gated_tokenizer.py         LlamaGatedTokenizer (Llama-3 chat scaffold)
    model_utils.py             load_gated_model(gates, base_model) -> (model, tokenizer)
  vllm_gates/                vLLM (paged attention) implementation, pinned to vllm 0.27.x
    kernel.py                  gated build of vLLM's unified Triton attention kernel
    backend.py                 Triton attention backend that feeds the kernel its channels
    model.py / model_llama.py  GptOssGatedForCausalLM / LlamaGatedForCausalLM for vLLM
    channels.py, runner_patch.py   the per-step channel side-band
    build_snapshot.py          base weights + gates -> loadable snapshot dir
    server.py                  OpenAI-compatible server, channels derived from roles
    offline.py                 GatedLLM: offline API taking (tokens, channels)
  serve/                     Harmony rendering / tool-call parsing helpers used by the server
gates/                       the five trained gate files (see gates/README.md)
demos/                       inspect_gates.py, hf_generate.py, build_and_serve_vllm.sh, load_gates.ipynb
tests/                       CPU unit tests on tiny configs
```

## Install

Python 3.12+. The pins matter: the HF path threads `token_channels` through
`transformers==5.9.0`'s attention modules, and the vLLM path patches `vllm==0.27.1`'s
attention kernel by exact source anchors (a version drift fails loudly at import).

```bash
uv sync                       # HF path + tests
uv sync --extra vllm          # add vLLM serving (GPU box)
uv sync --extra demos         # matplotlib / jupyter for the demos
```

`uv sync` installs the package, which is what registers the vLLM plugin entry point
(`vllm.general_plugins`); a bare `PYTHONPATH=src` checkout can run the HF path but not
vLLM with spawned workers.

## Quick start

**Inspect a gates file (CPU, seconds):**

```bash
python demos/inspect_gates.py gates/gpt-oss-20b/lr0.1/gates.safetensors --plot heatmap.pdf
```

**HuggingFace:**

```python
import torch
from heads_of_security.attention_gating.model_utils import load_gated_model

model, tok = load_gated_model("gates/gpt-oss-20b/lr0.1/gates.safetensors", "openai/gpt-oss-20b", device="cuda:0")
enc = tok.apply_chat_template_with_channels(messages, tools=tools, add_generation_prompt=True)
out = model.generate(input_ids=enc["input_ids"].cuda(), attention_mask=torch.ones_like(enc["input_ids"]).cuda(),
                     token_channels=enc["token_channels"].cuda(), max_new_tokens=200, do_sample=False)
```

`token_channels=None` runs the base model bit-exactly. `demos/hf_generate.py` is the full
version: it renders a chat with an injected tool result, prints the per-token channels, and
generates gated and ungated. Llama-70B needs `--device auto` on 4 x 80 GB.

Or directly on the model classes:

```python
from heads_of_security.attention_gating import GptOssGatedForCausalLM, LlamaGatedForCausalLM
m = GptOssGatedForCausalLM.from_gates("gates/gpt-oss-20b/lr0.1/gates.safetensors",
                                      base_model="openai/gpt-oss-20b", dtype=torch.bfloat16, device_map="cuda:0")
```

**vLLM:**

```bash
python -m heads_of_security.vllm_gates.build_snapshot \
    --gates gates/gpt-oss-20b/lr0.1/gates.safetensors --model openai/gpt-oss-20b --out ~/gated_snapshots/gptoss20b-gated
python -m heads_of_security.vllm_gates.server --snapshot ~/gated_snapshots/gptoss20b-gated \
    --family gptoss --base-model openai/gpt-oss-20b --port 8000
```

The server speaks the OpenAI chat-completions API. Tool results (`role: tool`) are labelled
untrusted, everything else trusted, exactly as during training. Requesting model
`<model-name>-vanilla` on the same server drops the channel tape and returns the base model.
`demos/build_and_serve_vllm.sh gptoss|llama` wraps both steps and shows a `curl` example.
vLLM serving requires `enforce_eager` and prefix caching off (the channel tape is republished
per step, and channels are not a function of the tokens); the server sets both.

## Gate file format

A gates file is a plain `safetensors` shard with only gate parameters:

```
model.layers.{i}.self_attn.head_gates   float32 [C, C, H]   query channel x key channel x head
model.layers.{i}.self_attn.sink_gates   float32 [C, H]      gpt-oss only
```

The loaders infer the channel count from the shape and `gating_sink` from the presence of
`sink_gates`; nothing else is needed. See `gates/README.md` for the five released files, the
training recipe, and what each does to attention.

## Tests

```bash
uv run pytest
```

All tests run on CPU with tiny randomly-initialised configs (the tokenizer tests download the
two base tokenizers). They check, for both architectures: bit-exact identity to the base
model with `token_channels=None` and with open gates; training-path vs inference-path
equivalence of the gate operator; KV-cache / `generate()` consistency; the sink gate and
sliding-window handling on gpt-oss; channel alignment under static caches; the `from_gates`
loader; and byte-parity of the channel tokenizers with the stock chat templates.

## License

MIT, see `LICENSE`.
