# Trained gates

Five gate files, one per model / learning rate. Each is a bare `safetensors` shard holding
only the gate parameters, keyed by the HuggingFace module path of the attention layer they
belong to. Base weights are not included: the loaders (`from_gates`, `build_snapshot`) pull
the base model from the Hub or a local directory and layer the gates on top.

| file | base model | layers x heads | tensors per layer | parameters | size |
|---|---|---|---|---|---|
| `gpt-oss-20b/lr0.01/gates.safetensors`  | `openai/gpt-oss-20b` | 24 x 64 | `head_gates [3,3,64]`, `sink_gates [3,64]` | 18,432 | 77 KB |
| `gpt-oss-20b/lr0.025/gates.safetensors` | `openai/gpt-oss-20b` | 24 x 64 | same | 18,432 | 77 KB |
| `gpt-oss-20b/lr0.05/gates.safetensors`  | `openai/gpt-oss-20b` | 24 x 64 | same | 18,432 | 77 KB |
| `gpt-oss-20b/lr0.1/gates.safetensors`   | `openai/gpt-oss-20b` | 24 x 64 | same | 18,432 | 77 KB |
| `llama-3.3-70b-instruct/lr0.008/gates.safetensors` | `meta-llama/Llama-3.3-70B-Instruct` | 80 x 64 | `head_gates [3,3,64]` | 46,080 | 188 KB |

## Tensor semantics

- `model.layers.{i}.self_attn.head_gates` — `float32 [C, C, H]`, indexed
  `[query channel, key channel, head]`. Raw logits. At inference `logsigmoid(head_gates[q, k, h])`
  is added to the pre-softmax attention score of every (query token of channel `q`, key token
  of channel `k`) pair in head `h`.
- `model.layers.{i}.self_attn.sink_gates` — `float32 [C, H]`, gpt-oss only. `logsigmoid` of it
  is added to the head's attention-sink logit for queries of channel `q`. gpt-oss's sink is a
  fixed per-head logit in the softmax denominator; gating it lets a head suppress untrusted
  content *and* the sink together, which is the only way to suppress untrusted keys without
  also reading less overall. Llama has no sink, hence no `sink_gates`.

Channels (`C = 3`):

| id | channel | tokens |
|---|---|---|
| 0 | trusted | system prompt, user turns, assistant turns (incl. generated tokens) |
| 1 | untrusted | tool results (`role: tool`), and any role listed in the tokenizer's untrusted lists |
| 2 | scaffold | chat-template tokens: BOS, role headers, turn terminators, tool schemas |

The gating configuration is **inferred from the file**: the channel count from the first
tensor dimension, `gating_sink` from the presence of `sink_gates`. No side config is needed.

## What the gates do

Softmax is shift-invariant along a row, so for content keys only differences within a query
row matter. `python demos/inspect_gates.py <file>` prints the row-centred table. For the
trusted query row, the mean bias on untrusted keys relative to trusted keys is:

| file | untrusted vs trusted key (nats) | attention odds | strongest layers |
|---|---|---|---|
| gpt-oss lr0.01  | -0.025 | x0.975 | 8, 18, 21 |
| gpt-oss lr0.025 | -0.044 | x0.957 | 8, 18, 9 |
| gpt-oss lr0.05  | -0.064 | x0.938 | 8, 18, 14 |
| gpt-oss lr0.1   | -0.116 | x0.890 | 8, 18, 12 |
| llama lr0.008   | -0.011 | x0.989 | 23, 17, 35 |

These are means over all layers and heads; the effect is concentrated in a minority of heads
(see the heat-maps the demo writes with `--plot`). The learning rate is the knob that trades
robustness against utility: higher rates suppress untrusted keys more strongly.

## How they were trained

Direct Preference Optimization (DPO, torchtune's `DPOLoss`) on channel-labelled preference
pairs, with the base model frozen and only the gate tensors trainable. Every gate is
initialised to logit 1.0 (content and sink alike), which is an exact identity to the base
model. One epoch, effective batch size 8 (batch 1 x 8 accumulation steps), AdamW with no
weight decay, cosine schedule with no warm-up, bf16, learning rate as in the file name
(`0.01` to `0.1` for gpt-oss, `0.008` for Llama).

Each preference pair is a chat whose tool result (untrusted channel) carries an injected
instruction: the chosen response follows the user's task, the rejected response follows the
injection. For Llama the pairs are built from an instruction-following corpus with the
injection placed at a random position in the untrusted span (sequences up to 768 tokens);
for gpt-oss they are agentic tool-use traces with chain-of-thought (sequences up to 4096
tokens), so the trained gates see the model's own reasoning format.
