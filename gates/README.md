# Trained gates

This folder contains the trained attention gates for Heads of Security. Each file stores only the gates; the loader loads the base model separately.

## Available gates

Paths below are relative to this folder. `lr` is the learning rate used to train the gates.

| File | Base model |
| --- | --- |
| `gpt-oss-20b/lr0.1/gates.safetensors` | `openai/gpt-oss-20b` |
| `gpt-oss-20b/lr0.05/gates.safetensors` | `openai/gpt-oss-20b` |
| `gpt-oss-20b/lr0.025/gates.safetensors` | `openai/gpt-oss-20b` |
| `gpt-oss-20b/lr0.01/gates.safetensors` | `openai/gpt-oss-20b` |
| `llama-3.3-70b-instruct/lr0.008/gates.safetensors` | `meta-llama/Llama-3.3-70B-Instruct` |

GPT-OSS gates contain about 18,000 parameters per file; Llama gates contain about 46,000.

Higher learning rates generally reduce attack success, but can also reduce performance on normal tasks.

## Load local gates

After [installing the package](../README.md#quick-start), run this from the repository root:

```python
from heads_of_security import load_gated_model_local

model, tokenizer = load_gated_model_local(
    "gates/gpt-oss-20b/lr0.1/gates.safetensors",
    "openai/gpt-oss-20b",
    device="cuda:0",
)
```

Use `device="auto"` to distribute the model across GPUs. For hosted models and a full generation example, see the [main README](../README.md#quick-start) and [introduction notebook](../demos/introduction.ipynb).

## What the gates do

The gates control attention between three token channels:

| Channel | Content |
| --- | --- |
| Trusted | System, user, and assistant messages |
| Untrusted | Tool results |
| Template | Chat formatting, such as role headers and turn markers |

Each attention head has its own gates. They help the model use tool results while reducing the influence of injected instructions. Among the released GPT-OSS variants, higher training learning rates produce stronger average suppression of attention to untrusted tokens.

The [gate exploration notebook](../demos/gates_explore.ipynb) shows how the gates vary across layers and heads.

## Training your own gates

The repository includes code to train new gates with Direct Preference Optimization (DPO). Only the gates are trained; the base model stays frozen. Training favors responses that follow the user's task over injected instructions.

See the [training instructions](../README.md#training-your-own-gates) for setup and commands, and the [training configs](../configs/training/) for model settings. The default configs use the public Hugging Face dataset `imperial-cpg/heads-of-security-dpo`.

Training saves gates to `runs/<name>/gates/epoch_0.safetensors`. Pass that path to `load_gated_model_local` using the example above.

