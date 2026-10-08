# Heads of Security

## Summary

This repository provides the code and trained gates for **Heads of Security**, a method for enforcing instruction–data separation through learned per-head attention gates. The gates help the model use untrusted data, such as tool results, without following instructions embedded in it.

The code supports GPT-OSS-20B and Llama-3.3-70B-Instruct with Hugging Face Transformers and vLLM.

## Quick start

Install from the repository root with Python 3.12 or later and `uv`:

```bash
uv sync
```

Load a model with its trained gates:

```python
from heads_of_security import load_gated_model

model, tokenizer = load_gated_model("imperial-cpg/hos-gpt-oss-20b", device="cuda:0")
```

The loader downloads the gates and the corresponding base model. Use `device="auto"` to distribute a model across GPUs.

Available models:

| Model ID | Base model | 
| --- | --- | 
| `imperial-cpg/hos-gpt-oss-20b` | GPT-OSS-20B 
| `imperial-cpg/hos-llama-3.3-70b-instruct` | Llama-3.3-70B-Instruct |

See the [introduction notebook](demos/introduction.ipynb) for a complete example of preparing messages, assigning trust channels, and generating responses. To install the notebook dependencies:

```bash
uv sync --extra demos
```

## vLLM example

Install vLLM and run the demo on a GPU machine:

```bash
uv sync --extra vllm
bash demos/build_and_serve_vllm.sh gptoss
```

For Llama, replace `gptoss` with `llama` (uses four GPUs by default).

The [demo script](demos/build_and_serve_vllm.sh) builds the gated model and starts an OpenAI-compatible API on port 8000. It includes a sample request. The server assigns trust channels automatically, treating tool results as untrusted data.

## Repository structure and tests

```text
src/heads_of_security/
  attention_gating/   Transformers models, tokenizers, and gate loading
  vllm_gates/         vLLM integration and server
  serve/             Message formatting and tool-call parsing
gates/              Trained gate files
demos/              Notebooks and serving example
tests/              Tests for gating, tokenizers, loading, and generation
```

Run the tests:

```bash
uv sync --extra dev
uv run pytest
```

Tests use small models on CPU. Tokenizer tests download the base model tokenizers.

## Method summary

![Heads of Security method diagram](heads_of_security_diagram.png)

Each token is assigned to one of three channels: trusted instructions and conversation, untrusted tool results, or chat-template formatting. Learned gates in each attention head control how much tokens in one channel attend to tokens in another.

Only the gates are trained; the base model stays frozen. Training favors responses that follow the user's task over responses that follow injected instructions. This teaches the gates to reduce the influence of malicious instructions in untrusted data.

See [the gate documentation](gates/README.md) for training and file details.
