"""Channel-gated attention (gpt-oss, Llama) inside vLLM.

The HF gated path (``heads_of_security.attention_gating``) is the reference
implementation; this package reproduces its math under vLLM's paged-attention
serving stack:

- :mod:`kernel`        — unified triton attention kernel + channel-pair log-bias
- :mod:`backend`       — Triton backend that feeds the kernel its channels
- :mod:`model_gpt_oss` — ``GptOssGatedForCausalLM`` (vLLM model registry name)
- :mod:`model_llama`   — ``LlamaGatedForCausalLM``
- :mod:`gate_tables`   — per-rank logsigmoid gate tables shared by both models
- :mod:`channels`      — the out-of-band per-token channel side-band
- :mod:`runner_patch`  — publishes each step's channel tape from the runner
- :mod:`build_snapshot`— builds a loadable model dir (base weights + gates)
- :mod:`offline`       — ``GatedLLM``: offline API taking (tokens, channels)

Call :func:`register` before constructing the engine. Two ways it reaches the
processes vLLM creates (EngineCore, TP workers):

- **plugin** (preferred): ``pyproject.toml`` exposes ``register`` under the
  ``vllm.general_plugins`` entry point, and vLLM calls it in every process
  (``arg_utils``, ``v1/engine/core.py``, ``v1/worker/worker_base.py`` — the
  last one before ``init_device`` picks the model runner). This works under
  ``spawn`` and needs the package installed (``pip install -e .``);
  :func:`plugin_available` reports whether the metadata is visible.
- **fork inheritance** (fallback for a ``PYTHONPATH=src`` checkout): the parent
  calls ``register()`` and vLLM forks its children, so the parent must not have
  touched CUDA and must not itself be a forked child of a CUDA-initialized
  process (``torch.cuda._is_in_bad_fork()``). ``server.py`` / ``offline.py``
  check this and fail with the cause.
"""

from __future__ import annotations

PLUGIN_NAME = "heads_of_security_vllm_gates"
_REGISTERED = False


def plugin_available() -> bool:
    """True when vLLM will run :func:`register` itself in every process it
    creates: the ``vllm.general_plugins`` entry point is installed and not
    excluded by ``VLLM_PLUGINS``."""
    import os
    from importlib.metadata import entry_points

    names = {ep.name for ep in entry_points(group="vllm.general_plugins")}
    if PLUGIN_NAME not in names:
        return False
    allowed = os.environ.get("VLLM_PLUGINS")
    return allowed is None or PLUGIN_NAME in {a.strip() for a in allowed.split(",")}


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    import os

    # The channel side-band is wired into the V1 GPU model runner
    # (runner_patch wraps vllm.v1.worker.gpu_model_runner.GPUModelRunner).
    # vllm 0.27 auto-selects a V2 runner (vllm/v1/worker/gpu/) for plain
    # decoder models like Llama; that runner never calls the wrapped
    # _prepare_inputs, so no tape is ever published and every gated request
    # dies with "no step channels published". Pin V1 before the engine reads
    # the env. (gpt-oss happened to fall back to V1 on its own.)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm import ModelRegistry

    from heads_of_security.vllm_gates import runner_patch

    ModelRegistry.register_model(
        "GptOssGatedForCausalLM", "heads_of_security.vllm_gates.model_gpt_oss:GptOssGatedForCausalLM"
    )
    ModelRegistry.register_model(
        "LlamaGatedForCausalLM", "heads_of_security.vllm_gates.model_llama:LlamaGatedForCausalLM"
    )
    runner_patch.install()
    _REGISTERED = True
