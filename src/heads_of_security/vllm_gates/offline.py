"""Offline gated-generation API: (token_ids, token_channels) in, text out.

``GatedLLM`` wraps ``vllm.LLM`` with the three invariants the gated stack
needs, refusing to run rather than degrade silently:

- the engine stays in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``), so
  :func:`heads_of_security.vllm_gates.register` reaches the model registry and runner;
- prefix caching is off — physical blocks must not be shared across requests,
  because channels are not a function of the tokens (see ``channels.py``);
- ``enforce_eager`` — the channel tape is swapped per step, which CUDA-graph
  replay would not observe. Lifting this needs a persistent-buffer tape.

Channels ride ``SamplingParams.extra_args["token_channels"]`` per request.
"""

from __future__ import annotations

import os

from heads_of_security.vllm_gates import plugin_available, register


class GatedLLM:
    def __init__(self, model: str, **llm_kwargs):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # flashinfer's sampling JIT needs nvcc, which GPU nodes often lack. The
        # torch sampler is exact for greedy/logprob eval.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        register()
        from vllm import LLM

        llm_kwargs.setdefault("enforce_eager", True)
        llm_kwargs.setdefault("enable_prefix_caching", False)
        llm_kwargs.setdefault("dtype", "bfloat16")
        assert llm_kwargs["enforce_eager"], "gated runs require enforce_eager=True"
        assert not llm_kwargs["enable_prefix_caching"], (
            "gated runs require enable_prefix_caching=False (channels are not a "
            "function of the tokens, so shared blocks would leak trust labels)"
        )
        if int(llm_kwargs.get("tensor_parallel_size") or 1) > 1 and not plugin_available():
            # TP workers are separate processes. With the vllm.general_plugins
            # entry point installed they run register() themselves (any start
            # method); without it they see the patches only by fork
            # inheritance. vLLM silently switches to spawn once CUDA is
            # initialized in this process -- the workers would then die on the
            # unknown gated arch, so fail here with the actual cause instead.
            import torch

            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
            assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork", (
                "gated TP runs require fork workers (spawn loses the patches) "
                "unless the package is installed (pip install -e .)"
            )
            assert not torch.cuda.is_initialized() and not torch.cuda._is_in_bad_fork(), (
                "CUDA state is not fork-clean in this process: vLLM's fork "
                "workers would not inherit the gate patches. Construct GatedLLM "
                "before touching CUDA, or pip install -e . for the plugin path."
            )
        self.llm = LLM(model=model, **llm_kwargs)

    def generate(
        self,
        token_ids: list[list[int]],
        token_channels: list[list[int] | None],
        sampling_params,
    ):
        """Batched generation.

        ``token_channels[i] is None`` means UNGATED: the zero-bias row of the
        gate table, an arithmetic identity to the base model (the vanilla arm).
        It is NOT all-trusted -- on gpt-oss an all-zeros tape is a different
        model, because a uniform bias still drains attention mass to the
        unbiased sink. Pass an explicit ``[0] * len(ids)`` for an all-trusted run.
        """
        from vllm import TokensPrompt

        assert len(token_ids) == len(token_channels)
        prompts, params = [], []
        for ids, chans in zip(token_ids, token_channels):
            if chans is not None:
                assert len(chans) == len(ids), (len(chans), len(ids))
            prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
            sp = sampling_params.clone()  # msgspec Struct: clone, then mutate
            extra = dict(sp.extra_args or {})
            if chans is not None:
                extra["token_channels"] = [int(c) for c in chans]
            sp.extra_args = extra
            params.append(sp)
        return self.llm.generate(prompts, params)
