"""Heads of Security: channel-gated attention for gpt-oss and Llama."""

__all__ = ["load_gated_model", "load_gated_model_local"]


def __getattr__(name):
    # Lazy, so `import heads_of_security.vllm_gates` (the vLLM plugin entry point)
    # does not pull in the HF model classes.
    if name in __all__:
        from heads_of_security.attention_gating import model_utils

        return getattr(model_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
