"""Side-band that carries token channels to the gated attention kernel.

Channels are out-of-band by design (the gate's whole point is that trust is
not encoded in the token stream), so vLLM's request path cannot carry them —
this module is the side-band:

- the runner patch (``runner_patch.py``) publishes one flattened int32 tensor
  per engine step, aligned with the step's flattened query rows;
- each gated attention layer keeps a *physical* channel cache mirroring its
  paged KV cache ([num_blocks * block_size]), written through the same
  ``slot_mapping`` the KV write uses, so the kernel can look up the channel of
  any cached key with block-table addressing.

Generated tokens carry channel 0 (trusted), matching the HF gated path where
assistant/CoT tokens are trusted.

Prefix caching must be OFF for gated runs: vLLM shares physical blocks between
requests with identical token prefixes, but channels are not a function of the
tokens (the same text can be trusted in one request and untrusted in another),
so a shared block would smuggle one request's trust labels into another.
``offline.GatedLLM`` enforces this.
"""

from __future__ import annotations

import weakref

import torch

_STEP: dict = {"channels": None}
# Keyed by the layer OBJECT via a weak map: keying by id() leaks the cache after
# an engine is dropped and lets a later layer allocated at the same address
# inherit a stale tensor.
_KV_CHANNEL_CACHES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# Index of the all-zero row/column appended to every gate table: a request
# labelled entirely with it gets bias logsigmoid-free 0.0 on every (q, k) pair,
# i.e. arithmetic identity to the ungated base model. Requests that declare no
# channels use it, so "vanilla" through this stack really is the base model and
# not merely an all-trusted run (which is NOT identity -- a uniform bias still
# drains attention mass to the unbiased sink).
_UNGATED_INDEX: dict = {"value": None}


def set_ungated_index(index: int) -> None:
    _UNGATED_INDEX["value"] = index


def has_ungated_index() -> bool:
    """True once a gated model has finalized its tables in this process. False for
    a stock model served with the plugin installed, whose requests carry no channels
    and must simply not be taped."""
    return _UNGATED_INDEX["value"] is not None


def ungated_index() -> int:
    value = _UNGATED_INDEX["value"]
    if value is None:
        raise RuntimeError(
            "ungated channel index unset -- the gated model did not finalize its "
            "gate tables. Refusing to guess: channel 0 would silently apply the "
            "trusted->trusted bias to an arm labelled vanilla."
        )
    return value


def set_step_channels(channels: torch.Tensor | None) -> None:
    """Publish this engine step's flattened per-query-token channels."""
    _STEP["channels"] = channels


def step_channels(num_tokens: int) -> torch.Tensor:
    t = _STEP["channels"]
    if t is None:
        raise RuntimeError(
            "no step channels published — the heads_of_security.vllm_gates runner patch "
            "is not active (engine running in another process without the "
            "plugin?). Refusing to run the gated model on unlabeled tokens."
        )
    if t.shape[0] < num_tokens:
        raise RuntimeError(
            f"step channels cover {t.shape[0]} tokens < {num_tokens} scheduled"
        )
    return t


def kv_channel_cache(layer, num_slots: int, device: torch.device) -> torch.Tensor:
    """Per-attention-layer physical channel cache, lazily allocated.

    int32 (not int8): the triton gather indexes with it directly, and the
    whole thing is num_blocks*block_size*4 bytes — tens of MB across all
    layers, noise next to the KV cache itself.
    """
    cache = _KV_CHANNEL_CACHES.get(layer)
    if cache is None or cache.numel() < num_slots or cache.device != device:
        cache = torch.zeros(num_slots, dtype=torch.int32, device=device)
        _KV_CHANNEL_CACHES[layer] = cache
    return cache


def reset() -> None:
    _STEP["channels"] = None
    _KV_CHANNEL_CACHES.clear()
