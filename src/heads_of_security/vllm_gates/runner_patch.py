"""Publish per-step token channels from inside vLLM's GPU model runner.

``GPUModelRunner._prepare_inputs`` builds the step's flattened token batch as:
requests in ``input_batch`` row order, each contributing its scheduled tokens
at absolute positions ``[num_computed, num_computed + n)``. This wrapper
reproduces exactly that layout for the channel tape:

- prompt positions take the request's channels (from
  ``SamplingParams.extra_args["token_channels"]``);
- positions past the prompt (generated tokens) are 0 = trusted;
- a request that declares **no** channels is labelled with the ungated index
  (the zero row/column of the gate table), so it runs as the plain base model
  rather than as an all-trusted gated run.

The tape is published via :mod:`heads_of_security.vllm_gates.channels` before the
original ``_prepare_inputs`` runs, so it is in place when attention executes.
"""

from __future__ import annotations

import numpy as np
import torch

from heads_of_security.vllm_gates import channels as ch

_REQUEST_CHANNELS: dict[str, np.ndarray] = {}
_INSTALLED = False


def _ingest_scheduler_output(scheduler_output) -> None:
    # Pop BEFORE ingest: vLLM supports a step where an id is both finished and
    # newly scheduled (an aborted request resubmitted under the same id). The
    # other order deletes the channels just ingested and the request then runs
    # ungated -- silently, which is the worst way to lose a defense.
    for req_id in scheduler_output.finished_req_ids:
        _REQUEST_CHANNELS.pop(req_id, None)
    for new_req in scheduler_output.scheduled_new_reqs:
        sp = getattr(new_req, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) or {}
        tc = extra.get("token_channels")
        if tc is not None:
            tape = np.asarray(tc, dtype=np.int32)
            # The kernel gathers with these ids unmasked. The upper bound is
            # checked against the model's table in _build_step_channels; the
            # lower bound here (a negative id would pass that check).
            assert tape.ndim == 1 and (tape.size == 0 or int(tape.min()) >= 0), (
                f"negative channel id in tape for {new_req.req_id}")
            _REQUEST_CHANNELS[new_req.req_id] = tape
        else:
            # No channels declared: drop any stale tape under this id. Request
            # ids restart at 0 per LLM instance, and an abort that never reached
            # finished_req_ids leaves its entry behind -- either way the request
            # would otherwise run gated under a vanilla label.
            _REQUEST_CHANNELS.pop(new_req.req_id, None)


def _build_step_channels(runner, num_scheduled_tokens: np.ndarray) -> torch.Tensor:
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    starts = runner.input_batch.num_computed_tokens_cpu
    parts: list[np.ndarray] = []
    for row in range(num_reqs):
        n = int(num_scheduled_tokens[row])
        prompt_ch = _REQUEST_CHANNELS.get(req_ids[row])
        if prompt_ch is not None:
            n_prompt = int(runner.input_batch.num_prompt_tokens[row])
            assert len(prompt_ch) == n_prompt, (
                f"channel tape {len(prompt_ch)} != prompt {n_prompt} tokens for "
                f"{req_ids[row]}: the tail would be relabelled trusted")
            # A value past the padded table's last row would index the gate
            # gather out of bounds -- and one EQUAL to the ungated index is
            # almost certainly a 3-channel tape sent to a 2-channel model
            # (e.g. scaffold=2 against llama's [2,2,H] gates), which would
            # silently un-gate those tokens rather than error.
            assert prompt_ch.size == 0 or prompt_ch.max() < ch.ungated_index(), (
                f"channel {int(prompt_ch.max())} in tape for {req_ids[row]} "
                f">= ungated index {ch.ungated_index()}: tape does not match "
                f"the model's gating_num_channels")
        if prompt_ch is None:
            # No channels declared: run this request ungated (zero bias),
            # not all-trusted.
            parts.append(np.full(n, ch.ungated_index(), dtype=np.int32))
            continue
        seg = np.zeros(n, dtype=np.int32)  # generated tokens are trusted
        if n > 0:
            lo = int(starts[row])
            take = min(max(len(prompt_ch) - lo, 0), n)
            if take > 0:
                seg[:take] = prompt_ch[lo : lo + take]
        parts.append(seg)
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    return torch.from_numpy(flat).to(runner.device, non_blocking=True)


def install() -> None:
    """Wrap GPUModelRunner._prepare_inputs once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._prepare_inputs
    original_update = GPUModelRunner._update_states

    def _update_states(self, scheduler_output, *args, **kwargs):
        # _prepare_inputs is SKIPPED on a step with no scheduled tokens, and that
        # is exactly the step the scheduler uses to deliver finished_req_ids --
        # so popping there leaks every request's tape. _update_states always
        # runs, so the bookkeeping lives here.
        _ingest_scheduler_output(scheduler_output)
        return original_update(self, scheduler_output, *args, **kwargs)

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens, *args, **kwargs):
        if not ch.has_ungated_index():
            # A stock model in a process where the plugin installed this patch (the
            # vllm.general_plugins entry point runs register() everywhere). Nothing
            # to tape: no gated layer exists to read it, and building one would fail
            # on the unset ungated index. A gated layer that somehow ran without a
            # tape still fails loudly in channels.step_channels.
            return original(self, scheduler_output, num_scheduled_tokens, *args, **kwargs)
        ch.set_step_channels(_build_step_channels(self, num_scheduled_tokens))
        return original(self, scheduler_output, num_scheduled_tokens, *args, **kwargs)

    GPUModelRunner._update_states = _update_states
    GPUModelRunner._prepare_inputs = _prepare_inputs
    _INSTALLED = True
