"""Model-agnostic channel-gating core shared by the Llama and gpt-oss gated models.

The gating *math* is architecture-independent; only the attention ``forward``
(projection / norms / RoPE / cache / sink handling) differs between architectures.
This module factors out the reusable pieces so the per-architecture classes stay thin:

- :class:`GatedAttentionMixin` -- the gate parameters (``head_gates`` and, on
  architectures with an attention sink, ``sink_gates``) plus the gate operator
  ``_apply_qk_gate`` (a per-(query channel, key channel, head) log-bias added to the
  pre-softmax attention mask) and ``_qk_gate_mask``, which wraps the channel-alignment
  bookkeeping (static-cache padding) around it.
- :class:`GatedCausalLMMixin` -- ``freeze_base``, generation-time channel bookkeeping,
  and the ``from_gates`` loader (with a per-subclass ``_BASE_MODEL_REGISTRY``).

Channels are small integers attached to every token (see ``channel_tokenizer``):
0 = trusted (system / user / assistant), 1 = untrusted (tool results), 2 = chat-template
scaffold. A gate table entry ``head_gates[c_q, c_k, h]`` is a logit; ``logsigmoid`` of it
is added to the attention score of every (query of channel ``c_q``, key of channel
``c_k``) pair in head ``h``. Because softmax is shift-invariant along a row, an all-equal
table is the exact identity: the base model is recovered bit for bit at init.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention-side gating
# ---------------------------------------------------------------------------


# Heads gathered per iteration when building the qk gate mask. Trades kernel
# launches for peak memory; see _apply_qk_gate.
_QK_GATE_HEAD_CHUNK = 8

# Every trainable gate parameter name. The single source of truth for "what is a gate":
# freeze_base and the device fix-up after load_state_dict both read this.
GATE_PARAM_SUFFIXES: tuple[str, ...] = ("head_gates", "sink_gates")


class GatedAttentionMixin:
    """Gate parameters + gate operator, mixed into an HF attention module.

    Subclasses call :meth:`_init_qk_gates` from ``__init__`` and :meth:`_qk_gate_mask`
    (or :meth:`_apply_qk_gate` directly) inside their ``forward``.
    """

    def _init_qk_gates(self, config) -> None:
        """Create ``head_gates`` ``[C, C, num_attention_heads]`` for qk-gating.

        With ``gating_sink``, also create ``sink_gates`` ``[C, num_attention_heads]`` --
        one gate per (query channel, head) on the attention sink logit.

        Why the sink needs a gate at all. ``logsigmoid(g) <= 0``, so the content gate can
        only ever *subtract*. In a sink-free model that costs nothing: softmax is
        shift-invariant, so "suppress untrusted" and "amplify trusted" are the same
        operation. gpt-oss's sink is a fixed per-head logit sitting in the denominator,
        which breaks that invariance -- the absolute level of the content logits now
        matters. A content-only gate can therefore express "read untrusted less" only as
        "read less overall", paying for every act of discrimination in capacity lost to
        the sink. Gating the sink restores the missing degree of freedom: a head can now
        suppress untrusted content *and* the sink together, keeping trusted attention intact.

        It also makes the init an exact identity. At init every content gate is
        ``logsigmoid(init)``; giving the sink the same value shifts every logit in the row
        -- content and sink alike -- by one constant, which softmax cancels. Without the
        sink gate that same init would silently move mass onto the sink.
        """
        self.gating_num_channels = config.gating_num_channels
        self.gating_variant = getattr(config, "gating_variant", "qk_gating")
        if self.gating_variant != "qk_gating":
            raise ValueError(f"only gating_variant='qk_gating' is supported; got {self.gating_variant!r}")
        self.gating_init_logit = getattr(config, "gating_init_logit", 4.0)
        num_heads = config.num_attention_heads
        shape: tuple[int, ...] = (self.gating_num_channels, self.gating_num_channels, num_heads)
        self.head_gates = nn.Parameter(torch.full(shape, self.gating_init_logit, dtype=torch.float32))

        self.gating_sink = bool(getattr(config, "gating_sink", False))
        if self.gating_sink:
            self.sink_gates = nn.Parameter(
                torch.full(
                    (self.gating_num_channels, num_heads),
                    self.gating_init_logit,
                    dtype=torch.float32,
                )
            )

    @staticmethod
    def _align_channels(
        token_channels: torch.Tensor,
        new_len: int,
        k_total: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(q_channels, k_channels)`` for the gate lookup.

        The channel-alignment bookkeeping, shared by every consumer of the gate
        table: ``q_channels`` are the last ``new_len`` entries (the actual query
        positions, NOT shifted by any right-padding of the key buffer), and
        ``k_channels`` are right-padded to the full key length ``k_total`` (e.g. a
        static-cache buffer).
        """
        batch_size = token_channels.shape[0]
        tc_len = token_channels.shape[-1]
        q_channels = token_channels[:, -new_len:]
        if tc_len < k_total:
            pad = token_channels.new_zeros(batch_size, k_total - tc_len)
            k_channels = torch.cat([token_channels, pad], dim=-1)
        elif tc_len > k_total:
            raise ValueError(f"token_channels length {tc_len} exceeds key length {k_total}")
        else:
            k_channels = token_channels
        return q_channels, k_channels

    def _qk_gate_mask(
        self,
        attention_mask: torch.Tensor,
        token_channels: torch.Tensor,
        new_len: int,
        k_total: int,
    ) -> torch.Tensor:
        """Add the per-channel-pair log-bias to ``attention_mask``.

        Returns a ``[B, H, Q, K]`` mask. That is one full-size tensor more than the
        ``[B, 1, Q, K]`` mask the base model uses, and it stays live for the whole
        of eager attention -- measured on gpt-oss-20b as 4.03 vs 3.03 full-size
        tensors of peak, i.e. gating costs exactly one extra. Prefer adding the
        bias inside a head-chunked attention where the architecture allows it (see
        ``gpt_oss.gated_attention.gated_eager_attention_forward``); this path
        remains for architectures still on HF's mask hook, and for training, where
        the one-hot einsum form is wanted for its backward.
        """
        q_channels, k_channels = self._align_channels(token_channels, new_len, k_total)
        return self._apply_qk_gate(
            self.head_gates, attention_mask, q_channels, k_channels, training=self.training
        )

    @staticmethod
    def _apply_qk_gate(
        head_gates: torch.Tensor,  # [C, C, H]
        attention_mask: torch.Tensor,  # [B, 1, Q, K]
        q_channels: torch.Tensor,  # [B, Q] long
        k_channels: torch.Tensor,  # [B, K] long
        training: bool = False,
    ) -> torch.Tensor:
        log_gates = F.logsigmoid(head_gates)  # [C, C, H]
        if training:
            # One-hot einsum: backward-friendly (dense matmul kernels, no
            # scatter-add contention on the tiny [C,C,H] gradient target).
            q_oh = F.one_hot(q_channels, num_classes=head_gates.shape[0]).to(log_gates.dtype)
            k_oh = F.one_hot(k_channels, num_classes=head_gates.shape[1]).to(log_gates.dtype)
            pair_log_gates = torch.einsum("bqc,bkd,cdh->bhqk", q_oh, k_oh, log_gates)
            return attention_mask + pair_log_gates.to(attention_mask.dtype)

        # Inference. The returned mask must be [B, H, Q, K], because the bias varies
        # per head and eager attention's only hook is the mask -- that one full-size
        # tensor is unavoidable here. What IS avoidable is building the gathered bias
        # at full size *alongside* it: that doubled the cost, and at ~11k-token
        # prompts each such tensor is 14.4 GiB (64 heads x S^2 x 2 bytes).
        #
        # So: allocate the mask once, then gather the bias a few heads at a time and
        # accumulate in place. Peak extra becomes (chunk / H) of a full tensor.
        # Casting the tiny [C, C, H] table before gathering also keeps the gather in
        # the mask's dtype instead of fp32 (values unchanged -- logsigmoid still runs
        # in the gates' own precision).
        log_gates = log_gates.to(attention_mask.dtype)
        num_heads = log_gates.shape[-1]
        out = attention_mask.expand(-1, num_heads, -1, -1).clone()
        q_idx = q_channels.unsqueeze(2)
        k_idx = k_channels.unsqueeze(1)
        # 8 heads ~= 1.8 GiB of temporary at S=11k, versus 14.4 for all 64 at once.
        chunk = max(1, min(_QK_GATE_HEAD_CHUNK, num_heads))
        for start in range(0, num_heads, chunk):
            stop = min(start + chunk, num_heads)
            part = log_gates[..., start:stop][q_idx, k_idx]  # [B, Q, K, chunk]
            out[:, start:stop] += part.permute(0, 3, 1, 2)
        return out


# ---------------------------------------------------------------------------
# Model-side gating
# ---------------------------------------------------------------------------


def infer_gating_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Infer the gating config from the gate tensors in a state dict.

    ``*.head_gates`` must be ``[C, C, H]`` on every gated layer; ``*.sink_gates``
    ``[C, H]``, present on all gated layers or none, marks a checkpoint trained with
    the attention sink gated (``gating_sink``). Inconsistent / wrong-rank / missing
    entries fail here or downstream at ``load_state_dict``.
    """
    qk = {k: v for k, v in state_dict.items() if k.endswith(".head_gates")}
    if not qk:
        raise ValueError("state dict has no '*.head_gates' tensors")
    other = sorted(k for k in state_dict if not k.endswith((".head_gates", ".sink_gates")))
    if other:
        raise ValueError(f"state dict carries non-gate tensors: {other[:4]}...")

    shapes = {tuple(x.shape) for x in qk.values()}
    if len(shapes) != 1:
        raise ValueError(f"head_gates tensors have inconsistent shapes: {shapes}")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[1] != shape[0]:
        raise ValueError(f"head_gates must be [C, C, H]; got {shape}")

    inferred = {
        "num_hidden_layers": len(qk),
        "gating_num_channels": shape[0],
        "num_attention_heads": shape[-1],
        "gating_variant": "qk_gating",
    }
    # A checkpoint carrying sink_gates was trained with the sink gated; one without was
    # not. Inferring it here rather than from a flag keeps sink-free checkpoints loading
    # unchanged -- absent key -> gating_sink False -> the plain forward, bit for bit.
    sink = {k: v for k, v in state_dict.items() if k.endswith(".sink_gates")}
    if sink:
        sink_shapes = {tuple(x.shape) for x in sink.values()}
        if len(sink_shapes) != 1:
            raise ValueError(f"sink_gates tensors have inconsistent shapes: {sink_shapes}")
        s = next(iter(sink_shapes))
        if s != (shape[0], shape[-1]):
            raise ValueError(
                f"sink_gates shape {s} does not match head_gates [C, C, H] = {shape}; "
                f"expected [C, H] = {(shape[0], shape[-1])}"
            )
        if len(sink) != len(qk):
            raise ValueError(
                f"{len(sink)} sink_gates tensors for {len(qk)} head_gates layers -- "
                "a partially-gated checkpoint would silently gate only some layers"
            )
        inferred["gating_sink"] = True
    return inferred


class GatedCausalLMMixin:
    """Shared ``*GatedForCausalLM`` surface: freezing, generation bookkeeping, loader.

    Subclasses must define ``_BASE_MODEL_REGISTRY`` (layer-count -> HF id) for
    ``from_gates`` auto-inference, and set ``self.gating_generated_channel`` in
    ``__init__`` (use :meth:`_init_generated_channel`).
    """

    _BASE_MODEL_REGISTRY: dict[int, str] = {}

    def _init_generated_channel(self, config) -> None:
        # Channel assigned to tokens emitted during .generate().
        self.gating_generated_channel = getattr(config, "gating_generated_channel", 0)

    def freeze_base(self) -> int:
        """Freeze all parameters except the gate tensors (``*.head_gates`` and, under
        ``gating_sink``, ``*.sink_gates``). Returns the trainable count."""
        for p in self.parameters():
            p.requires_grad_(False)
        trainable = 0
        for name, p in self.named_parameters():
            if name.endswith(GATE_PARAM_SUFFIXES):
                p.requires_grad_(True)
                trainable += p.numel()
        return trainable

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        if (token_channels := model_kwargs.get("token_channels")) is not None:
            new = torch.full(
                (token_channels.shape[0], num_new_tokens),
                self.gating_generated_channel,
                dtype=token_channels.dtype,
                device=token_channels.device,
            )
            model_kwargs["token_channels"] = torch.cat([token_channels, new], dim=-1)
        return model_kwargs

    @classmethod
    def _resolve_base_model(cls, n_layers: int) -> str:
        registry = cls._BASE_MODEL_REGISTRY
        if n_layers not in registry:
            raise ValueError(
                f"Cannot infer base_model: no entry for {n_layers}-layer architecture "
                f"in {cls.__name__}._BASE_MODEL_REGISTRY {registry}. "
                "Pass base_model=<hf-id-or-path> explicitly."
            )
        base_model = registry[n_layers]
        logger.warning(
            "from_gates: inferred base_model=%r from %d-layer snapshot. "
            "Pass base_model=... to override.",
            base_model,
            n_layers,
        )
        return base_model

    @classmethod
    def from_gates(
        cls,
        gates: "str | Path | dict[str, torch.Tensor]",
        base_model: str | None = None,
        **kwargs,
    ):
        """Load a gated model from a trained gates state dict (path or dict).

        The base model is loaded via ``cls.config_class``/``from_pretrained`` and the
        gates are layered on with ``load_state_dict(..., strict=False)``. ``base_model``
        is auto-inferred from the per-layer count via ``cls._BASE_MODEL_REGISTRY`` when
        omitted. Every kwarg goes to ``from_pretrained`` (``dtype``, ``device_map``, ...).
        """
        gates = load_gates_file(gates)
        inferred = infer_gating_from_state_dict(gates)
        if base_model is None:
            base_model = cls._resolve_base_model(inferred["num_hidden_layers"])

        config = cls.config_class.from_pretrained(base_model)
        config.gating_variant = inferred["gating_variant"]
        config.gating_num_channels = inferred["gating_num_channels"]
        # Every inferred gating field has to reach the config. If gating_sink were dropped
        # here the model would be built without sink_gates and strict=False would silently
        # discard those tensors.
        if "gating_sink" in inferred:
            config.gating_sink = inferred["gating_sink"]

        model = cls.from_pretrained(base_model, config=config, **kwargs)
        incompatible = model.load_state_dict(gates, strict=False)
        # strict=False is required (the base weights are absent from `gates`), but it also
        # makes a gate tensor with no home a no-op. Nothing in `gates` may go unclaimed.
        if incompatible.unexpected_keys:
            raise ValueError(
                f"gates file has {len(incompatible.unexpected_keys)} tensor(s) the model "
                f"has no parameter for: {sorted(incompatible.unexpected_keys)[:4]}... "
                "The gating config and the checkpoint disagree."
            )
        cls._move_gates_to_layer_devices(model)
        return model

    @staticmethod
    def _move_gates_to_layer_devices(model) -> None:
        """After ``load_state_dict`` under device_map, gate tensors may land on the
        wrong device. Move each to match its owning module's device."""
        for layer in model.model.layers:
            mixer = getattr(layer, "self_attn", None)
            if mixer is None:
                continue
            for gate_attr in GATE_PARAM_SUFFIXES:
                gate = getattr(mixer, gate_attr, None)
                if gate is None:
                    continue
                # Anchor device off any real weight in the mixer.
                ref = next((p for p in mixer.parameters() if p is not gate), None)
                if ref is not None and gate.device != ref.device:
                    setattr(mixer, gate_attr, nn.Parameter(gate.data.to(ref.device)))


def load_gates_file(gates: str | Path | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize a gates argument (path or dict) into a state dict."""
    if isinstance(gates, (str, Path)):
        return load_file(str(gates))
    return gates
