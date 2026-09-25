"""Gate tables shared by the gpt-oss and Llama vLLM models.

The trained ``head_gates`` / ``sink_gates`` are full-width ``[C, C, H]`` / ``[C, H]`` logit
tensors loaded unsharded on every rank; these helpers turn them into this rank's
``logsigmoid``'d fp32 slices, padded with the all-zero "ungated" row/column that makes a
request with no channels an arithmetic identity to the base model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_gate_log_table(head_gates: torch.Tensor, num_local_heads: int) -> torch.Tensor:
    """This rank's logsigmoid'd gate table, padded with the ungated row/column.

    ``head_gates`` is the full-width ``[C, C, H_total]`` checkpoint tensor; the
    slice matches vLLM's contiguous head partitioning in the column-parallel
    qkv projection. The extra all-zero row and column at index ``C`` (see
    ``channels.ungated_index``) makes a request labelled entirely with that
    index an arithmetic identity to the base model — channel 0 would still
    apply the trained trusted->trusted bias, which is not identity because a
    uniform bias shifts mass onto anything unbiased (e.g. the gpt-oss sink).
    """
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    from heads_of_security.vllm_gates import channels as ch

    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    assert head_gates.shape[-1] == num_local_heads * world, (
        f"head_gates cover {head_gates.shape[-1]} heads, expected "
        f"{num_local_heads} local x {world} ranks"
    )
    table = F.logsigmoid(head_gates.data.float())
    table = table[:, :, rank * num_local_heads : (rank + 1) * num_local_heads]
    num_channels = table.shape[0]
    assert table.shape[1] == num_channels, f"head_gates must be [C, C, H], got {tuple(head_gates.shape)}"
    padded = torch.zeros(num_channels + 1, num_channels + 1, table.shape[2],
                         dtype=table.dtype, device=table.device)
    padded[:num_channels, :num_channels] = table
    ch.set_ungated_index(num_channels)
    return padded.contiguous()


def build_sink_gate_log_table(sink_gates: torch.Tensor, num_local_heads: int) -> torch.Tensor:
    """This rank's logsigmoid'd sink-gate row per channel, padded like the
    content table: the appended all-zero row at index ``C`` keeps the ungated
    index an arithmetic identity (sink logit unchanged)."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    assert sink_gates.shape[-1] == num_local_heads * world, (
        f"sink_gates cover {sink_gates.shape[-1]} heads, expected "
        f"{num_local_heads} local x {world} ranks"
    )
    table = F.logsigmoid(sink_gates.data.float())
    table = table[:, rank * num_local_heads : (rank + 1) * num_local_heads]
    num_channels = table.shape[0]
    padded = torch.zeros(num_channels + 1, table.shape[1],
                         dtype=table.dtype, device=table.device)
    padded[:num_channels] = table
    return padded.contiguous()


def check_variant(config) -> None:
    variant = getattr(config, "gating_variant", "qk_gating")
    if variant != "qk_gating":
        raise ValueError(f"unsupported gating_variant {variant!r} for vLLM serving (only qk_gating)")
