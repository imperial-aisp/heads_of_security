#!/usr/bin/env python
"""Inspect a gates file on CPU: what is in it, and what the gates do to attention.

    python demos/inspect_gates.py gates/gpt-oss-20b/lr0.1/gates.safetensors
    python demos/inspect_gates.py gates/llama-3.3-70b-instruct/lr0.008/gates.safetensors --plot out.pdf

A gates file is a bare safetensors shard holding only the trained gate parameters, keyed
by the HuggingFace module path:

    model.layers.{i}.self_attn.head_gates   float32 [C, C, H]   (every gated layer)
    model.layers.{i}.self_attn.sink_gates   float32 [C, H]      (gpt-oss only: the attention sink)

``head_gates[q, k, h]`` is a logit. At inference ``logsigmoid`` of it is added to the
attention score of every (query token of channel ``q``, key token of channel ``k``) pair in
head ``h``. Channels: 0 = trusted (system / user / assistant), 1 = untrusted (tool results),
2 = chat-template scaffold.

Because softmax is shift-invariant along a row, only *differences within a query row* matter
for content keys. The number this script reports is therefore the row-centred log-bias
``logsigmoid(g[q, k]) - mean_k' logsigmoid(g[q, k'])`` (per layer and head, and summarised
as a C x C table): negative means the head reads that key channel less than it would have,
positive more. For gpt-oss the sink column is centred on the same content-row mean.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

CHANNELS = ["trusted", "untrusted", "scaffold"]


def load(path: Path):
    """-> (B [L, C, C, H], B_sink [L, C, H] or None): logsigmoid of the gates, layer-ordered."""
    sd = load_file(str(path))
    layer_of = lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1))  # noqa: E731
    hg = sorted((k for k in sd if k.endswith(".head_gates")), key=layer_of)
    sg = sorted((k for k in sd if k.endswith(".sink_gates")), key=layer_of)
    if not hg:
        raise SystemExit(f"{path}: no head_gates tensors")
    B = F.logsigmoid(torch.stack([sd[k].float() for k in hg])).numpy()
    B_sink = F.logsigmoid(torch.stack([sd[k].float() for k in sg])).numpy() if sg else None
    return B, B_sink


def row_centred(B, B_sink=None):
    """-> (V [Q, K(+1), L, H], column names): each query row centred on its content-key mean."""
    M = np.moveaxis(B, (1, 2), (0, 1))  # [Q, K, L, H]
    ref = M.mean(axis=1, keepdims=True)
    V = M - ref
    cols = CHANNELS[: B.shape[2]]
    if B_sink is not None:
        sink = np.moveaxis(B_sink, 1, 0)[:, None]  # [Q, 1, L, H]
        V = np.concatenate([V, sink - ref], axis=1)
        cols = cols + ["sink"]
    return V, cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gates", type=Path)
    ap.add_argument("--plot", type=Path, default=None,
                    help="write a per-layer x per-head heatmap of the trusted-query untrusted-vs-trusted shift")
    args = ap.parse_args()

    B, B_sink = load(args.gates)
    L, C, _, H = B.shape
    print(f"{args.gates}")
    print(f"  {L} layers x {H} heads, {C} channels ({', '.join(CHANNELS[:C])}), "
          f"sink gates: {'yes' if B_sink is not None else 'no'}")
    print(f"  parameters: {B.size + (B_sink.size if B_sink is not None else 0):,}  "
          f"(max logsigmoid = {B.max():+.4f}; gates can only attenuate, so this is <= 0)")

    V, cols = row_centred(B, B_sink)
    q_names = CHANNELS[:C]
    print("\n  mean row-centred log-bias (rows: query channel, columns: key channel)")
    print("  " + " " * 12 + "".join(f"{c:>12}" for c in cols))
    for q in range(C):
        print(f"  q={q_names[q]:<10}" + "".join(f"{V[q, k].mean():>+12.4f}" for k in range(len(cols))))
    tu = (V[0, 1] - V[0, 0]).mean()
    print(f"\n  trusted query: untrusted key vs trusted key = {tu:+.4f} nats "
          f"-> attention odds x{np.exp(tu):.3f} (mean over layers and heads)")
    per_layer = (V[0, 1] - V[0, 0]).mean(axis=1)
    strongest = np.argsort(per_layer)[:5]
    print("  layers with the strongest untrusted suppression (trusted query): "
          + ", ".join(f"L{i} ({per_layer[i]:+.3f})" for i in strongest))

    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        D = (V[0, 1] - V[0, 0]).T  # [H, L]: heads on y, layers on x
        lim = np.percentile(np.abs(D), 99) or 1.0
        fig, ax = plt.subplots(figsize=(max(4.0, 0.08 * L + 1.6), 0.08 * H + 1.2))
        im = ax.imshow(D, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="equal", interpolation="nearest",
                       extent=(0.5, L + 0.5, H + 0.5, 0.5))
        ax.set_xlabel("layer")
        ax.set_ylabel("head")
        ax.set_title(f"{args.gates.parent.parent.name} {args.gates.parent.name}: "
                     "log-bias on untrusted vs trusted keys (trusted query)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="nats")
        fig.savefig(args.plot, bbox_inches="tight")
        print(f"\n  wrote {args.plot}")


if __name__ == "__main__":
    main()
