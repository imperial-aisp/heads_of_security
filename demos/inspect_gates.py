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
        from matplotlib.colors import LinearSegmentedColormap

        plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                             "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#333333",
                             "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
                             "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
        # coolwarm with its grey midpoint blended toward white, so "no effect" reads as no effect.
        base = matplotlib.colormaps["coolwarm"](np.linspace(0, 1, 256))
        x = np.linspace(0, 1, 256)
        w = 0.75 * (1 - np.abs(2 * x - 1)) ** 2
        base[:, :3] = base[:, :3] * (1 - w[:, None]) + w[:, None]
        cmap = LinearSegmentedColormap.from_list("coolwarm_whitened", base)

        # Shift of attention from untrusted to trusted keys for a trusted query: warm = suppresses
        # untrusted. Heads on y, layers on x, square cells sized in inches, 1-based axes.
        M = B[:, 0, 0, :] - B[:, 0, 1, :]
        D = M.T
        ny, nx = D.shape
        v = np.percentile(np.abs(M), 99) or 1.0
        cell, pad = 0.055, dict(left=0.55, right=1.35, bottom=0.47, top=0.15)
        w_ax, h_ax = nx * cell, ny * cell
        figsize = (pad["left"] + w_ax + pad["right"], pad["bottom"] + h_ax + pad["top"])
        fig = plt.figure(figsize=figsize)
        fig.set_layout_engine("none")
        ax = fig.add_axes([pad["left"] / figsize[0], pad["bottom"] / figsize[1], w_ax / figsize[0], h_ax / figsize[1]])
        im = ax.imshow(D, cmap=cmap, vmin=-v, vmax=v, aspect="equal", interpolation="nearest",
                       extent=(0.5, nx + 0.5, ny + 0.5, 0.5))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        for axis, count in ((ax.xaxis, nx), (ax.yaxis, ny)):
            step = 10 if count >= 40 else 5
            axis.set_ticks([1] + list(range(step, count + 1, step)))
        ax.grid(False)
        cax = fig.add_axes([(pad["left"] + w_ax + 0.12) / figsize[0], pad["bottom"] / figsize[1],
                            0.13 / figsize[0], h_ax / figsize[1]])
        fig.colorbar(im, cax=cax, label="Attention shift\n" + r"(untrusted $\rightarrow$ trusted)")
        fig.savefig(args.plot)
        print(f"\n  wrote {args.plot}")


if __name__ == "__main__":
    main()
