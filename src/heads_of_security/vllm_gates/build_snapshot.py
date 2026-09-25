"""Build a vLLM-loadable snapshot: base model weights + trained gates.

vLLM loads models from a checkpoint directory, so the gate tables (a few hundred KB)
are packaged as a sibling snapshot: every base file is symlinked, a
``gates.safetensors`` shard is added, the weight index gains its entries, and
``config.json`` renames the architecture to ``GptOssGatedForCausalLM`` /
``LlamaGatedForCausalLM`` and records the gating fields (channel count, whether
the sink is gated), all inferred from the gates file's tensor names and shapes.

``--model`` takes a HF repo id or an existing local snapshot directory; the
gated architecture name is derived from the base config (``GATED_ARCH``).

Usage::

    python -m heads_of_security.vllm_gates.build_snapshot \
        --gates gates/gpt-oss-20b/lr0.1/gates.safetensors \
        --model openai/gpt-oss-20b --out /path/to/snapshots/gptoss20b-gated

    python -m heads_of_security.vllm_gates.build_snapshot \
        --gates gates/llama-3.3-70b-instruct/lr0.008/gates.safetensors \
        --model meta-llama/Llama-3.3-70B-Instruct --out /path/to/snapshots/llama70b-gated
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Base architecture -> the gated class registered by heads_of_security.vllm_gates.register().
GATED_ARCH = {
    "GptOssForCausalLM": "GptOssGatedForCausalLM",
    "LlamaForCausalLM": "LlamaGatedForCausalLM",
}


def resolve_base_snapshot(model_id: str) -> Path:
    if (local := Path(model_id)).is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, ignore_patterns=["original/*", "metal/*"]))


def build(gates_path: Path, out_dir: Path, model_id: str = "openai/gpt-oss-20b") -> Path:
    from safetensors.torch import load_file, save_file

    base = resolve_base_snapshot(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = load_file(str(gates_path))
    assert gates, f"no tensors in {gates_path}"
    head_layers, sink_layers = set(), set()
    for name, tensor in gates.items():
        layer = name.rsplit(".self_attn.", 1)[0]
        if name.endswith(".self_attn.head_gates"):
            assert tensor.ndim == 3 and tensor.shape[1] == tensor.shape[0], (
                f"head_gates must be [C, C, H]: {tensor.shape}")
            head_layers.add(layer)
        elif name.endswith(".self_attn.sink_gates"):
            assert tensor.ndim == 2, tensor.shape
            sink_layers.add(layer)
        else:
            raise AssertionError(f"unexpected key in gates file: {name}")
    assert head_layers, "no head_gates in gates file"
    # Sink-gated checkpoints carry sink_gates on every gated layer or none:
    # a partial set would silently mix architectures.
    assert not sink_layers or sink_layers == head_layers, (
        f"sink_gates on {len(sink_layers)} layers != head_gates on "
        f"{len(head_layers)}"
    )

    # Symlink every base file except the two we rewrite.
    special = {"config.json", "model.safetensors.index.json"}
    for entry in base.iterdir():
        if entry.name in special or entry.name.startswith("."):
            continue
        dst = out_dir / entry.name
        if dst.is_symlink() or dst.exists():
            dst.unlink() if not dst.is_dir() else shutil.rmtree(dst)
        os.symlink(entry, dst)

    save_file(gates, str(out_dir / "gates.safetensors"))

    index = json.loads((base / "model.safetensors.index.json").read_text())
    for name in gates:
        index["weight_map"][name] = "gates.safetensors"
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    config = json.loads((base / "config.json").read_text())
    (base_arch,) = config["architectures"]
    assert base_arch in GATED_ARCH, (
        f"no gated counterpart known for {base_arch} (have {sorted(GATED_ARCH)})"
    )
    config["architectures"] = [GATED_ARCH[base_arch]]
    head_example = next(t for n, t in gates.items() if n.endswith(".head_gates"))
    config["gating_num_channels"] = int(head_example.shape[0])
    config["gating_variant"] = "qk_gating"
    config["gating_sink"] = bool(sink_layers)
    # The vLLM runner labels every generated token channel 0 (runner_patch);
    # the HF model reads gating_generated_channel from its config. Carry the
    # training value over when the gates dir records one, and refuse anything
    # but 0 here rather than let the two stacks label generations differently.
    gen_ch, gen_src = 0, None
    for cand in (gates_path.parent.parent / "latest" / "config.json",
                 gates_path.parent.parent / "config.json",
                 gates_path.parent / "config.json"):
        if cand.exists():
            gen_ch = int(json.loads(cand.read_text()).get("gating_generated_channel", 0))
            gen_src = cand
            break
    assert gen_ch == 0, (
        f"gating_generated_channel={gen_ch} in {gen_src}: the vLLM runner "
        f"hard-codes 0 for generated tokens -- port that before building")
    config["gating_generated_channel"] = gen_ch
    config["heads_of_security_gates_source"] = str(gates_path)
    # Whether the tables we just wrote are actually non-zero. The model's load guard
    # uses this to decide whether "all zeros after load" means "the loader silently
    # dropped the keys" (the failure it exists to catch) or "this arm is deliberately
    # zero" -- the uniform-gate ablation pins trusted/scaffold/sink at logit 0.0, and
    # an all-zero table is byte-identical to the zeros init, so the guard cannot tell
    # the two apart from values alone. Absent from older snapshots, where the guard
    # defaults to requiring non-zero, which is what every trained arm is.
    config["heads_of_security_head_gates_nonzero"] = bool(
        any(t.any() for n, t in gates.items() if n.endswith(".head_gates")))
    config["heads_of_security_sink_gates_nonzero"] = bool(
        any(t.any() for n, t in gates.items() if n.endswith(".sink_gates")))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"built gated snapshot: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    args = ap.parse_args()
    build(Path(args.gates), Path(args.out), args.model)


if __name__ == "__main__":
    main()
