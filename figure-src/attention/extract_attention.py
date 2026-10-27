"""Real attention data for the page's "Attention, before and after" widget.

For the first N scenarios of the demo Space (scenarios.json, copied from
huggingface.co/spaces/igorshilov/heads-of-security) this builds the prompt exactly as the Space
does, runs GPT-OSS-20B with the HoS gates on and off, and records

  * the model's next step, generated as the Space generates it (greedy, 512 new tokens);
  * the attention paid by the first generated token (the query) to every prompt token and to
    itself, in every one of the 24 x 64 heads, from a forward pass over the prompt plus that token.
    Generated tokens carry the model's generated-token channel (trusted), as during decoding.

Heads are "red" or "blue" by their gate shift delta = log sigma(g_TT) - log sigma(g_TU) (paper
Figure 2): red heads (delta > 0) close on untrusted tokens, blue heads open toward them.

Outputs:
  <out>/gates_delta.json  delta per layer and head, and the TOP_HEADS reddest and bluest heads
  <out>/<id>.json         tokens, channels, message segments; per arm the mean attention over all
                          heads, over the red heads and over the blue heads, their channel shares,
                          and the generated step (imported by the page)
  <out>/<id>.heads.bin    the full attention row of each of the reddest and bluest TOP_HEADS heads,
                          see write_heads() (fetched on demand by the "One head" view)
  --full-out/<id>.<arm>.npy  the complete [layers, heads, tokens] arrays, for later analysis

Run on a GPU (see run.slurm); needs the repo's environment (`uv sync`).
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from heads_of_security.attention_gating import GptOssGatedForCausalLM, GptOssGatedTokenizer
from heads_of_security.serve.harmony_parse import parse_harmony_completion

GATES_REPO = "imperial-cpg/hos-gpt-oss-20b"
BASE_MODEL = "openai/gpt-oss-20b"
DATE = "2026-10-13"  # pinned in the Space, so prompts match it token for token
MAX_NEW_TOKENS = 512
ARMS = ("base", "hos")
CHANNELS = ("trusted", "untrusted", "template")
TOP_HEADS = 50  # reddest and bluest heads kept in full in the .heads.bin file
FOLD_MIN = 40  # a run of this many template tokens inside a message is folded on the page


def format_call(name: str, args) -> str:
    """`name(k=v, ...)`, as the Space prints a tool call."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"{name}({args})"
    return f"{name}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (args or {}).items()) + ")"


def segments(tok, ids: list[int]) -> list[dict]:
    """Split the prompt into Harmony messages: [<|start|> header <|message|> body <|end|>]."""
    start, message = tok.base.convert_tokens_to_ids("<|start|>"), tok.base.convert_tokens_to_ids("<|message|>")
    out, i, n = [], 0, len(ids)
    while i < n:
        assert ids[i] == start, f"token {i} is not <|start|>"
        j = i + 1
        while j < n and ids[j] not in (start, message):
            j += 1
        header = tok.decode(ids[i + 1 : j])
        if j < n and ids[j] == message:
            k = j + 1
            while k < n and ids[k] != start:
                k += 1
            out.append({"start": i, "body": j + 1, "end": k, "header": header})
            i = k
        else:  # the generation prompt: `<|start|>assistant`, no body
            out.append({"start": i, "body": j, "end": j, "header": header})
            i = j
    return out


def folds(tok, ids: list[int], chans: list[int], segs: list[dict]) -> list[dict]:
    """The chat template's own system message (model identity, date, reasoning effort, channels), the whole
    developer message (the agent's instructions followed by the tool definitions), and any other long run of
    template-channel tokens inside a message body."""
    whole = {"system": "chat-template system message", "developer": "agent instructions and tool definitions"}
    out = []
    for s in segs:
        if s["header"] in whole and s["end"] - s["body"] >= FOLD_MIN:
            out.append({"start": s["body"], "end": s["end"], "label": whole[s["header"]]})
            continue
        i = s["body"]
        while i < s["end"]:
            if chans[i] == 2:
                j = i
                while j < s["end"] and chans[j] == 2:
                    j += 1
                if j - i >= FOLD_MIN:
                    text = tok.decode(ids[i:j])
                    label = "tool definitions" if "namespace functions" in text else "chat template"
                    out.append({"start": i, "end": j, "label": label})
                i = j
            else:
                i += 1
    return out


def gate_delta(gates_path: str) -> np.ndarray:
    """[layers, heads]: log sigma(g_TT) - log sigma(g_TU); channels 0 trusted, 1 untrusted, 2 template."""
    sd = load_file(gates_path)
    keys = sorted((k for k in sd if k.endswith(".head_gates")), key=lambda k: int(k.split("layers.")[1].split(".")[0]))
    B = F.logsigmoid(torch.stack([sd[k].float() for k in keys]))  # [L, Cq, Ck, H]
    return (B[:, 0, 0, :] - B[:, 0, 1, :]).numpy()


def pick_heads(delta: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """The TOP_HEADS largest-delta (red) and most-negative-delta (blue) heads as (layer, head), 0-based."""
    order = np.argsort(delta, axis=None)
    H = delta.shape[1]
    blue = [(int(i // H), int(i % H)) for i in order[:TOP_HEADS] if delta.flat[i] < 0]
    red = [(int(i // H), int(i % H)) for i in order[::-1][:TOP_HEADS] if delta.flat[i] > 0]
    return {"red": red, "blue": blue}


def attention_rows(model, input_ids, token_channels):
    """[layers, heads, tokens]: the last query position's attention to every prompt token."""
    rows = {}

    def hook(idx):
        def f(module, args, output):
            rows[idx] = output[1][0, :, -1, :].float().cpu()

        return f

    handles = [layer.self_attn.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), token_channels=token_channels, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([rows[i] for i in range(len(model.model.layers))]).numpy()


def shares(att: np.ndarray, chans: np.ndarray) -> np.ndarray:
    """[..., 4]: attention mass on trusted / untrusted / template tokens and on the sink."""
    s = np.stack([att[..., chans == c].sum(-1) for c in range(3)], axis=-1)
    return np.concatenate([s, np.clip(1 - s.sum(-1, keepdims=True), 0, 1)], axis=-1)


def write_heads(path: Path, per_arm: dict[str, np.ndarray], chans: dict[str, np.ndarray], heads: list[tuple[int, int]]) -> None:
    """Binary, little-endian. Header: b"HOSA", u16 version(2), u16 K (number of heads), u16 layers,
    u16 heads per layer, u32 columns (the prompt tokens plus the query token), u16 arms; then K pairs
    u8 layer, u8 head (0-based; the red heads first, then the blue ones). Then per arm (base, hos):
    f32 vmax[K]; u8 shares[K*4] (x/255); u8 q[K*N], the full row of each head, where
    attention = vmax * (q/255)^2 (sqrt-coded, so faint tokens keep resolution)."""
    L, H, N = next(iter(per_arm.values())).shape
    with open(path, "wb") as f:
        f.write(b"HOSA" + struct.pack("<HHHHIH", 2, len(heads), L, H, N, len(ARMS)))
        f.write(bytes(v for lh in heads for v in lh))
        for arm in ARMS:
            a = np.stack([per_arm[arm][l, h] for l, h in heads])  # [K, N]
            vmax = np.maximum(a.max(axis=1, keepdims=True), 1e-12)
            q = np.rint(255 * np.sqrt(np.clip(a / vmax, 0, 1))).astype(np.uint8)
            sh = np.rint(255 * shares(a, chans[arm])).astype(np.uint8)
            f.write(vmax.astype("<f4").tobytes())
            f.write(sh.tobytes())
            f.write(q.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=Path(__file__).with_name("scenarios.json"))
    ap.add_argument("--n", type=int, default=5, help="how many of the Space's scenarios to run (the page shows all five)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--full-out", type=Path, help="also save the complete [layers, heads, tokens] arrays here (.npy)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.full_out:
        args.full_out.mkdir(parents=True, exist_ok=True)

    gates_path = hf_hub_download(GATES_REPO, "gates.safetensors")
    delta = gate_delta(gates_path)
    heads = pick_heads(delta)
    (args.out / "gates_delta.json").write_text(json.dumps({
        "note": "delta[layer][head] = log sigma(g_TT) - log sigma(g_TU) from " + GATES_REPO + " (paper Figure 2); red/blue: the "
                f"{TOP_HEADS} largest and most negative, as [layer, head, delta], 1-based",
        "delta": [[round(float(v), 7) for v in row] for row in delta],  # enough digits to keep every sign
        "red": [[l + 1, h + 1, round(float(delta[l, h]), 4)] for l, h in heads["red"]],
        "blue": [[l + 1, h + 1, round(float(delta[l, h]), 4)] for l, h in heads["blue"]],
    }))
    red_mask, blue_mask = delta > 0, delta < 0
    print(f"{int(red_mask.sum())} red heads, {int(blue_mask.sum())} blue heads", flush=True)

    t0 = time.time()
    model = GptOssGatedForCausalLM.from_gates(gates_path, base_model=BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok = GptOssGatedTokenizer.from_pretrained(BASE_MODEL, chat_template_channel=int(model.config.gating_num_channels) == 3)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id
    print(f"loaded in {time.time() - t0:.0f}s; experts={type(model.model.layers[0].mlp.experts).__name__}", flush=True)

    for s in json.load(open(args.scenarios))[: args.n]:
        enc = tok.apply_chat_template_with_channels(s["messages"], tools=s["tools"], add_generation_prompt=True, date_string=DATE)
        ids = enc["input_ids"][0].tolist()
        chans = np.array(enc["token_channels"][0].tolist())
        input_ids = enc["input_ids"].to("cuda")
        segs = segments(tok, ids)
        record = {
            "id": s["id"], "title": s["title"], "suite": s["suite"], "user_goal": s["user_goal"], "attacker_goal": s["attacker_goal"],
            "user_tools": s["user_tools"], "attacker_tools": s["attacker_tools"], "date": DATE, "n_tokens": len(ids),
            "tokens": [tok.decode([i]) for i in ids], "channels": chans.tolist(), "segments": segs,
            "folds": folds(tok, ids, chans.tolist(), segs), "arms": {},
        }
        per_arm, arm_chans = {}, {}
        for arm in ARMS:
            token_channels = enc["token_channels"].to("cuda") if arm == "hos" else None
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids), token_channels=token_channels,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                )
            text = tok.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=False)
            # the query: the first generated token, appended to the prompt with the generated-token channel
            first = out[0, input_ids.shape[-1]:input_ids.shape[-1] + 1]
            gen_channel = int(getattr(model, "gating_generated_channel", 0))
            ids_q = torch.cat([input_ids, first[None]], dim=-1)
            chans_q = torch.cat([enc["token_channels"], torch.full((1, 1), gen_channel, dtype=enc["token_channels"].dtype)], dim=-1).to("cuda")
            att = attention_rows(model, ids_q, chans_q if arm == "hos" else None)  # [L, H, N + 1]
            per_arm[arm] = att
            if args.full_out:
                np.save(args.full_out / f"{s['id']}.{arm}.npy", att)
            mean = att.mean((0, 1))
            mean_red = att[red_mask].mean(0)
            mean_blue = att[blue_mask].mean(0)
            qchans = np.append(chans, gen_channel)
            arm_chans[arm] = qchans
            parsed = parse_harmony_completion(text)
            calls = [c["function"] for c in parsed["tool_calls"]]
            verdict = "none"
            if calls:
                name = calls[0]["name"]
                verdict = "attacker" if name in s["attacker_tools"] else "user" if name in s["user_tools"] else "other"
            elif parsed["content"]:
                verdict = "response"
            record["arms"][arm] = {
                "query": {"token": tok.decode(first), "id": int(first[0]), "channel": gen_channel},
                "mean": [float(f"{v:.6g}") for v in mean],
                "shares": dict(zip(CHANNELS + ("sink",), [float(f"{v:.5f}") for v in shares(mean, qchans)])),
                "mean_red": [float(f"{v:.6g}") for v in mean_red],
                "shares_red": dict(zip(CHANNELS + ("sink",), [float(f"{v:.5f}") for v in shares(mean_red, qchans)])),
                "mean_blue": [float(f"{v:.6g}") for v in mean_blue],
                "shares_blue": dict(zip(CHANNELS + ("sink",), [float(f"{v:.5f}") for v in shares(mean_blue, qchans)])),
                "generation": {
                    "raw": text, "reasoning": parsed["reasoning"], "content": parsed["content"],
                    "tool_calls": [format_call(c["name"], c["arguments"]) for c in calls], "verdict": verdict,
                    "finished": text.rstrip().endswith(("<|return|>", "<|call|>")), "new_tokens": int(out.shape[-1] - input_ids.shape[-1]),
                },
            }
            print(f"{s['id']} {arm}: {len(ids)} tokens, shares {record['arms'][arm]['shares']}, "
                  f"{record['arms'][arm]['generation']['new_tokens']} new tokens -> {verdict} {calls[:1]} ({time.time() - t1:.0f}s)", flush=True)
        (args.out / f"{s['id']}.json").write_text(json.dumps(record, ensure_ascii=False))
        write_heads(args.out / f"{s['id']}.heads.bin", per_arm, arm_chans, heads["red"] + heads["blue"])
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
