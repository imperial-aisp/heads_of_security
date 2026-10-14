"""Heads of Security demo: gpt-oss-20b with trained attention gates vs the base model.

Each scenario is an AgentDojo conversation from the paper's evaluation, cut right after the
tool result that carries the prompt injection. The visitor can rewrite that tool result;
both models then generate their next step (tools are not executed).

Runs on a ZeroGPU Space. Two Space variables select the setup:
  HOS_WEIGHTS   mxfp4 (default) keeps the MoE experts in MXFP4 (~13 GB);
                bf16 dequantizes them (~39 GB, needs HOS_GPU_SIZE=xlarge).
  HOS_GPU_SIZE  large (default, 48 GB) or xlarge (96 GB, twice the quota).
"""

import spaces  # noqa: I001  -- must be imported before torch so ZeroGPU can patch CUDA

import html
import json
import os
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import Mxfp4Config, TextIteratorStreamer

from heads_of_security.attention_gating import GptOssGatedForCausalLM, GptOssGatedTokenizer
from heads_of_security.serve.harmony_parse import parse_harmony_completion

GATES_REPO = "imperial-cpg/hos-gpt-oss-20b"
BASE_MODEL = "openai/gpt-oss-20b"
WEIGHTS = os.environ.get("HOS_WEIGHTS", "mxfp4")
GPU_SIZE = os.environ.get("HOS_GPU_SIZE", "large")
PAPER_URL = "https://arxiv.org/abs/XXXX.XXXXX"  # TODO: arXiv link
CODE_URL = "https://github.com/imperial-aisp/heads_of_security"
MODEL_URL = f"https://huggingface.co/{GATES_REPO}"
# The prompt states the current date; pin it so outputs don't drift from day to day.
DATE = "2026-10-13"
MAX_NEW_TOKENS = 512

# ZeroGPU wants models placed on cuda at import time; a real GPU is attached only
# inside @spaces.GPU functions.
_t0 = time.time()
model = GptOssGatedForCausalLM.from_gates(
    hf_hub_download(GATES_REPO, "gates.safetensors"),
    base_model=BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="cuda",
    **({} if WEIGHTS == "mxfp4" else {"quantization_config": Mxfp4Config(dequantize=True)}),
)
model.eval()
tok = GptOssGatedTokenizer.from_pretrained(
    BASE_MODEL, chat_template_channel=int(model.config.gating_num_channels) == 3
)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model.generation_config.pad_token_id = tok.pad_token_id
print(
    f"weights={WEIGHTS}, experts={type(model.model.layers[0].mlp.experts).__name__}, "
    f"gpu_size={GPU_SIZE}, load {time.time() - _t0:.0f}s",
    flush=True,
)

# Each scenario: the conversation up to the injected tool result (exactly as in the eval),
# the suite's tool definitions, and the tools AgentDojo's ground truth uses for the user's
# task and for the attacker's task.
SCENARIOS = json.loads((Path(__file__).parent / "scenarios.json").read_text())
BY_TITLE = {s["title"]: s for s in SCENARIOS}


def build_messages(s: dict, tool_response: str) -> list[dict]:
    return s["messages"][:-1] + [{**s["messages"][-1], "content": tool_response}]


def encode(s: dict, tool_response: str) -> dict:
    return tok.apply_chat_template_with_channels(
        build_messages(s, tool_response), tools=s["tools"], add_generation_prompt=True, date_string=DATE
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CSS = """
.hos-pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8em;font-weight:600;}
.hos-pill-neutral {background:#e5e7eb;color:#374151;}
.hos-pill-trusted {background:#dcfce7;color:#166534;}
.hos-pill-untrusted {background:#fee2e2;color:#991b1b;}
.hos-mono {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.85em;}
.hos-muted {opacity:0.8;}
.hos-card {display:block;width:100%;box-sizing:border-box;background:#f3f4f6;color:#111827;
           border:1px solid #d1d5db !important;border-radius:8px;padding:8px 12px;margin:0 0 10px 0;}
.hos-card-body {white-space:pre-wrap;word-break:break-word;margin-top:6px;}
.hos-card-tools {margin-top:10px;padding-top:8px;border-top:1px solid #d1d5db !important;}
details.hos-card > summary {list-style:none;cursor:pointer;}
details.hos-card > summary::-webkit-details-marker {display:none;}
details.hos-card > summary .hos-head::after {content:" \\25B8";}
details.hos-card[open] > summary .hos-head::after {content:" \\25BE";}
/* On Spaces, Gradio's resets (`* {border-width:0}`, `.prose * {color}`) load after this CSS,
   so borders and header colours need !important. */
.hos-head {font-size:0.85em;font-weight:600;color:#4b5563 !important;}
.hos-head-trusted {color:#15803d !important;}
.hos-head-untrusted {color:#b91c1c !important;}
/* Long reasoning scrolls inside its card, pinned to the newest text, so the tool call stays in view. */
.hos-scroll {max-height:320px;overflow-y:auto;display:flex;flex-direction:column-reverse;}
.hos-call-user {background:#dcfce7;border-color:#22c55e !important;color:#14532d;}
.hos-call-attacker {background:#fee2e2;border-color:#ef4444 !important;color:#7f1d1d;}
.hos-tokens {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.8em;line-height:1.7;
             white-space:pre-wrap;word-break:break-word;overflow-y:auto;margin-top:6px;flex:1;min-height:0;
             display:flex;flex-direction:column-reverse;}  /* opens scrolled to the end of the prompt */
/* The raw input fills the right column without adding to its height, so it matches the
   conversation column and scrolls inside. */
#raw-title {flex-grow:0 !important;}
#raw-input {flex:1 1 0;min-height:360px;position:relative !important;overflow:hidden;}
.hos-raw {position:absolute;inset:0;display:flex;flex-direction:column;box-sizing:border-box;
          border:1px solid #d1d5db !important;border-radius:8px;padding:8px 12px;}
.hos-links {display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 8px 0;}
.hos-links a {display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border:1px solid #d1d5db !important;
              border-radius:8px;text-decoration:none !important;font-weight:600;font-size:0.9em;color:#111827 !important;}
.hos-links a:hover {background:#f3f4f6;}
.dark .hos-links a {border-color:#4b5563 !important;color:#e5e7eb !important;}
.dark .hos-links a:hover {background:#1f2937;}
.hos-footer {font-size:0.75em;opacity:0.65;}
/* Drop Gradio's inner padding so cards line up with the headings and the tool response. */
#context, #context .html-container, #base-out .html-container, #hos-out .html-container {padding:0 !important;}
#context .hos-card:last-child {margin-bottom:0;}
/* The editable tool response is styled as a card like the other roles, on a white background. */
#tool-response {background:#ffffff !important;border:1px solid #d1d5db !important;border-radius:8px !important;
                padding:8px 12px !important;box-shadow:none !important;}
#tool-response:focus-within {border-color:#ef4444 !important;}
.form:has(> #tool-response) {border:none !important;background:none !important;box-shadow:none !important;}
#tool-response label > span:first-child {font-size:0.85em;font-weight:600;color:#4b5563;margin-bottom:6px;}
#tool-response label > span:first-child::after {content:"untrusted";margin-left:0.3em;color:#b91c1c;}
#tool-response textarea {background:transparent !important;color:#111827 !important;border:none !important;
                         box-shadow:none !important;padding:0 !important;
                         font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.85em;}
.dark .hos-pill-neutral {background:#374151;color:#e5e7eb;}
.dark .hos-pill-trusted {background:#14532d;color:#dcfce7;}
.dark .hos-pill-untrusted {background:#7f1d1d;color:#fee2e2;}
.dark .hos-card {background:#1f2937;color:#e5e7eb;border-color:#4b5563 !important;}
.dark .hos-call-user {background:#14532d;border-color:#22c55e !important;color:#dcfce7;}
.dark .hos-call-attacker {background:#7f1d1d;border-color:#ef4444 !important;color:#fee2e2;}
.dark .hos-card-tools {border-color:#4b5563 !important;}
.dark .hos-head {color:#9ca3af !important;}
.dark .hos-head-trusted {color:#4ade80 !important;}
.dark .hos-head-untrusted {color:#f87171 !important;}
.dark .hos-raw {border-color:#4b5563 !important;}
.dark #tool-response {background:#0b0f19 !important;border-color:#4b5563 !important;}
.dark #tool-response:focus-within {border-color:#ef4444 !important;}
.dark #tool-response label > span:first-child {color:#9ca3af;}
.dark #tool-response label > span:first-child::after {color:#f87171;}
.dark #tool-response textarea {color:#f9fafb !important;}
"""

CHANNELS = {
    0: ("trusted", "rgba(34, 197, 94, 0.25)"),
    1: ("untrusted", "rgba(239, 68, 68, 0.28)"),
    2: ("template", "rgba(156, 163, 175, 0.25)"),
}


def esc(text: str) -> str:
    return html.escape(text)


def head(text: str, kind: str = "neutral", trust: str | None = None) -> str:
    """A card header; `trust` appends a green "trusted" or red "untrusted" tag."""
    tag = f' · <span class="hos-head-{trust}">{trust}</span>' if trust else ""
    return f'<span class="hos-head hos-head-{kind}">{esc(text)}{tag}</span>'


def format_call(name: str, args) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"{name}({args})"
    return f"{name}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (args or {}).items()) + ")"


def render_context(s: dict) -> str:
    """The fixed part of the conversation: system prompt, user request, the assistant's tool call."""
    system, user, assistant = s["messages"][0], s["messages"][1], s["messages"][2]
    tools = "".join(
        f"<div><b>{esc(t['function']['name'])}</b>: {esc(t['function']['description'].strip().splitlines()[0])}</div>"
        for t in s["tools"]
    )
    call = assistant["tool_calls"][0]["function"]
    return (
        f'<details class="hos-card"><summary>{head("System prompt", trust="trusted")}</summary>'
        f'<div class="hos-card-body">{esc(system["content"])}</div>'
        f'<div class="hos-card-tools"><div style="margin-bottom:4px"><b>{len(s["tools"])} tools available</b></div>{tools}</div>'
        "</details>"
        + f'<div class="hos-card">{head("User", trust="trusted")}<div class="hos-card-body">{esc(user["content"])}</div></div>'
        + f'<div class="hos-card">{head("Assistant", trust="trusted")}'
        f'<div class="hos-card-body hos-mono">{esc(format_call(call["name"], call["arguments"]))}</div></div>'
    )


def tool_label(s: dict) -> str:
    return f"Tool response from {s['messages'][-1]['name']} ·"  # CSS appends a red "untrusted"


def render_channels(s: dict, tool_response: str) -> str:
    enc = encode(s, tool_response)
    ids, chans = enc["input_ids"][0].tolist(), enc["token_channels"][0].tolist()
    legend = " ".join(
        f'<span style="background:{bg};padding:1px 8px;border-radius:999px">{name}: {chans.count(c)}</span>'
        for c, (name, bg) in CHANNELS.items()
    )
    spans = "".join(
        f'<span title="{CHANNELS[c][0]}" style="background:{CHANNELS[c][1]}">'
        f'{esc(tok.base.decode([i])).replace(chr(10), "↵" + chr(10))}</span>'
        for i, c in zip(ids, chans)
    )
    return (
        f'<div class="hos-raw"><div class="hos-footer">{len(ids)} tokens</div><div style="margin-top:4px">{legend}</div>'
        f'<div class="hos-tokens"><div>{spans}</div></div></div>'
    )


def render_output(s: dict, text: str, *, done: bool, seconds: float = 0.0, n_tokens: int = 0) -> str:
    parsed = parse_harmony_completion(text)
    parts = []
    if parsed["reasoning"]:
        parts.append(
            f'<details class="hos-card" open><summary>{head("Reasoning")}</summary>'
            f'<div class="hos-card-body hos-muted hos-scroll"><div>{esc(parsed["reasoning"])}</div></div></details>'
        )
    for call in parsed["tool_calls"]:
        fn = call["function"]
        kind = "user" if fn["name"] in s["user_tools"] else "attacker" if fn["name"] in s["attacker_tools"] else "other"
        tone = {"user": "trusted", "attacker": "untrusted"}.get(kind, "neutral")
        parts.append(
            f'<div class="hos-card hos-call-{kind}">{head("Tool call", tone)}'
            f'<div class="hos-card-body hos-mono">{esc(format_call(fn["name"], fn["arguments"]))}</div></div>'
        )
    if parsed["content"]:
        parts.append(f'<div class="hos-card">{head("Response")}<div class="hos-card-body">{esc(parsed["content"])}</div></div>')
    if not done and not (parsed["tool_calls"] or parsed["content"]):
        parts.append('<div class="hos-footer">thinking…</div>')
    if done:
        finished = text.rstrip().endswith(("<|return|>", "<|call|>"))
        note = "" if finished else f" · stopped at the {MAX_NEW_TOKENS}-token limit"
        parts.append(f'<div class="hos-footer">{n_tokens} tokens in {seconds:.1f}s{note}</div>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def gpu_duration(title: str, tool_response: str, arm: str) -> int:
    # Measured on ZeroGPU (MXFP4): ~10 tok/s with gates, ~16 tok/s without; plus prefill and Triton JIT.
    return int(MAX_NEW_TOKENS * (0.1 if arm == "hos" else 0.065)) + 15


@spaces.GPU(duration=gpu_duration, size=GPU_SIZE)
def generate(title: str, tool_response: str, arm: str):
    s = BY_TITLE[title]
    enc = encode(s, tool_response)
    input_ids = enc["input_ids"].to("cuda")
    # token_channels=None runs the unmodified base model: same weights, gates off.
    token_channels = enc["token_channels"].to("cuda") if arm == "hos" else None
    streamer = TextIteratorStreamer(tok.base, skip_prompt=True, skip_special_tokens=False)

    def run():
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                token_channels=token_channels,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                streamer=streamer,
            )

    thread = threading.Thread(target=run)
    t0 = time.time()
    thread.start()
    text = ""
    for piece in streamer:
        text += piece
        yield render_output(s, text, done=False)
    thread.join()
    n_tokens = len(tok.base(text, add_special_tokens=False)["input_ids"])
    yield render_output(s, text, done=True, seconds=time.time() - t0, n_tokens=n_tokens)


def run_hos(title, tool_response):
    yield from generate(title, tool_response, "hos")


def run_base(title, tool_response):
    yield from generate(title, tool_response, "base")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def original_response(title: str) -> str:
    return BY_TITLE[title]["messages"][-1]["content"]


def on_scenario(title):
    s = BY_TITLE[title]
    response = original_response(title)
    return render_context(s), gr.Textbox(value=response, label=tool_label(s)), render_channels(s, response), "", ""


def on_tool_response(title, tool_response):
    return render_channels(BY_TITLE[title], tool_response)


# Octicons (MIT): file, mark-github and package.
ICON_PAPER = (
    '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M2 1.75C2 .784 '
    "2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 "
    "16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 "
    ".25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"
    '"/></svg>'
)
ICON_CODE = (
    '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 '
    "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82"
    "-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-"
    "3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
    "1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 "
    "1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
    '"/></svg>'
)
ICON_MODELS = (
    '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="m8.878.392 5.25 '
    "3.045c.54.314.872.89.872 1.514v6.098a1.75 1.75 0 0 1-.872 1.514l-5.25 3.045a1.75 1.75 0 0 1-1.756 0l-5.25-3.045A1.75 "
    "1.75 0 0 1 1 11.049V4.951c0-.624.332-1.201.872-1.514L7.122.392a1.75 1.75 0 0 1 1.756 0ZM7.875 1.69l-4.63 2.685L8 "
    "7.133l4.755-2.758-4.63-2.685a.248.248 0 0 0-.25 0ZM2.5 5.677v5.372c0 .09.047.171.125.216l4.625 2.683V8.432Zm6.25 "
    "8.271 4.625-2.683a.25.25 0 0 0 .125-.216V5.677L8.75 8.432Z"
    '"/></svg>'
)
LINKS = '<div class="hos-links">' + "".join(
    f'<a href="{url}" target="_blank" rel="noopener">{icon}<span>{label}</span></a>'
    for url, icon, label in [(PAPER_URL, ICON_PAPER, "Paper"), (CODE_URL, ICON_CODE, "Code"), (MODEL_URL, ICON_MODELS, "Models")]
) + "</div>"

INTRO = f"""# Heads of Security

{LINKS}

**Heads of Security (HoS)** is a prompt injection defence that enforces instruction-data separation
through learned per-head attention gates. These gates are trained to redistribute attention between
trusted and untrusted tokens, enabling the model to reason over untrusted content without following
instructions injected into it.

Below are a few examples from the [AgentDojo](https://agentdojo.spylab.ai) benchmark.
The model receives a standard system prompt and access to a range of tools.
The user sends a request, the model calls a tool and receives the response.
This response contains both data needed for the user's task and an adversarial injection.
System and user prompts are treated as trusted, while all tool outputs are treated as untrusted.
"""

LEGEND = (
    '<div class="hos-footer">Tool calls are shown in '
    '<span class="hos-pill hos-pill-trusted">green</span> when the user\'s task needs them and in '
    '<span class="hos-pill hos-pill-untrusted">red</span> when the attacker\'s task does. '
    "Both models share the same weights (the base model has the gates switched off), use greedy decoding "
    "and medium reasoning effort. Tools are not executed.</div>"
)

# On huggingface.co the app sits in an iframe that iframe-resizer sizes to the page. It only re-measures
# on DOM mutations, so layout-only growth while streaming can leave the bottom of the page cut off.
HEAD = """<script>
(function () {
  if (window.parent === window) return;
  let tries = 0;
  (function hook() {
    if (!window.parentIFrame) { if (tries++ < 60) setTimeout(hook, 500); return; }
    new ResizeObserver(() => window.parentIFrame.size()).observe(document.body);
  })();
})();
</script>"""

first = SCENARIOS[0]
with gr.Blocks() as demo:
    gr.Markdown(INTRO)
    scenario = gr.Dropdown([s["title"] for s in SCENARIOS], value=first["title"], label="Scenario")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Conversation")
            context = gr.HTML(render_context(first), elem_id="context")
            tool_response = gr.Textbox(
                value=original_response(first["title"]), label=tool_label(first), lines=12, max_lines=40, elem_id="tool-response"
            )
            with gr.Row():
                restore = gr.Button("Reset", size="sm")
                run = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### Raw model input", elem_id="raw-title")
            channels = gr.HTML(render_channels(first, original_response(first["title"])), elem_id="raw-input")
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("### Base GPT-OSS-20B")
            base_out = gr.HTML(elem_id="base-out", min_height=120)
        with gr.Column():
            gr.Markdown("### GPT-OSS-20B w/ Heads of Security")
            hos_out = gr.HTML(elem_id="hos-out", min_height=120)
    gr.HTML(LEGEND)

    scenario.change(on_scenario, scenario, [context, tool_response, channels, hos_out, base_out])
    tool_response.change(
        on_tool_response, [scenario, tool_response], channels, trigger_mode="always_last", show_progress="hidden"
    )
    restore.click(original_response, scenario, tool_response)
    run.click(run_base, [scenario, tool_response], base_out)
    run.click(run_hos, [scenario, tool_response], hos_out)

if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD)
