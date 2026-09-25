"""OpenAI-compatible server for a gated model on vLLM (async, batched).

Channels are derived **server-side from message roles**, exactly the policy the
gated tokenizer implements and the one the gates were trained with: ``tool``
results untrusted, everything else trusted. That keeps the wire format plain
OpenAI (no channel field to forge) -- trust comes from the harness's own
structure, which is the point of the defense.

Arm selection is by requested model name: ``<model-name>-vanilla`` drops the
channel tape (bit-exactly the base model), anything else keeps it.

Usage::

    python -m heads_of_security.vllm_gates.server \
        --snapshot /path/to/snapshots/gptoss20b-gated --family gptoss \
        --base-model openai/gpt-oss-20b --port 8000

    python -m heads_of_security.vllm_gates.server \
        --snapshot /path/to/snapshots/llama70b-gated --family llama \
        --base-model meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

CFG: argparse.Namespace
ENGINE = None
TOK = None
RESPONSE_CACHE: dict[str, dict] = {}
CACHE_HITS = 0
CACHE_STORES = 0
app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: object = None
    name: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


_SAMPLING_LOGGED = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    seed: int | None = None
    # Honored (stock vllm serve honors it; dropping it silently would run an
    # arm-asymmetric protocol for any task with non-empty stop strings).
    stop: list[str] | str | None = None
    # Assistant prefill appended after the generation prompt (stock vllm serve
    # gets the same effect via continue_final_message; this server renders its
    # own channel tape, so the prefill arrives as an explicit field).
    prefill: str | None = None


def _content_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text") or block.get("content") or "")
    return "".join(parts)


def to_template_messages(messages: list[ChatMessage]) -> list[dict]:
    """Wire messages -> template dicts (arguments as objects, not JSON strings)."""
    out = []
    for m in messages:
        entry: dict = {"role": m.role, "content": _content_str(m.content)}
        if m.name:
            entry["name"] = m.name
        if m.tool_calls:
            calls = []
            for call in m.tool_calls:
                call = dict(call)
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except ValueError:
                        # Keep the raw string: an empty object would silently
                        # rewrite history as a no-argument call.
                        pass
                call["function"] = fn
                calls.append(call)
            entry["tool_calls"] = calls
        out.append(entry)
    return out


def _context_length_error(message: str, param: str | None = None) -> JSONResponse:
    """A 400 shaped like OpenAI's context-length error.

    Clients branch on ``error.code`` (agent harnesses typically record a per-task
    skip on ``context_length_exceeded`` / ``param == "max_tokens"`` and re-raise
    anything else).
    """
    return JSONResponse(status_code=400, content={"error": {
        "message": message + " Please reduce the length of the messages.",
        "type": "invalid_request_error", "param": param,
        "code": "context_length_exceeded"}})


def response_cache_key(
    req: ChatCompletionRequest,
    *,
    gated: bool,
    ids: list[int],
    channels: list[int] | None,
    max_tokens: int,
    stop: list[str],
) -> str:
    """Hash the exact wire request and all server-side choices affecting output."""
    material = {
        "request": req.model_dump(mode="json"),
        "rendered_input_ids": ids,
        "rendered_token_channels": channels,
        "effective": {
            "gated": gated,
            "snapshot": CFG.snapshot,
            "ungated": CFG.ungated,
            "family": CFG.family,
            "reasoning_effort": CFG.reasoning_effort,
            "date_string": CFG.date_string,
            "greedy": CFG.greedy,
            "temperature": 0.0 if CFG.greedy else req.temperature,
            "top_p": 1.0 if CFG.greedy else req.top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": req.seed,
        },
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def response_cache_get(key: str) -> dict | None:
    global CACHE_HITS
    cached = RESPONSE_CACHE.get(key)
    if cached is None:
        return None
    CACHE_HITS += 1
    return copy.deepcopy(cached)


def response_cache_put(key: str, response: dict) -> None:
    global CACHE_STORES
    RESPONSE_CACHE[key] = copy.deepcopy(response)
    CACHE_STORES += 1


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": CFG.model_name, "object": "model"},
        {"id": f"{CFG.model_name}-vanilla", "object": "model"},
        # Alias for harnesses that address a generic vLLM endpoint: served gated by default.
        {"id": "vllm_parsed", "object": "model"},
    ]}


@app.get("/health")
def health():
    return {"status": "ok", "response_cache": {
        "enabled": CFG.memoize_responses,
        "entries": len(RESPONSE_CACHE),
        "hits": CACHE_HITS,
        "stores": CACHE_STORES,
    }}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    from vllm import SamplingParams, TokensPrompt

    if req.model in (CFG.model_name, "vllm_parsed"):
        gated = not CFG.ungated
    elif req.model == f"{CFG.model_name}-vanilla":
        gated = False
    else:
        # Never guess the arm: a typo would have run gated while the trace said
        # vanilla, i.e. the loser's benchmark scoring the winner's model.
        raise HTTPException(status_code=404, detail=f"model {req.model!r} is not served here")
    render_kwargs: dict = dict(add_generation_prompt=True, return_tensors=None)
    if CFG.family == "gptoss":
        render_kwargs.update(tools=req.tools, date_string=CFG.date_string,
                             reasoning_effort=CFG.reasoning_effort)
    elif req.tools:
        render_kwargs["tools"] = req.tools
    enc = TOK.apply_chat_template_with_channels(
        to_template_messages(req.messages), **render_kwargs,
    )
    ids = list(enc["input_ids"])
    chans = [int(c) for c in enc["token_channels"]]
    # A short tape is relabelled trusted from the shortfall onward -- i.e. the
    # injected tool result would silently become trusted, which is exactly the
    # failure that produces plausible-but-wrong ASR numbers. Fail loudly.
    assert len(chans) == len(ids), f"tape {len(chans)} != {len(ids)} ids"
    if req.prefill:
        pf = TOK.base.encode(req.prefill, add_special_tokens=False)
        ids += pf
        chans += [0] * len(pf)  # T_CHANNEL: prefill is trusted assistant text
    extra = {"token_channels": chans} if gated else None
    stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    # Length handling mirrors stock `vllm serve`, so a gated arm and a stock
    # baseline skip the same requests.
    if len(ids) >= CFG.max_model_len:
        # An unhandled 500 becomes openai.InternalServerError, which a client's
        # retry wrapper may not catch -- one long prompt would then take the
        # client's remaining tasks with it.
        return _context_length_error(
            f"Prompt too long: {len(ids)} tokens (max_model_len={CFG.max_model_len}).")
    if req.max_tokens is not None and len(ids) + req.max_tokens > CFG.max_model_len:
        # The other stacks REJECT an explicit completion budget that does not
        # fit (scored as a skip). Running it would instead generate up to the
        # window and truncate (finish_reason=length, content null on gpt-oss):
        # a different protocol for this arm only, on exactly the long prompts.
        return _context_length_error(
            f"Client-requested 'max_tokens' is too large: {req.max_tokens} tokens "
            f"(max_model_len={CFG.max_model_len}, input tokens len = {len(ids)}, "
            f"{req.max_tokens} > {CFG.max_model_len} - {len(ids)}).",
            param="max_tokens")
    # Effective budget = min(remaining context, client max_tokens, server cap):
    # the HF server's rule, so a client that omits max_tokens is capped the
    # same way on both stacks.
    max_tokens = min(CFG.max_model_len - len(ids), req.max_tokens or CFG.max_tokens)

    global _SAMPLING_LOGGED
    if not _SAMPLING_LOGGED:
        _SAMPLING_LOGGED = True
        print(f"[sampling] greedy={CFG.greedy} -> temperature={0.0 if CFG.greedy else req.temperature} "
              f"top_p={1.0 if CFG.greedy else req.top_p} (client sent temperature={req.temperature} top_p={req.top_p})",
              flush=True)

    cache_key = None
    if CFG.memoize_responses:
        cache_key = response_cache_key(
            req, gated=gated, ids=ids, channels=chans if gated else None,
            max_tokens=max_tokens, stop=stop,
        )
        if cached := response_cache_get(cache_key):
            print(f"[{time.strftime('%H:%M:%S')}] response-cache hit "
                   f"key={cache_key[:12]} prompt={len(ids)}", flush=True)
            return cached

    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0 if CFG.greedy else req.temperature,
        top_p=1.0 if CFG.greedy else req.top_p,
        stop=stop or None,
        seed=req.seed,
        # Harmony markers are structure we parse; llama output is plain text.
        skip_special_tokens=(CFG.family != "gptoss"),
        extra_args=extra,
    )

    t0 = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    final = None
    async for out in ENGINE.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id):
        final = out
    text = final.outputs[0].text

    if CFG.family != "gptoss":
        # No channel structure to parse: the generation IS the assistant
        # content (llama-style). When the request carried tools, the model was
        # asked for `{"name": .., "parameters": ..}` objects and clients with
        # native tool calling read `message.tool_calls` -- so parse them out.
        # Without tools the text is returned untouched (some harnesses embed
        # their tools in the system prompt and parse `<function=..>` tags themselves).
        n_out = len(final.outputs[0].token_ids)
        engine_finish = final.outputs[0].finish_reason
        if req.tools:
            from heads_of_security.serve.llama_tool_parse import parse_llama_tool_calls

            parsed = parse_llama_tool_calls(text)
            message = {"role": "assistant", "content": parsed["content"],
                       "tool_calls": parsed["tool_calls"] or None}
            finish = ("length" if engine_finish == "length" and not parsed["tool_calls"]
                      else parsed["finish_reason"])
        else:
            message = {"role": "assistant", "content": text, "tool_calls": None}
            finish = engine_finish
        print(f"[{time.strftime('%H:%M:%S')}] {'gated' if gated else 'vanilla'} "
              f"prompt={len(ids)} completion={n_out} {time.time() - t0:.1f}s "
              f"tools={len(req.tools or [])} calls={len(message['tool_calls'] or [])} "
              f"finish={engine_finish}", flush=True)
        response = {
            "id": request_id, "object": "chat.completion", "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": n_out,
                      "total_tokens": len(ids) + n_out},
        }
        if cache_key is not None:
            response_cache_put(cache_key, response)
        return response

    from heads_of_security.serve.harmony_parse import parse_harmony_completion

    parsed = parse_harmony_completion(text)
    # A truncated tool call leaves un-parseable `arguments`; the client does a
    # bare json.loads on it and dies, taking the suite process with it. Drop
    # such calls so the trajectory ends cleanly and is scored as a failure.
    kept = []
    for call in parsed["tool_calls"] or []:
        try:
            json.loads(call["function"]["arguments"])
        except (ValueError, KeyError, TypeError):
            print(f"[{time.strftime('%H:%M:%S')}] dropping unparseable tool call "
                  f"{str(call)[:120]}", flush=True)
            continue
        kept.append(call)
    parsed["tool_calls"] = kept
    # OpenAI wire format wants arguments as a JSON string; harmony_parse already
    # produces that shape, so tool calls pass through untouched.
    message = {"role": "assistant", "content": parsed["content"],
               "reasoning": parsed["reasoning"], "tool_calls": parsed["tool_calls"] or None}
    n_out = len(final.outputs[0].token_ids)
    engine_finish = final.outputs[0].finish_reason
    if engine_finish == "length" and not parsed["tool_calls"]:
        parsed["finish_reason"] = "length"
    print(f"[{time.strftime('%H:%M:%S')}] {'gated' if gated else 'vanilla'} "
          f"prompt={len(ids)} completion={n_out} {time.time() - t0:.1f}s "
          f"tools={len(req.tools or [])} finish={engine_finish}", flush=True)
    response = {
        "id": request_id, "object": "chat.completion", "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": parsed["finish_reason"]}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": n_out,
                  "total_tokens": len(ids) + n_out},
    }
    if cache_key is not None:
        response_cache_put(cache_key, response)
    return response


async def build_engine(args):
    """Construct AsyncLLM *inside* the running loop.

    ``AsyncLLM`` starts background tasks (its output handler) on the loop that
    is current at construction time. Built before ``uvicorn.run``, those tasks
    bind to a loop that never runs while uvicorn drives a different one — the
    server then never binds its port and the whole thing hangs silently with a
    reassuring "serving ..." line already printed. vLLM's own api_server builds
    the engine inside the async entrypoint for exactly this reason.
    """
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from heads_of_security.vllm_gates import plugin_available, register

    # AsyncLLM always runs EngineCore out-of-process, so VLLM_ENABLE_V1_MULTIPROCESSING
    # buys nothing here. What matters is how register() and the runner patch
    # reach that process and its workers.
    cuda_dirty = torch.cuda.is_initialized() or torch.cuda._is_in_bad_fork()
    if args.ungated:
        # No registration to inherit, so nothing *requires* fork -- and fork may not work
        # at all (the EngineCore child can come up already flagged bad-fork and die with
        # "Cannot re-initialize CUDA in forked subprocess", even when the parent is
        # clean). spawn is both safe and sufficient here.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    elif plugin_available():
        # vLLM runs register() itself in every process it creates (the
        # vllm.general_plugins entry point), so the start method is free and
        # spawn is the safe default: whether a forked EngineCore can
        # initialize CUDA is NOT decidable from the parent: is_initialized()
        # and _is_in_bad_fork() can both read False and the forked child still
        # die with "Cannot re-initialize CUDA in forked subprocess" -- torch
        # poisons fork for its children whenever the CUDA runtime was touched
        # without NVML (device_count() without pynvml), which leaves no trace
        # the parent can query. Spawn costs a slower engine start (the child
        # re-imports); gated serving is verified under both.
        # Escape hatch: set VLLM_WORKER_MULTIPROC_METHOD=fork explicitly.
        chosen = os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if chosen != "spawn" and cuda_dirty:
            print(f"VLLM_WORKER_MULTIPROC_METHOD={chosen} requested but this process "
                  "is not fork-clean; overriding to spawn", flush=True)
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        print(f"engine start method: {os.environ['VLLM_WORKER_MULTIPROC_METHOD']} "
              "(gated registration via the vllm.general_plugins entry point)", flush=True)
    else:
        # No entry-point metadata (PYTHONPATH=src checkout): the children see
        # register() and the runner patch only by fork inheritance, so fork is
        # mandatory and this process must be CUDA-clean. Fail with the cause
        # instead of a fork traceback from the EngineCore.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork", (
            "without the vllm.general_plugins entry point (pip install -e .) the "
            "gated server needs fork workers; spawn loses the registration")
        assert not cuda_dirty, (
            "this process cannot fork a CUDA-clean EngineCore: "
            f"cuda_initialized={torch.cuda.is_initialized()} "
            f"in_bad_fork={torch.cuda._is_in_bad_fork()}. Start the server from a "
            "fresh process (exec, not os.fork/multiprocessing from a CUDA-using "
            "parent), or `pip install -e .` so the vllm.general_plugins entry "
            "point makes registration spawn-safe")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if not args.ungated:
        register()
    engine_args = AsyncEngineArgs(
        model=args.ungated or args.snapshot,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.tensor_parallel_size > 1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Both of these exist ONLY for the channel tape: it is republished per step, so
        # CUDA-graph replay would not observe it, and channels are not a function of the
        # tokens so a shared prefix block would leak trust labels between requests.
        # Neither applies without gates.
        enforce_eager=not args.ungated,
        enable_prefix_caching=bool(args.ungated),
        dtype="bfloat16",
    )
    return AsyncLLM.from_engine_args(engine_args)


async def _serve() -> None:
    global ENGINE

    ENGINE = await build_engine(CFG)
    config = uvicorn.Config(app, host=CFG.host, port=CFG.port, log_level="warning")
    server = uvicorn.Server(config)
    print(f"engine ready; binding {CFG.host}:{CFG.port} ...", flush=True)
    await server.serve()


def main() -> None:
    global CFG, ENGINE, TOK
    ap = argparse.ArgumentParser()
    # --snapshot is the GATED path (a snapshot carrying gate tables). --ungated serves a
    # plain checkpoint through the same renderer, so a baseline sees byte-identical
    # prompts to the gated arm.
    ap.add_argument("--snapshot", default=None,
                    help="gated snapshot dir (mutually exclusive with --ungated)")
    ap.add_argument("--ungated", default=None, metavar="MODEL",
                    help="serve this plain model/dir with NO gates and no channel tape")
    # Role policy (see channel_tokenizer). A bare flag means explicitly-empty.
    ap.add_argument("--untrusted-inline-roles", nargs="*", default=None)
    ap.add_argument("--untrusted-header-roles", nargs="*", default=None)
    ap.add_argument("--role-remap", nargs="*", default=None, help="pairs like tool=input")
    ap.add_argument("--base-model", default="openai/gpt-oss-20b")
    ap.add_argument("--family", default="gptoss", choices=["gptoss", "llama"],
                    help="rendering + output-parsing family; llama renders "
                         "channels via LlamaGatedTokenizer and returns raw text")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--model-name", default="vllm_gated")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--date-string", default="2024-06-01")
    ap.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True,
                    help="ignore client temperature/top_p; "
                         "--no-greedy honors the client's sampling params")
    ap.add_argument("--memoize-responses", action="store_true",
                    help="cache completed responses by exact request and effective "
                         "generation settings; requires greedy decoding")
    CFG = ap.parse_args()
    if bool(CFG.snapshot) == bool(CFG.ungated):
        ap.error("give exactly one of --snapshot (gated) or --ungated (plain model)")
    if CFG.memoize_responses and not CFG.greedy:
        ap.error("--memoize-responses requires greedy decoding")
    if CFG.ungated:
        # The vllm.general_plugins entry point runs register() in EVERY vLLM process,
        # installing the channel runner patch; on a model with no gate tables that patch
        # refuses to serve ("ungated channel index unset"). Set before vLLM reads its env.
        os.environ["VLLM_PLUGINS"] = ""

    role_kwargs = {}
    if CFG.untrusted_inline_roles is not None:
        role_kwargs["untrusted_inline_roles"] = list(CFG.untrusted_inline_roles)
    if CFG.untrusted_header_roles is not None:
        role_kwargs["untrusted_header_roles"] = list(CFG.untrusted_header_roles)
    if CFG.role_remap:
        role_kwargs["role_remap"] = dict(kv.split("=", 1) for kv in CFG.role_remap)

    if CFG.family == "llama":
        from heads_of_security.attention_gating.gated_tokenizer import LlamaGatedTokenizer

        TOK = LlamaGatedTokenizer.from_pretrained(CFG.base_model, chat_template_channel=True, **role_kwargs)
    else:
        from heads_of_security.attention_gating.gpt_oss.gated_tokenizer import GptOssGatedTokenizer

        TOK = GptOssGatedTokenizer.from_pretrained(CFG.base_model, chat_template_channel=True, **role_kwargs)
    # The renderer above is pinned to 3 channels (scaffold separate). A snapshot
    # whose gates were trained with a different channel count would still serve:
    # a 2-channel table only dies later, at the first request, on the runner's
    # bounds check, and the mirror case (3-channel table, 2-channel tape) would
    # not fail at all -- it would silently apply the trusted column to scaffold
    # tokens, i.e. a different model from the one that was trained. Fail here,
    # before taking the GPU.
    #
    # Skipped under --ungated: there is no snapshot to read, and no gate table
    # whose channel count could disagree with the renderer.
    n_model = None
    if CFG.snapshot:
        snap_cfg = json.loads((Path(CFG.snapshot) / "config.json").read_text())
        n_model = int(snap_cfg.get("gating_num_channels", TOK.num_channels))
        assert n_model == TOK.num_channels, (
            f"{CFG.snapshot} has gating_num_channels={n_model} but this server renders "
            f"{TOK.num_channels} channels (chat_template_channel=True). Serve the "
            f"matching snapshot: the channel ids would not mean what the gates were "
            f"trained on.")
    print(f"serving {CFG.model_name} (+ -vanilla) on {CFG.host}:{CFG.port}"
          + (f" [{n_model}-channel gates]" if n_model else " [ungated]"), flush=True)
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
