"""Anchored source patching: build gated variants of vLLM modules at import.

The channel gate needs three additions *inside* vLLM's unified triton attention
kernel (a per-(q-channel, k-channel, head) log-bias on the scores) and a few
lines of plumbing in the backend that calls it. Vendoring the ~1200-line kernel
would freeze us to one vLLM snapshot silently; instead the installed source is
read, patched through exact-match anchors, and exec'd into a fresh module. Every
anchor must match exactly once — a vLLM upgrade that moves the code fails loudly
here instead of drifting.

Pinned against vllm 0.27.x; ``assert_vllm_version`` guards the import.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

SUPPORTED_VLLM = ("0.27",)


def assert_vllm_version() -> None:
    import vllm

    if not any(vllm.__version__.startswith(v) for v in SUPPORTED_VLLM):
        raise RuntimeError(
            f"heads_of_security.vllm_gates is pinned to vllm {SUPPORTED_VLLM}, found "
            f"{vllm.__version__}. Re-verify every anchor in _patching.py "
            f"(and the parity tests) before lifting the pin."
        )


def patched_module(source_module: str, new_name: str, patches: list[tuple[str, str]]) -> ModuleType:
    """Load ``source_module``'s source, apply (anchor, replacement) pairs, import.

    Each ``anchor`` must occur exactly once. The patched source is materialized
    to a real file (keyed by content hash) because Triton's ``@jit`` insists on
    ``inspect``-able sources; it is then imported under ``new_name`` and
    registered in ``sys.modules``.
    """
    import vllm

    spec = importlib.util.find_spec(source_module)
    assert spec is not None and spec.origin is not None, source_module
    with open(spec.origin) as f:
        src = f.read()

    for anchor, replacement in patches:
        n = src.count(anchor)
        if n != 1:
            raise RuntimeError(
                f"anchor matched {n} times (need exactly 1) in {source_module}:\n{anchor}"
            )
        src = src.replace(anchor, replacement)

    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    cache_dir = Path(
        os.environ.get("HEADS_OF_SECURITY_VLLM_GATES_CACHE", Path.home() / ".cache" / "heads_of_security_vllm_gates")
    ) / f"vllm-{vllm.__version__}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{new_name.rsplit('.', 1)[-1]}-{digest}.py"
    if not path.exists():
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(src)
        tmp.replace(path)  # atomic; content-hashed name makes races benign

    file_spec = importlib.util.spec_from_file_location(new_name, path)
    assert file_spec is not None and file_spec.loader is not None
    module = importlib.util.module_from_spec(file_spec)
    sys.modules[new_name] = module
    file_spec.loader.exec_module(module)
    return module
