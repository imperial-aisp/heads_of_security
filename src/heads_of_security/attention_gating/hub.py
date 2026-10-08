"""Publish a gates file as a Hub repo that ``load_gated_model(repo_id)`` can load.

    from heads_of_security.attention_gating.hub import push_gates
    push_gates("gates/gpt-oss-20b/lr0.1/gates.safetensors", "imperial-cpg/hos-gpt-oss-20b",
               base_model="openai/gpt-oss-20b", training={"method": "dpo", "lr": 0.1, "epochs": 1})

The repo gets ``gates.safetensors`` (copied as is), ``gates_config.json`` (the base
model, plus the gating shape and training info for readers), and a model card whose
``base_model`` front-matter links it to the base model on the Hub.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from safetensors.torch import load_file

from heads_of_security.attention_gating.gating_core import infer_gating_from_state_dict
from heads_of_security.attention_gating.model_utils import GATES_CONFIG_FILE, GATES_FILE

_CARD = """---
base_model: {base_model}
library_name: heads-of-security
tags:
- prompt-injection
- attention-gating
---

# {name}

Trained attention gates for [`{base_model}`](https://huggingface.co/{base_model}). The gates
bias attention between trusted, untrusted (tool results) and chat-template tokens, so the
model attends less to injected instructions. Only the gate tensors ({n_params:,} parameters)
are stored here; the base weights are loaded from `{base_model}`.

```python
from heads_of_security import load_gated_model
model, tok = load_gated_model("{repo_id}")
```

Gating: {num_layers} layers x {num_heads} heads, {num_channels} channels, sink gates: {sink}.
Training: {training}.
"""


def push_gates(
    gates_path: str | Path,
    repo_id: str,
    base_model: str,
    *,
    training: Optional[dict] = None,
    private: bool = True,
) -> str:
    """Create (or update) ``repo_id`` with the gates and their config. Returns the repo URL."""
    gates = load_file(str(gates_path))
    inferred = infer_gating_from_state_dict(gates)
    config = {
        "base_model": base_model,
        "gating_num_channels": inferred["gating_num_channels"],
        "gating_sink": bool(inferred.get("gating_sink", False)),
        "num_hidden_layers": inferred["num_hidden_layers"],
        "num_attention_heads": inferred["num_attention_heads"],
        "training": training or {},
    }
    card = _CARD.format(
        base_model=base_model,
        name=repo_id.split("/")[-1],
        repo_id=repo_id,
        n_params=sum(t.numel() for t in gates.values()),
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        num_channels=config["gating_num_channels"],
        sink=config["gating_sink"],
        training=json.dumps(config["training"]),
    )

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copyfile(gates_path, tmp / GATES_FILE)
        (tmp / GATES_CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
        (tmp / "README.md").write_text(card)
        api.upload_folder(repo_id=repo_id, folder_path=str(tmp), commit_message="Upload gates")
    return f"https://huggingface.co/{repo_id}"
