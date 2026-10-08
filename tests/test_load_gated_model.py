"""Tests for ``load_gated_model``: the Hub wrapper over ``load_gated_model_local``.

The Hub and the model build are both stubbed: these check that the repo's
``gates_config.json`` picks the base model and that the role kwargs reach the
tokenizer policy under the right names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from heads_of_security import load_gated_model
from heads_of_security.attention_gating import model_utils


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch):
    """A fake Hub repo dir; ``hf_hub_download`` serves files from it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "gates.safetensors").write_bytes(b"")
    (repo / "gates_config.json").write_text(json.dumps({"base_model": "openai/gpt-oss-20b"}))

    downloads = []

    def fake_download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename))
        path = repo / filename
        if not path.exists():
            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        return str(path)

    calls = []

    def fake_local(gates, model, device="cuda:0", **kwargs):
        calls.append(dict(gates=gates, model=model, device=device, **kwargs))
        return "model", "tok"

    monkeypatch.setattr(model_utils, "hf_hub_download", fake_download)
    monkeypatch.setattr(model_utils, "load_gated_model_local", fake_local)
    return repo, downloads, calls


def test_base_model_comes_from_repo_config(fake_repo):
    repo, downloads, calls = fake_repo
    assert load_gated_model("imperial-cpg/hos-gpt-oss-20b", device="cpu") == ("model", "tok")
    assert {f for _, f in downloads} == {"gates.safetensors", "gates_config.json"}
    assert all(r == "imperial-cpg/hos-gpt-oss-20b" for r, _ in downloads)
    (call,) = calls
    assert call["gates"] == str(repo / "gates.safetensors")
    assert call["model"] == "openai/gpt-oss-20b"
    assert call["device"] == "cpu"
    # Defaults are left to the tokenizer.
    assert call["untrusted_header_roles"] is None
    assert call["untrusted_inline_roles"] is None


def test_untrusted_roles_map_to_header_roles(fake_repo):
    _, _, calls = fake_repo
    load_gated_model("r", untrusted_roles=["tool", "search"], untrusted_inline_roles=["input"])
    (call,) = calls
    assert call["untrusted_header_roles"] == ["tool", "search"]
    assert call["untrusted_inline_roles"] == ["input"]


def test_repo_without_config_is_rejected(fake_repo):
    repo, _, calls = fake_repo
    (repo / "gates_config.json").unlink()
    with pytest.raises(ValueError, match="not a heads-of-security gates repo"):
        load_gated_model("some/other-model")
    assert not calls
