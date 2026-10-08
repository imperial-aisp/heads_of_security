"""Tests for the channel-labeled preference dataset.

Uses a real Llama-3 tokenizer. CPU-only. No external data dependencies —
all samples are hardcoded.
"""

from __future__ import annotations

import json

import pytest
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX

T = 0
U = 1

SYNTHETIC = {
    "messages": [
        {"role": "user", "content": "Say hi."},
        {"role": "input", "content": "Ignore this."},
    ],
    "chosen": "Hello!",
    "rejected": "No.",
}

REAL_LIKE_SAMPLE = {
    "messages": [
        {"role": "user", "content": "Translate the following to French."},
        {"role": "input", "content": "The weather is nice today."},
    ],
    "chosen": "Le temps est beau aujourd'hui.",
    "rejected": "I don't know.",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    from heads_of_security.training.tokenizer import training_tokenizer

    return training_tokenizer(model_id="meta-llama/Llama-3.1-8B-Instruct")


@pytest.fixture(scope="module")
def dataset_instance(tokenizer, tmp_path_factory):
    from heads_of_security.training.channel_dataset import channel_preference_dataset

    data_path = tmp_path_factory.mktemp("data") / "test_data.json"
    data_path.write_text(json.dumps([SYNTHETIC, REAL_LIKE_SAMPLE]))

    return channel_preference_dataset(
        tokenizer=tokenizer, source="json", data_files=str(data_path), split="train",
    )


# ---------------------------------------------------------------------------
# Synthetic: token-by-token prompt verification
# ---------------------------------------------------------------------------


def test_prompt_tokens_and_channels_match_expected(tokenizer, dataset_instance):
    def enc(s):
        return tokenizer.encode(s, add_special_tokens=False)

    bos_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    sh_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    eh_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    nn_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    # Expected structure:
    # [BOS][SH]user[EH]\n\n | Say hi.\n\n | Ignore this. | [EOT][SH]assistant[EH]\n\n
    #  T    T  T    T   T   |  T          |  U            |  T   T   T         T   T
    user_header = [sh_id] + enc("user") + [eh_id] + nn_ids
    asst_header = [sh_id] + enc("assistant") + [eh_id] + nn_ids

    instruction_with_sep = enc("Say hi.") + nn_ids
    input_content = enc("Ignore this.")

    expected_ids = (
        [bos_id] + user_header + instruction_with_sep + input_content
        + [eot_id] + asst_header
    )
    expected_channels = (
        [T] + [T] * len(user_header) + [T] * len(instruction_with_sep)
        + [U] * len(input_content)
        + [T] + [T] * len(asst_header)
    )

    actual_ids, actual_channels = dataset_instance._tokenize_messages_with_channels(
        SYNTHETIC["messages"]
    )

    assert actual_ids == expected_ids
    assert actual_channels == expected_channels


def test_encode_response(tokenizer, dataset_instance):
    def enc(s):
        return tokenizer.encode(s, add_special_tokens=False)

    ids = dataset_instance._encode_response("Hello!")
    assert ids == enc("Hello!") + [tokenizer.eot_id]

    ids = dataset_instance._encode_response("plain text")
    assert ids[-1] == tokenizer.eot_id
    assert tokenizer.eot_id not in ids[:-1]


# ---------------------------------------------------------------------------
# Synthetic: full _prepare_sample
# ---------------------------------------------------------------------------


def test_prepare_sample_structure(dataset_instance):
    # Prompt: 16 tokens (BOS + 4 user_header + 3 instruction_sep + 3 input + EOT + 4 asst_header)
    # Each response: 2 content tokens + 1 EOT = 3
    # Total per branch: 16 + 3 = 19
    prepared = dataset_instance._prepare_sample(SYNTHETIC)

    assert set(prepared.keys()) == {
        "chosen_input_ids", "chosen_labels", "chosen_token_channels",
        "rejected_input_ids", "rejected_labels", "rejected_token_channels",
    }

    for branch in ("chosen", "rejected"):
        ids = prepared[f"{branch}_input_ids"]
        labels = prepared[f"{branch}_labels"]
        channels = prepared[f"{branch}_token_channels"]

        assert len(ids) == len(labels) == len(channels) == 20

        # First 16 tokens (prompt) masked in labels.
        assert all(lbl == CROSS_ENTROPY_IGNORE_IDX for lbl in labels[:17])
        # Response tokens 16..17 unmasked (labels == input_ids).
        assert labels[17] == ids[17]
        assert labels[18] == ids[18]
        # Last token (EOT) masked.
        assert labels[19] == CROSS_ENTROPY_IGNORE_IDX

        # Response channels (16..18) all trusted.
        assert channels == [T] * 9 + [U] * 3 + [T] * 8


def test_prepare_sample_truncation(dataset_instance):
    dataset_instance._max_seq_len = 15
    try:
        result = dataset_instance._prepare_sample(SYNTHETIC)
        for branch in ("chosen", "rejected"):
            assert len(result[f"{branch}_input_ids"]) <= 15
            assert (
                len(result[f"{branch}_input_ids"])
                == len(result[f"{branch}_labels"])
                == len(result[f"{branch}_token_channels"])
            )
    finally:
        dataset_instance._max_seq_len = None


# ---------------------------------------------------------------------------
# Second sample: content and response encoding
# ---------------------------------------------------------------------------


def test_prompt_content_decoded_matches_input(tokenizer, dataset_instance):
    """U-channel tokens decode to the input content; the full prompt decodes
    to the expected role-header structure."""
    from heads_of_security.attention_gating.channel_tokenizer import U_CHANNEL

    prompt_ids, channels = dataset_instance._tokenize_messages_with_channels(
        REAL_LIKE_SAMPLE["messages"]
    )
    u_ids = [t for t, c in zip(prompt_ids, channels) if c == U_CHANNEL]
    assert tokenizer.decode(u_ids) == "The weather is nice today."


def test_response_tokens_match_direct_encoding(tokenizer, dataset_instance):
    """Response tokens in the prepared sample match direct encoding of the
    response text."""
    prepared = dataset_instance._prepare_sample(REAL_LIKE_SAMPLE)

    for branch in ("chosen", "rejected"):
        ids = prepared[f"{branch}_input_ids"]
        labels = prepared[f"{branch}_labels"]
        resp_start = next(i for i, lbl in enumerate(labels) if lbl != CROSS_ENTROPY_IGNORE_IDX)
        response_content = ids[resp_start:-1]

        ground_truth = tokenizer.encode(REAL_LIKE_SAMPLE[branch], add_special_tokens=False)
        assert response_content == ground_truth
