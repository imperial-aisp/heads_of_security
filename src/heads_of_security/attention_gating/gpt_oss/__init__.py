"""Channel-gated gpt-oss (``openai/gpt-oss-20b`` / ``-120b``).

qk-gating only. See :mod:`.gated_attention` for how the gate interacts with
gpt-oss's attention sinks and per-layer sliding-window masks.
"""

from .gated_attention import GptOssAttentionGated
from .gated_model import (
    GptOssGatedConfig,
    GptOssGatedDecoderLayer,
    GptOssGatedForCausalLM,
    GptOssGatedModel,
)
from .gated_tokenizer import GptOssGatedTokenizer

__all__ = [
    "GptOssAttentionGated",
    "GptOssGatedConfig",
    "GptOssGatedDecoderLayer",
    "GptOssGatedForCausalLM",
    "GptOssGatedModel",
    "GptOssGatedTokenizer",
]
