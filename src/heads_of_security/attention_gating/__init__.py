"""Channel-gated Llama and gpt-oss models (HuggingFace eager path)."""

from .gated_attention import LlamaAttentionGated
from .gated_model import (
    LlamaGatedConfig,
    LlamaGatedDecoderLayer,
    LlamaGatedForCausalLM,
    LlamaGatedModel,
)
from .gated_tokenizer import LlamaGatedTokenizer
from .gpt_oss import (
    GptOssAttentionGated,
    GptOssGatedConfig,
    GptOssGatedDecoderLayer,
    GptOssGatedForCausalLM,
    GptOssGatedModel,
    GptOssGatedTokenizer,
)

__all__ = [
    "GptOssAttentionGated",
    "GptOssGatedConfig",
    "GptOssGatedDecoderLayer",
    "GptOssGatedForCausalLM",
    "GptOssGatedModel",
    "GptOssGatedTokenizer",
    "LlamaAttentionGated",
    "LlamaGatedConfig",
    "LlamaGatedDecoderLayer",
    "LlamaGatedForCausalLM",
    "LlamaGatedModel",
    "LlamaGatedTokenizer",
]
