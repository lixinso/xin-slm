"""Tokenizer utilities for xinSLM v08 (nanochat-style pipeline).

Goals:
- Keep dependencies optional.
- Provide a working default tokenizer even without HuggingFace.

If `tiktoken` is available, `TiktokenTokenizer` provides a fast, battle-tested BPE.
Otherwise, `ByteTokenizer` is used as a dependency-free fallback.

Note: nanochat uses a ~65k vocab; matching that exactly usually requires a trained
BPE tokenizer. For quick local experiments, the fallback byte tokenizer (256) is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


class BaseTokenizer:
    vocab_size: int

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError


@dataclass
class ByteTokenizer(BaseTokenizer):
    """Dependency-free tokenizer: UTF-8 bytes (vocab_size=256)."""

    def __post_init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            text = str(text)
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        b = bytes(int(x) & 0xFF for x in ids)
        return b.decode("utf-8", errors="replace")


@dataclass
class TiktokenTokenizer(BaseTokenizer):
    """tiktoken-based tokenizer.

    Common encodings:
    - "gpt2" (50,257 vocab)
    - "cl100k_base" (~100k vocab)
    """

    encoding_name: str = "gpt2"

    def __post_init__(self):
        try:
            import tiktoken  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "tiktoken is not installed. Install it with `pip install tiktoken`."
            ) from e

        self._enc = tiktoken.get_encoding(self.encoding_name)
        self.vocab_size = int(self._enc.n_vocab)

    def encode(self, text: str) -> List[int]:
        return list(self._enc.encode(text))

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


def get_default_tokenizer(prefer: str = "tiktoken") -> BaseTokenizer:
    """Create a tokenizer with sensible defaults.

    prefer:
      - "tiktoken": tries tiktoken gpt2, falls back to ByteTokenizer
      - "byte": always ByteTokenizer
    """

    prefer = (prefer or "").lower()
    if prefer == "byte":
        return ByteTokenizer()

    # Default: try tiktoken, fallback to byte.
    try:
        return TiktokenTokenizer("gpt2")
    except Exception:
        return ByteTokenizer()
