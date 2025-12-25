"""Datasets and collation for xinSLM v08.

Implements two data paths (nanochat-style):

Stage 1 (base pretraining):
- Raw text stream -> tokenize -> pack into fixed-length blocks.
- Loss on all tokens.

Stage 2 (chat SFT):
- Conversations (JSONL with messages) -> render template -> tokenize.
- Loss masked to assistant tokens only (labels=-100 elsewhere).

Both outputs are dicts with:
- input_ids: LongTensor [T] or [B,T]
- labels: LongTensor [T] or [B,T]

The model uses ignore_index=-100 for masked tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset, IterableDataset


IGNORE_INDEX = -100


@dataclass
class PackedTextConfig:
    seq_len: int
    text_field: str = "text"
    add_eos: bool = False


class PackedTextIterableDataset(IterableDataset):
    """Stream text (from an iterable of dicts or strings), tokenize and pack blocks."""

    def __init__(
        self,
        source: Union[Iterable, Callable[[], Iterable]],
        tokenizer,
        cfg: PackedTextConfig,
    ):
        super().__init__()
        self.source = source
        self.tokenizer = tokenizer
        self.cfg = cfg

    def _get_source_iter(self) -> Iterator:
        if callable(self.source):
            return iter(self.source())
        return iter(self.source)

    def _iter_text(self) -> Iterator[str]:
        for ex in self._get_source_iter():
            if isinstance(ex, str):
                text = ex
            elif isinstance(ex, dict):
                text = ex.get(self.cfg.text_field, "")
            else:
                text = getattr(ex, self.cfg.text_field, "")
            if text:
                yield text

    def __iter__(self):
        buf: List[int] = []
        for text in self._iter_text():
            ids = self.tokenizer.encode(text)
            if self.cfg.add_eos and hasattr(self.tokenizer, "eos_token_id") and getattr(self.tokenizer, "eos_token_id") is not None:
                ids = ids + [int(getattr(self.tokenizer, "eos_token_id"))]
            buf.extend(ids)

            # Need seq_len+1 to create shifted labels
            while len(buf) >= self.cfg.seq_len + 1:
                chunk = buf[: self.cfg.seq_len + 1]
                buf = buf[self.cfg.seq_len + 1 :]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


def read_text_lines(paths: Sequence[str]) -> Iterator[str]:
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip("\n")
                if line:
                    yield line


@dataclass
class ChatTemplate:
    """Minimal, stable chat template.

    Keep it dead simple so masking is straightforward.

    Format:
      <|role|>\n{content}\n
    """

    def render_prefix(self, role: str) -> str:
        return f"<|{role}|>\n"

    def render_suffix(self) -> str:
        return "\n"


def render_messages_to_text(messages: List[Dict[str, str]], template: Optional[ChatTemplate] = None) -> str:
    """Render messages to a single prompt string using the same template as SFT.

    This is used for inference to ensure formatting matches training.
    """

    template = template or ChatTemplate()
    out: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = msg.get("content") or ""
        out.append(template.render_prefix(role))
        out.append(content)
        out.append(template.render_suffix())
    return "".join(out)


def render_conversation_with_labels(
    messages: List[Dict[str, str]],
    tokenizer,
    max_len: int,
    template: Optional[ChatTemplate] = None,
) -> Dict[str, torch.Tensor]:
    """Render messages -> token ids + labels masked to assistant tokens.

    Returns tensors of length <= max_len.

    Labels are shifted for next-token prediction and use IGNORE_INDEX for:
    - all non-assistant tokens
    - the final token position (no next token)
    """

    template = template or ChatTemplate()

    token_ids: List[int] = []
    learn_mask: List[bool] = []

    def add_span(text: str, learn: bool):
        nonlocal token_ids, learn_mask
        ids = tokenizer.encode(text)
        token_ids.extend(ids)
        learn_mask.extend([learn] * len(ids))

    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = msg.get("content") or ""

        add_span(template.render_prefix(role), learn=False)
        add_span(content, learn=(role == "assistant"))
        add_span(template.render_suffix(), learn=False)

        if len(token_ids) >= max_len:
            token_ids = token_ids[:max_len]
            learn_mask = learn_mask[:max_len]
            break

    input_ids = torch.tensor(token_ids, dtype=torch.long)

    # Build labels with masking
    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    for i, learn in enumerate(learn_mask):
        if learn:
            labels[i] = input_ids[i]

    # Shift labels for next-token prediction
    if labels.numel() > 1:
        labels[:-1] = labels[1:]
    labels[-1] = IGNORE_INDEX

    return {"input_ids": input_ids, "labels": labels}


class ChatSFTJsonlDataset(Dataset):
    """Loads JSONL with schema: {"messages": [{"role":..., "content":...}, ...]}"""

    def __init__(
        self,
        path: str,
        tokenizer,
        max_len: int,
        messages_field: str = "messages",
        template: Optional[ChatTemplate] = None,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.messages_field = messages_field
        self.template = template or ChatTemplate()

        self._offsets: List[int] = []
        with open(self.path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(pos)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            raw = f.readline().decode("utf-8", errors="replace")
        ex = json.loads(raw)
        messages = ex[self.messages_field]
        return render_conversation_with_labels(messages, self.tokenizer, self.max_len, self.template)


def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences to max length in batch."""

    if not batch:
        raise ValueError("Empty batch")

    max_len = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), fill_value=IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        x = item["input_ids"]
        y = item["labels"]
        input_ids[i, : x.numel()] = x
        labels[i, : y.numel()] = y

    return {"input_ids": input_ids, "labels": labels}
