# xinSLM v08 — Training Pipeline (nanochat-style)

This folder implements a minimal **two-stage** pipeline (same model, different data):

1) **Base pretraining** (raw text, loss on all tokens)
2) **Chat SFT** (conversations, loss only on assistant tokens via masking)

This mirrors the conceptual flow of https://github.com/karpathy/nanochat (MIT License).

---

## 1) High-level overview

### Diagram (Mermaid)

```mermaid
flowchart TB
  %% =========================
  %% Styling
  %% =========================
  classDef data fill:#eef2ff,stroke:#4f46e5,stroke-width:1px,color:#111827;
  classDef file fill:#ecfeff,stroke:#0891b2,stroke-width:1px,color:#0f172a;
  classDef code fill:#f0fdf4,stroke:#16a34a,stroke-width:1px,color:#052e16;
  classDef artifact fill:#fff7ed,stroke:#f97316,stroke-width:1px,color:#7c2d12;
  classDef model fill:#fdf4ff,stroke:#a855f7,stroke-width:1px,color:#3b0764;

  %% =========================
  %% Stage 1: Base pretraining
  %% =========================
  subgraph S1[Stage 1 — Base pretraining (raw text, loss on all tokens)]
    direction TB

    T[(example_data_text.txt\n(or your .txt files))]:::data
    S1SCRIPT[base_train.py]:::code
    TOK1[tokenizer_utils.py\nByteTokenizer | TiktokenTokenizer]:::code
    PACK[PackedTextIterableDataset\n(pack token stream into blocks)]:::code
    XY1[(Batch\ninput_ids, labels\nlabels = next-token targets for ALL tokens)]:::file
    MODEL[model.py\nNanoChatModel]:::model
    TRAIN[train_utils.py\ntrain_loop()]:::code
    CKPT[(checkpoints/base.pt)]:::artifact

    T --> S1SCRIPT
    S1SCRIPT --> TOK1
    TOK1 --> PACK
    PACK --> XY1
    XY1 --> TRAIN
    MODEL --> TRAIN
    TRAIN --> CKPT
  end

  %% =========================
  %% Stage 2: Chat SFT
  %% =========================
  subgraph S2[Stage 2 — Chat SFT (conversations, loss on assistant only)]
    direction TB

    J[(example_data_chat.jsonl\n(or your chat JSONL))]:::data
    S2SCRIPT[chat_sft.py]:::code
    RENDER[render_conversation_with_labels()\n+ ChatTemplate\n<|role|>\\n{content}\\n]:::code
    TOK2[tokenizer_utils.py\nByteTokenizer | TiktokenTokenizer]:::code
    MASK[(Build labels mask\nlabels=-100 except assistant tokens)]:::file
    SHIFT[(Shift labels for next-token prediction)]:::file
    XY2[(Batch\ninput_ids, labels\nlabels masked w/ -100)]:::file
    MODEL2[model.py\nNanoChatModel]:::model
    TRAIN2[train_utils.py\ntrain_loop()]:::code
    CKPT2[(checkpoints/chat.pt)]:::artifact

    J --> S2SCRIPT
    S2SCRIPT --> RENDER
    RENDER --> TOK2
    TOK2 --> MASK
    MASK --> SHIFT
    SHIFT --> XY2
    XY2 --> TRAIN2
    MODEL2 --> TRAIN2
    TRAIN2 --> CKPT2
  end

  %% Resume edge between stages
  CKPT -. --resume checkpoints/base.pt--> S2SCRIPT
```

### Diagram (ASCII)

```
Stage 1: base pretraining (raw text)

  .txt -> tokenize -> pack blocks -> (input_ids, labels) -> train -> checkpoints/base.pt
                                              labels = next-token targets for ALL tokens


Stage 2: chat SFT (assistant-only)

  .jsonl(messages) -> render template -> tokenize -> build labels mask -> train -> checkpoints/chat.pt
                                            labels = -100 except assistant tokens
                                            (then shifted for next-token prediction)

  checkpoints/base.pt --resume--> continues training for SFT
```

```
Stage 1: PRETRAIN (raw text)                 Stage 2: SFT (chat)

text files (.txt)                            JSONL conversations
  -> tokenize                                  -> render template
  -> pack blocks                               -> tokenize
  -> predict all tokens                        -> mask labels to assistant
  -> save base checkpoint                      -> resume from base checkpoint

base_train.py  --------------------------->   chat_sft.py
(checkpoints/base.pt)                         (checkpoints/chat.pt)
```

---

## 2) Files and responsibilities

- `model.py`
  - `NanoChatModel`: decoder-only Transformer (RMSNorm + RoPE + GQA + QK norm + ReLU² + logit softcap)
  - forward signature: `model(input_ids, targets=labels)`
  - masked tokens use `ignore_index = -100`

- `tokenizer_utils.py`
  - `ByteTokenizer` (dependency-free, vocab_size=256)
  - `TiktokenTokenizer` (optional; requires `tiktoken`)
  - `get_default_tokenizer()` chooses tokenizer based on `--tokenizer`

- `data_utils.py`
  - Stage 1 dataset: `PackedTextIterableDataset` (packs token stream into fixed blocks)
  - Stage 2 dataset: `ChatSFTJsonlDataset` (loads JSONL conversations)
  - `render_conversation_with_labels()` builds `(input_ids, labels)` with assistant-only learning
  - `pad_collate()` pads variable-length chat samples

- `train_utils.py`
  - `train_loop()` shared training loop for both stages
  - checkpoint save/load (`out_path`, `resume_from`)

- `base_train.py`
  - CLI entry for Stage 1

- `chat_sft.py`
  - CLI entry for Stage 2

---

## 3) Stage 1: Base pretraining (raw text)

### Input format
One or more UTF-8 text files. In quickstart this is `example_data_text.txt`.

### Data flow
1) Read lines from `--text` paths
2) Tokenize each line
3) Concatenate into a token stream
4) Pack into blocks of length `context_len + 1`
5) Create:
   - `input_ids = chunk[:-1]`
   - `labels = chunk[1:]`

So the model learns standard next-token prediction on *all tokens*.

### Output
A checkpoint written to `--out` (default `checkpoints/base.pt`).

---

## 4) Stage 2: Chat SFT (assistant-only masked loss)

### Input format (JSONL)
Each line is a JSON object:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Chat template
A minimal, stable formatting is used (see `ChatTemplate` in `data_utils.py`):

```
<|role|>\n{content}\n
```

### Label masking rule
- tokens originating from `role == "assistant"` are **learned**
- everything else (system/user + role headers + separators) is **masked** by setting labels to `-100`

Then labels are **shifted** for next-token prediction.

### Output
A checkpoint written to `--out` (default `checkpoints/chat.pt`).

---

## 5) Checkpoints and resume behavior

- `train_utils.load_checkpoint()` loads:
  - `model` weights
  - `optimizer` state (if present)
  - `step`

- For nanochat-style staging:
  - run Stage 1 to create `checkpoints/base.pt`
  - run Stage 2 with `--resume checkpoints/base.pt`

Important: the Stage 2 model config (layers/heads/embd/context/vocab) must match the config used in Stage 1.

---

## 6) Minimal local runs

### Stage 1
```bash
python3 base_train.py \
  --text example_data_text.txt \
  --tokenizer byte \
  --context_len 128 \
  --max_steps 50 \
  --out checkpoints/base.pt
```

### Stage 2
```bash
python3 chat_sft.py \
  --data example_data_chat.jsonl \
  --tokenizer byte \
  --context_len 128 \
  --resume checkpoints/base.pt \
  --max_steps 50 \
  --out checkpoints/chat.pt
```

If you install `tiktoken` (`pip install tiktoken`), you can replace `--tokenizer byte` with `--tokenizer tiktoken`.

---

## 7) Notes / limitations

- Running requires PyTorch (`torch`) installed in your environment.
- The `ByteTokenizer` is great for sanity checks but not competitive; real training typically uses a trained BPE tokenizer.
- This v08 pipeline is intentionally minimal and does not include evaluation tasks (MMLU/GSM8K/etc.) yet.

---

## 8) Reference and license

nanochat (MIT License):
- https://github.com/karpathy/nanochat
- https://github.com/karpathy/nanochat/blob/master/LICENSE
