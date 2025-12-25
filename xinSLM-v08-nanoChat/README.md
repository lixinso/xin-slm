# xinSLM v08 — nanoChat-inspired

## References
- https://github.com/karpathy/nanochat (MIT License)

## Notes
- Model architecture: `model_architecture_of_nanochat.md`

## Quick start (local sanity checks)

### Stage 1: base pretraining on raw text

```bash
python3 base_train.py \
	--text example_data_text.txt \
	--tokenizer byte \
	--context_len 128 \
	--max_steps 50 \
	--out checkpoints/base.pt
```

### Stage 2: chat SFT on JSONL conversations

```bash
python3 chat_sft.py \
	--data example_data_chat.jsonl \
	--tokenizer byte \
	--context_len 128 \
	--resume checkpoints/base.pt \
	--max_steps 50 \
	--out checkpoints/chat.pt
```

If you have `tiktoken` installed, replace `--tokenizer byte` with `--tokenizer tiktoken`.