# GPT-Style Language Model

This document outlines the architecture implemented in `train_gpt_language_model.py`, which is inspired by the nanoGPT project. The model demonstrates a minimal Transformer-based language model trained on WikiText-2 with a simple Byte-Pair Encoding (BPE) tokenizer.

## Architecture

- **BPE Tokenizer:** A toy Byte-Pair Encoding tokenizer learns character-level merges to map text into a small integer vocabulary.
- **Token & Position Embeddings:** Tokens are embedded into `N_EMBD`-dimensional vectors and summed with trainable positional embeddings.
- **Transformer Blocks:** A stack of `N_LAYER` decoder-only blocks, each containing:
  - Multi-head self-attention with `N_HEAD` heads
  - LayerNorm and residual connections
  - Feed-forward network with GELU activation
- **Final LayerNorm & Linear Head:** Normalizes the output of the last block before projecting to vocabulary logits for next-token prediction.

The training loop uses the AdamW optimizer with a small context length (`BLOCK_SIZE`) and reports loss on a held-out validation batch each epoch.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (OpenAI GPT‑2, 2019)
- [nanoGPT GitHub repository](https://github.com/karpathy/nanoGPT)
