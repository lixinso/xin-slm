"""Chat inference for xinSLM v08.

Loads a checkpoint, formats a prompt using the same chat template as SFT, and
runs autoregressive generation.

Example:
  source .venv/bin/activate
  python chat_infer.py --checkpoint checkpoints/chat.pt --tokenizer byte --prompt "What is 2+2?" \
    --max_new_tokens 64 --temperature 0.8 --top_k 50

Note:
- The model config must match the checkpoint (n_layer/n_head/n_embd/context_len).
- For quick local sanity, using --tokenizer byte is fine.
"""

from __future__ import annotations

import argparse

import torch

from model import ModelConfig, NanoChatModel
from tokenizer_utils import get_default_tokenizer
from data_utils import ChatTemplate, render_messages_to_text
from train_utils import load_checkpoint, resolve_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--tokenizer", default="tiktoken", choices=["tiktoken", "byte"], help="Tokenizer backend")

    # Prompt
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--prompt", required=True, help="User prompt")

    # Model (must match training)
    p.add_argument("--context_len", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=256)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)

    # Runtime
    p.add_argument("--device", default="auto")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    device = resolve_device(args.device)
    tokenizer = get_default_tokenizer(prefer=args.tokenizer)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_len=args.context_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
    )

    model = NanoChatModel(cfg).to(device)
    model.eval()

    # Load weights
    load_checkpoint(args.checkpoint, model, optimizer=None)

    template = ChatTemplate()
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
        {"role": "assistant", "content": ""},
    ]
    prompt_text = render_messages_to_text(messages, template=template)

    input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=None,
    )

    # Decode only the generated continuation
    gen_ids = out[0].tolist()
    prompt_len = input_ids.size(1)
    continuation = tokenizer.decode(gen_ids[prompt_len:])

    print("\n=== Prompt ===\n")
    print(prompt_text)
    print("\n=== Model continuation ===\n")
    print(continuation)


if __name__ == "__main__":
    main()
