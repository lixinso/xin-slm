"""Stage 2: Chat SFT (masked loss) for xinSLM v08.

Conceptually mirrors nanochat's chat_sft.py:
- load pretrained checkpoint
- train on conversation data
- compute loss on assistant tokens only (labels=-100 elsewhere)

Input format: JSONL where each line is:
  {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from model import ModelConfig, NanoChatModel
from tokenizer_utils import get_default_tokenizer
from data_utils import ChatSFTJsonlDataset, pad_collate
from train_utils import TrainConfig, train_loop


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to JSONL conversations")
    p.add_argument("--tokenizer", default="tiktoken", choices=["tiktoken", "byte"], help="Tokenizer backend")

    # Model (must match base pretraining when resuming)
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=256)

    # Training
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/chat.pt")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from (e.g., checkpoints/base.pt)")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    tokenizer = get_default_tokenizer(prefer=args.tokenizer)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_len=args.context_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
    )
    model = NanoChatModel(cfg)

    dataset = ChatSFTJsonlDataset(
        path=args.data,
        tokenizer=tokenizer,
        max_len=cfg.context_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )

    tcfg = TrainConfig(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        out_path=args.out,
        resume_from=args.resume,
    )

    print(
        f"[chat_sft] vocab={cfg.vocab_size} ctx={cfg.context_len} layers={cfg.n_layer} heads={cfg.n_head} embd={cfg.n_embd}"
    )
    print(f"[chat_sft] resume={tcfg.resume_from} device={tcfg.device} batch={args.batch_size} grad_accum={tcfg.grad_accum} steps={tcfg.max_steps}")

    train_loop(model, loader, tcfg)


if __name__ == "__main__":
    main()
