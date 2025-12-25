"""Stage 1: Base pretraining (raw text) for xinSLM v08.

This mirrors nanochat's base_train.py at a conceptual level:
- tokenize raw text
- pack into fixed-length blocks
- train next-token prediction on all tokens

For quick local runs, point --text to one or more .txt files.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from model import ModelConfig, NanoChatModel
from tokenizer_utils import get_default_tokenizer
from data_utils import PackedTextConfig, PackedTextIterableDataset, read_text_lines
from train_utils import TrainConfig, train_loop


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--text", nargs="+", required=True, help="One or more UTF-8 text files for pretraining")
    p.add_argument("--tokenizer", default="tiktoken", choices=["tiktoken", "byte"], help="Tokenizer backend")

    # Model
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=256)

    # Training
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/base.pt")
    p.add_argument("--resume", default=None)

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

    source_factory = lambda: read_text_lines(args.text)
    ds_cfg = PackedTextConfig(seq_len=cfg.context_len, text_field="text")
    dataset = PackedTextIterableDataset(source=source_factory, tokenizer=tokenizer, cfg=ds_cfg)

    loader = DataLoader(dataset, batch_size=args.batch_size)

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
        f"[base_train] vocab={cfg.vocab_size} ctx={cfg.context_len} layers={cfg.n_layer} heads={cfg.n_head} embd={cfg.n_embd}"
    )
    print(f"[base_train] device={tcfg.device} batch={args.batch_size} grad_accum={tcfg.grad_accum} steps={tcfg.max_steps}")

    train_loop(model, loader, tcfg)


if __name__ == "__main__":
    main()
