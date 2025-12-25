"""Training utilities for xinSLM v08 (nanochat-style).

Keeps a single training loop for both stages:
- Base pretraining: labels predict all tokens.
- Chat SFT: labels are masked with IGNORE_INDEX (-100) except assistant tokens.

This module is intentionally minimal: no fancy logging frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class TrainConfig:
    device: str = "cpu"  # "cpu" | "cuda" | "mps" | "auto"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum: int = 1
    max_steps: int = 1000
    log_every: int = 50
    ckpt_every: int = 500
    out_path: str = "checkpoints/model.pt"
    resume_from: Optional[str] = None


def resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(path: str, model, optimizer, step: int, extra: Optional[Dict[str, Any]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train_loop(model, dataloader, cfg: TrainConfig) -> None:
    device = resolve_device(cfg.device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    start_step = 0
    if cfg.resume_from:
        ckpt = load_checkpoint(cfg.resume_from, model, optimizer)
        start_step = int(ckpt.get("step", 0))

    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, cfg.max_steps):
        loss_sum = 0.0

        for micro in range(cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, targets=labels)
            (loss / cfg.grad_accum).backward()
            loss_sum += float(loss.detach().cpu())

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % cfg.log_every == 0 or step == start_step:
            avg_loss = loss_sum / max(cfg.grad_accum, 1)
            print(f"step {step+1}/{cfg.max_steps} | loss {avg_loss:.4f}")

        if cfg.out_path and cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(cfg.out_path, model, optimizer, step + 1)

    if cfg.out_path:
        save_checkpoint(cfg.out_path, model, optimizer, cfg.max_steps)
