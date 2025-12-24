"""
xinSLM v08 — nanoChat-D4-inspired Transformer Model

Architecture features (from nanochat D4):
- RMSNorm (no learned affine parameters)
- RoPE (rotary position embeddings)
- Causal self-attention with GQA + QK normalization
- MLP with ReLU² activation
- No bias in linear layers
- Untied embedding and unembedding weights
- Logit softcap (tanh @ 15)

Reference: https://github.com/karpathy/nanochat (MIT License)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model configuration matching nanochat D4 defaults."""
    vocab_size: int = 65536
    n_layer: int = 4
    n_head: int = 2
    n_kv_head: int = 2  # GQA: if < n_head, uses grouped-query attention
    n_embd: int = 256
    context_len: int = 1024
    mlp_mult: float = 4.0  # MLP hidden dim = n_embd * mlp_mult
    logit_softcap: float = 15.0
    rope_theta: float = 10000.0
    eps: float = 1e-6  # RMSNorm epsilon

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        return int(self.n_embd * self.mlp_mult)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (no learned affine parameters).
    
    y = x / sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Precompute RoPE frequency tensor for complex exponentials.
    
    Returns: freqs_cis of shape [max_seq_len, dim//2] as complex64
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # Position indices
    t = torch.arange(max_seq_len, device=device)
    # Outer product: [seq_len, dim//2]
    freqs = torch.outer(t, freqs)
    # Complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * theta)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to x.
    
    x: [B, T, n_head, head_dim]
    freqs_cis: [T, head_dim//2] complex
    
    Returns: x with RoPE applied, same shape
    """
    B, T, H, D = x.shape
    # Reshape x to complex: [B, T, H, D//2, 2] -> [B, T, H, D//2] complex
    x_complex = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
    # Broadcast freqs_cis: [T, D//2] -> [1, T, 1, D//2]
    freqs_cis = freqs_cis[:T].unsqueeze(0).unsqueeze(2)
    # Apply rotation
    x_rotated = x_complex * freqs_cis
    # Convert back to real: [B, T, H, D//2, 2] -> [B, T, H, D]
    x_out = torch.view_as_real(x_rotated).reshape(B, T, H, D)
    return x_out.type_as(x)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with:
    - Grouped-Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - QK normalization
    - No bias in linear layers
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = self.n_head // self.n_kv_head  # repetition factor for GQA

        # Q/K/V projections (no bias)
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        # Output projection (no bias)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.eps = config.eps

    def _qk_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head RMS normalization for Q and K."""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # QK normalization
        q = self._qk_norm(q)
        k = self._qk_norm(k)

        # GQA: repeat K and V heads to match Q heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, T, self.n_kv_head, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.n_kv_head, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_head, self.head_dim)

        # Transpose for attention: [B, n_head, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # Using PyTorch's efficient implementation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.wo(y)


class MLP(nn.Module):
    """
    MLP with ReLU² activation (squared ReLU).
    
    h = relu(W1 @ x)^2
    out = W2 @ h
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.c_proj = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c_fc(x)
        h = F.relu(h).pow(2)  # ReLU²
        return self.c_proj(h)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm and residual connections.
    
    x = x + Attention(RMSNorm(x))
    x = x + MLP(RMSNorm(x))
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, eps=config.eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class NanoChatModel(nn.Module):
    """
    Full decoder-only Transformer model (nanochat D4 style).
    
    Features:
    - Token embeddings (untied from output)
    - N transformer blocks
    - Final RMSNorm
    - LM head with logit softcap
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (untied)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Optional initial RMSNorm (as shown in D4 diagram)
        self.embed_norm = RMSNorm(config.n_embd, eps=config.eps)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.n_embd, eps=config.eps)
        
        # LM head (untied from embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(config.head_dim, config.context_len, config.rope_theta),
            persistent=False
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale down residual projections
            if hasattr(module, '_is_residual'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logit softcap: cap * tanh(logits / cap)"""
        cap = self.config.logit_softcap
        return cap * torch.tanh(logits / cap)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] for computing loss (optional)
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.config.context_len, f"Sequence length {T} exceeds context_len {self.config.context_len}"

        # Token embeddings
        x = self.wte(idx)
        
        # Initial normalization
        x = self.embed_norm(x)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Final normalization
        x = self.final_norm(x)

        # LM head + softcap
        logits = self.lm_head(x)
        logits = self._softcap(logits)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100  # For masked tokens in chat SFT
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Conditioning sequence [B, T]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling threshold
            eos_token_id: If set, stop when this token is generated
            
        Returns:
            Generated sequence [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            idx_cond = idx if idx.size(1) <= self.config.context_len else idx[:, -self.config.context_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # [B, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    vocab_size: int = 65536,
    n_layer: int = 4,
    n_head: int = 2,
    n_kv_head: int = 2,
    n_embd: int = 256,
    context_len: int = 1024,
    **kwargs
) -> NanoChatModel:
    """Factory function to create a model with custom config."""
    config = ModelConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        context_len=context_len,
        **kwargs
    )
    return NanoChatModel(config)


# Quick test
if __name__ == "__main__":
    print("Testing NanoChatModel (D4 config)...")
    
    # Create model with D4 defaults
    config = ModelConfig()
    model = NanoChatModel(config)
    
    print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    B, T = 2, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    logits, loss = model(idx, targets)
    print(f"Input shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    print("\n✓ All tests passed!")
