"""
Rotary Positional Embedding (RoPE) implementation for SLM
Based on Llama 3.2 specifications with support for long context scaling
"""
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation
    Supports both standard and scaled versions for long context
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 128000,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Calculate inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build rotary embedding table
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Pre-compute cos and sin values for efficiency"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # Apply scaling for long context
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin embeddings for the given sequence length
        
        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length (if None, uses x.shape[-2])
            
        Returns:
            cos, sin tensors of shape [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # See if we need to extend the cache
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        position_ids: Position indices (optional)
        
    Returns:
        Rotated query and key tensors
    """
    # Handle position_ids if provided
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEScaledRotaryEmbedding(RotaryEmbedding):
    """
    Scaled RoPE for handling very long contexts
    Implements the RoPE-2θ scaling mentioned in Llama 3.2
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 128000,
        base: float = 10000.0,
        scaling_type: str = "linear",  # "linear" or "dynamic"
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.scaling_type = scaling_type
        
        if scaling_type == "linear":
            # Linear scaling: increase base frequency
            base = base * scaling_factor
        elif scaling_type == "dynamic":
            # Dynamic scaling: will be applied during forward pass
            pass
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")
        
        super().__init__(dim, max_position_embeddings, base, scaling_factor, device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Pre-compute cos and sin values with scaling"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        if self.scaling_type == "dynamic":
            # Apply dynamic scaling based on sequence length
            if seq_len > self.max_position_embeddings:
                scaling_factor = seq_len / self.max_position_embeddings
                t = t / scaling_factor
        
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)