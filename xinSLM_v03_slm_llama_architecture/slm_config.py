"""
Configuration for Small Language Model based on Llama 3.2 1B architecture
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SLMConfig:
    """Configuration class for SLM model based on Llama 3.2 1B specifications"""
    
    # Model architecture
    vocab_size: int = 128256  # Expanded vocabulary size like Llama 3.2
    hidden_size: int = 2048   # Hidden dimension
    intermediate_size: int = 8192  # FFN intermediate size (4x hidden)
    num_hidden_layers: int = 16    # Number of decoder layers
    num_attention_heads: int = 32  # Number of attention heads
    num_key_value_heads: int = 8   # Number of key-value heads for GQA
    max_position_embeddings: int = 128000  # Max sequence length
    rope_base: float = 10000.0     # RoPE base frequency
    rope_scaling: Optional[dict] = None  # RoPE scaling configuration
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 2000
    
    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    
    # Initialization
    initializer_range: float = 0.02
    
    # Architecture specifics
    use_cache: bool = True
    tie_word_embeddings: bool = True  # Share input/output embeddings
    rms_norm_eps: float = 1e-6
    output_attentions: bool = False
    output_hidden_states: bool = False
    
    # Distillation
    use_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_alpha: float = 0.5
    distillation_temperature: float = 3.0
    
    # Training specifics
    batch_size: int = 64
    sequence_length: int = 8192  # Start with 8K context
    gradient_accumulation_steps: int = 1
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"
        
        # Set head dimensions
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads