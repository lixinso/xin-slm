"""
RLHF Configuration for SLM v04 - InstructGPT Style Implementation
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path to import SLM components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from slm_config import SLMConfig

@dataclass
class RLHFConfig:
    """Configuration for RLHF training following InstructGPT methodology"""
    
    # Base model configuration (inherits from SLM v03)
    base_model_config: SLMConfig = None
    
    # === SFT (Supervised Fine-tuning) Configuration ===
    sft_learning_rate: float = 1e-5
    sft_epochs: int = 3
    sft_batch_size: int = 16
    sft_max_length: int = 512
    sft_gradient_accumulation_steps: int = 2
    
    # === Reward Model Configuration ===
    reward_model_learning_rate: float = 9e-6
    reward_model_epochs: int = 1
    reward_model_batch_size: int = 64
    reward_model_max_length: int = 512
    reward_model_gradient_accumulation_steps: int = 1
    
    # Reward model architecture
    reward_dropout: float = 0.0
    reward_hidden_size: int = 512  # For reward head
    
    # === PPO Configuration ===
    ppo_learning_rate: float = 1.41e-5
    ppo_batch_size: int = 512  # Total batch size for PPO
    ppo_mini_batch_size: int = 64  # Mini-batch size for gradient updates
    ppo_epochs: int = 4  # PPO epochs per batch
    ppo_max_length: int = 512
    
    # PPO hyperparameters (from InstructGPT)
    ppo_clip_range: float = 0.2
    ppo_clip_range_vf: Optional[float] = None
    ppo_gamma: float = 1.0  # Discount factor
    ppo_gae_lambda: float = 0.95  # GAE lambda
    ppo_entropy_coeff: float = 0.01  # Entropy bonus
    ppo_value_loss_coeff: float = 0.5  # Value loss coefficient
    ppo_max_grad_norm: float = 0.5  # Gradient clipping
    
    # KL divergence control (key to InstructGPT success)
    kl_coeff: float = 0.2  # KL penalty coefficient
    adaptive_kl: bool = True  # Adaptive KL coefficient
    target_kl: float = 0.1  # Target KL divergence
    kl_coeff_max: float = 2.0  # Maximum KL coefficient
    kl_coeff_min: float = 0.02  # Minimum KL coefficient
    
    # === Training Configuration ===
    max_train_steps: int = 20000
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10
    warmup_steps: int = 100
    
    # === Data Configuration ===
    sft_data_path: str = "data/sft_data.jsonl"
    reward_data_path: str = "data/reward_data.jsonl"
    ppo_prompts_path: str = "data/ppo_prompts.jsonl"
    
    # Data processing
    response_template: str = "\n\nAssistant: "
    instruction_template: str = "Human: "
    
    # === Evaluation Configuration ===
    eval_batch_size: int = 32
    generation_max_length: int = 256
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True
    
    # === Hardware Configuration ===
    device: str = "cpu"  # Use CPU for compatibility (change to "mps" if you want Apple Silicon)
    mixed_precision: bool = False  # Disabled for CPU
    dataloader_num_workers: int = 0  # 0 for CPU compatibility
    
    # === Safety and Monitoring ===
    log_with_wandb: bool = False
    project_name: str = "slm-v04-rlhf"
    run_name: Optional[str] = None
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    def __post_init__(self):
        """Initialize base model config if not provided"""
        if self.base_model_config is None:
            # Use optimized SLM v03 configuration as base
            self.base_model_config = SLMConfig(
                vocab_size=32000,
                hidden_size=384,
                intermediate_size=1536,
                num_hidden_layers=6,
                num_attention_heads=6,
                num_key_value_heads=2,
                max_position_embeddings=512,
                
                # Training parameters optimized for RLHF
                learning_rate=self.sft_learning_rate,
                weight_decay=0.1,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                gradient_clip_norm=1.0,
                warmup_steps=self.warmup_steps,
                
                batch_size=self.sft_batch_size,
                sequence_length=self.sft_max_length,
                gradient_accumulation_steps=self.sft_gradient_accumulation_steps,
                
                save_steps=self.save_steps,
                eval_steps=self.eval_steps,
                logging_steps=self.logging_steps,
                
                attention_dropout=0.0,  # Reduced for stability
                hidden_dropout=0.0,
                
                use_cache=True,
                tie_word_embeddings=True,
                rms_norm_eps=1e-6,
                output_attentions=False,
                output_hidden_states=False,
                
                use_distillation=False,
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, 'to_dict'):
                config_dict[field_name] = field_value.to_dict()
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLHFConfig":
        """Create config from dictionary"""
        if 'base_model_config' in config_dict and config_dict['base_model_config'] is not None:
            config_dict['base_model_config'] = SLMConfig(**config_dict['base_model_config'])
        return cls(**config_dict)

def get_default_rlhf_config() -> RLHFConfig:
    """Get default RLHF configuration optimized for SLM v04"""
    return RLHFConfig()

def get_fast_rlhf_config() -> RLHFConfig:
    """Get fast RLHF configuration for development/testing"""
    config = RLHFConfig()
    
    # Reduce training steps for faster iteration
    config.sft_epochs = 1
    config.reward_model_epochs = 1
    config.max_train_steps = 1000
    config.save_steps = 100
    config.eval_steps = 50
    
    # Smaller batch sizes for development
    config.sft_batch_size = 8
    config.reward_model_batch_size = 32
    config.ppo_batch_size = 128
    config.ppo_mini_batch_size = 32
    
    return config