"""
Reward Model Implementation for RLHF - InstructGPT Style
Learns human preferences using Bradley-Terry model on pairwise comparisons
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path to import SLM components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from model import SLMModel


class RewardModel(nn.Module):
    """
    Reward Model for RLHF following InstructGPT methodology.
    
    Takes the base SLM model and adds a scalar reward head.
    Trained using Bradley-Terry model on human preference comparisons.
    """
    
    def __init__(self, base_model: SLMModel, config):
        super().__init__()
        self.config = config
        self.transformer = base_model
        
        # Freeze transformer parameters during reward training (optional)
        # In practice, fine-tuning the whole model often works better
        self.freeze_transformer = False
        
        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Reward head - projects to scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, config.reward_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.reward_dropout),
            nn.Linear(config.reward_hidden_size, 1)  # Scalar reward
        )
        
        # Value head for PPO (shares transformer backbone)
        self.value_head = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, config.reward_hidden_size),
            nn.ReLU(), 
            nn.Dropout(config.reward_dropout),
            nn.Linear(config.reward_hidden_size, 1)  # Scalar value
        )
        
        # Initialize reward and value heads
        self._init_heads()
    
    def _init_heads(self):
        """Initialize reward and value heads with small weights"""
        for module in [self.reward_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Small initialization to avoid large initial rewards
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_values: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through reward model.
        
        Args:
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return dict
            return_values: Whether to return value estimates for PPO
            
        Returns:
            rewards: Scalar rewards [batch_size]
            values: Value estimates [batch_size, seq_len] (if return_values=True)
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract last hidden states
        hidden_states = transformer_outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Get rewards from last token (following InstructGPT)
        # Use attention mask to find actual last tokens
        if attention_mask is not None:
            # Find the last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden_states = hidden_states[batch_indices, sequence_lengths]  # [batch_size, hidden_size]
        else:
            # If no attention mask, use last token
            last_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Compute scalar rewards
        rewards = self.reward_head(last_hidden_states).squeeze(-1)  # [batch_size]
        
        # Compute values for PPO if requested
        values = None
        if return_values:
            values = self.value_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        if return_dict:
            return {
                'rewards': rewards,
                'values': values,
                'hidden_states': hidden_states
            }
        else:
            return rewards, values
    
    def compute_preference_loss(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry preference loss for reward model training.
        
        Loss = -log(sigmoid(reward_chosen - reward_rejected))
        
        Args:
            chosen_input_ids: Preferred responses [batch_size, seq_len]
            rejected_input_ids: Less preferred responses [batch_size, seq_len]
            chosen_attention_mask: Mask for chosen responses
            rejected_attention_mask: Mask for rejected responses
            
        Returns:
            loss: Bradley-Terry preference loss
        """
        # Get rewards for chosen and rejected responses
        chosen_outputs = self.forward(
            chosen_input_ids, 
            attention_mask=chosen_attention_mask,
            return_dict=True
        )
        rejected_outputs = self.forward(
            rejected_input_ids,
            attention_mask=rejected_attention_mask, 
            return_dict=True
        )
        
        chosen_rewards = chosen_outputs['rewards']  # [batch_size]
        rejected_rewards = rejected_outputs['rewards']  # [batch_size]
        
        # Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        # Loss = -log P(chosen > rejected) = -log sigmoid(r_chosen - r_rejected)
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        
        # Additional metrics for monitoring
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_chosen_mean = chosen_rewards.mean()
        reward_rejected_mean = rejected_rewards.mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'reward_chosen': reward_chosen_mean,
            'reward_rejected': reward_rejected_mean,
            'reward_margin': reward_margin,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards
        }
    
    def get_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convenience method to get rewards"""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs['rewards']
    
    def save_pretrained(self, save_path: str):
        """Save reward model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_path, 'reward_model.pt'))
        
        # Save config
        import json
        config_dict = {
            'reward_hidden_size': self.config.reward_hidden_size,
            'reward_dropout': self.config.reward_dropout,
            'freeze_transformer': self.freeze_transformer
        }
        with open(os.path.join(save_path, 'reward_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_pretrained(cls, load_path: str, base_model: SLMModel, config):
        """Load reward model"""
        model = cls(base_model, config)
        
        # Load state dict
        state_dict_path = os.path.join(load_path, 'reward_model.pt')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model


class RewardTrainer:
    """Trainer for reward model using preference data"""
    
    def __init__(self, model: RewardModel, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.reward_model_learning_rate,
            weight_decay=config.base_model_config.weight_decay,
            betas=(config.base_model_config.beta1, config.base_model_config.beta2),
            eps=config.base_model_config.eps
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def train_step(self, batch) -> dict:
        """Single training step on preference batch"""
        self.model.train()
        
        # Extract batch data
        chosen_input_ids = batch['chosen_input_ids']
        rejected_input_ids = batch['rejected_input_ids']
        chosen_attention_mask = batch.get('chosen_attention_mask')
        rejected_attention_mask = batch.get('rejected_attention_mask')
        
        # Forward pass and compute loss
        loss_dict = self.model.compute_preference_loss(
            chosen_input_ids=chosen_input_ids,
            rejected_input_ids=rejected_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            rejected_attention_mask=rejected_attention_mask
        )
        
        loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.base_model_config.gradient_clip_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        self.global_step += 1
        
        # Return metrics
        return {
            'loss': loss.item(),
            'accuracy': loss_dict['accuracy'].item(),
            'reward_chosen': loss_dict['reward_chosen'].item(),
            'reward_rejected': loss_dict['reward_rejected'].item(),
            'reward_margin': loss_dict['reward_margin'].item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, eval_dataloader) -> dict:
        """Evaluate reward model on validation data"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_chosen = 0.0
        total_reward_rejected = 0.0
        total_reward_margin = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                loss_dict = self.model.compute_preference_loss(
                    chosen_input_ids=batch['chosen_input_ids'],
                    rejected_input_ids=batch['rejected_input_ids'],
                    chosen_attention_mask=batch.get('chosen_attention_mask'),
                    rejected_attention_mask=batch.get('rejected_attention_mask')
                )
                
                total_loss += loss_dict['loss'].item()
                total_accuracy += loss_dict['accuracy'].item()
                total_reward_chosen += loss_dict['reward_chosen'].item()
                total_reward_rejected += loss_dict['reward_rejected'].item()
                total_reward_margin += loss_dict['reward_margin'].item()
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_accuracy': total_accuracy / num_batches,
            'eval_reward_chosen': total_reward_chosen / num_batches,
            'eval_reward_rejected': total_reward_rejected / num_batches,
            'eval_reward_margin': total_reward_margin / num_batches
        }