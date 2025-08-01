"""
PPO Trainer Implementation for RLHF - InstructGPT Style
Proximal Policy Optimization with KL divergence control for stable training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
import numpy as np
from dataclasses import dataclass
import logging

# Add parent directory to path to import SLM components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from model import SLMModel
from reward_model import RewardModel


@dataclass
class PPOBatch:
    """Data structure for PPO training batch"""
    queries: torch.Tensor           # Input prompts [batch_size, seq_len]
    responses: torch.Tensor         # Generated responses [batch_size, response_len]  
    logprobs: torch.Tensor          # Log probabilities [batch_size, response_len]
    values: torch.Tensor            # Value estimates [batch_size, response_len]
    rewards: torch.Tensor           # Rewards from reward model [batch_size]
    advantages: torch.Tensor        # GAE advantages [batch_size, response_len]
    returns: torch.Tensor           # Discounted returns [batch_size, response_len]
    attention_mask: torch.Tensor    # Attention mask [batch_size, total_len]


class PPOTrainer:
    """PPO Trainer for RLHF following InstructGPT methodology"""
    
    def __init__(
        self,
        policy_model: SLMModel,
        ref_model: SLMModel,
        reward_model: RewardModel,
        tokenizer,
        config
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
            
        # Optimizer for policy and value function
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.ppo_learning_rate,
            weight_decay=config.base_model_config.weight_decay,
            betas=(config.base_model_config.beta1, config.base_model_config.beta2),
            eps=config.base_model_config.eps
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # KL controller for adaptive KL penalty
        self.kl_controller = AdaptiveKLController(
            init_kl_coeff=config.kl_coeff,
            target_kl=config.target_kl,
            kl_coeff_max=config.kl_coeff_max,
            kl_coeff_min=config.kl_coeff_min
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def generate_responses(
        self,
        queries: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate responses from policy model for given queries.
        
        Returns:
            responses: Generated token sequences
            logprobs: Log probabilities of generated tokens
            values: Value estimates for each token
        """
        self.policy_model.eval()
        
        batch_size, query_len = queries.shape
        device = queries.device
        
        # Initialize generation
        input_ids = queries.clone()
        generated_tokens = []
        logprobs_list = []
        values_list = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass through policy model
                outputs = self.policy_model(input_ids=input_ids)
                logits = outputs[0]  # [batch_size, seq_len, vocab_size]
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature
                if do_sample and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Create distribution and sample
                probs = F.softmax(next_token_logits, dim=-1)
                
                if do_sample:
                    if top_p < 1.0:
                        # Top-p (nucleus) sampling
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                        probs = F.softmax(next_token_logits, dim=-1)
                    
                    dist = Categorical(probs)
                    next_tokens = dist.sample()
                    logprobs = dist.log_prob(next_tokens)
                else:
                    # Greedy sampling
                    next_tokens = torch.argmax(probs, dim=-1)
                    logprobs = torch.log(probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1))
                
                # Get values from reward model (using value head)
                reward_outputs = self.reward_model(
                    input_ids=input_ids,
                    return_values=True,
                    return_dict=True
                )
                values = reward_outputs['values'][:, -1]  # [batch_size]
                
                # Store results
                generated_tokens.append(next_tokens)
                logprobs_list.append(logprobs)
                values_list.append(values)
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Check for EOS tokens (assuming tokenizer.eos_token_id exists)
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    if torch.all(next_tokens == self.tokenizer.eos_token_id):
                        break
        
        # Stack results
        responses = torch.stack(generated_tokens, dim=1)  # [batch_size, response_len]
        logprobs = torch.stack(logprobs_list, dim=1)      # [batch_size, response_len]
        values = torch.stack(values_list, dim=1)          # [batch_size, response_len]
        
        return responses, logprobs, values
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 1.0,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation) advantages and returns.
        
        Args:
            rewards: Per-step rewards [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        batch_size, seq_len = rewards.shape
        
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        # GAE computation (backwards through time)
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                nextnonterminal = 0  # Assume terminal
                nextvalues = 0
            else:
                nextnonterminal = 1
                nextvalues = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
            advantages[:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute PPO policy loss with clipping"""
        
        # Compute probability ratios
        logratio = logprobs - old_logprobs
        ratio = torch.exp(logratio)
        
        # Compute advantages (normalize per batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO clipped loss
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(
            ratio, 
            1.0 - self.config.ppo_clip_range,
            1.0 + self.config.ppo_clip_range
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2)
        
        # Apply attention mask
        if attention_mask is not None:
            policy_loss = policy_loss * attention_mask
            policy_loss = policy_loss.sum() / attention_mask.sum()
        else:
            policy_loss = policy_loss.mean()
            
        return policy_loss
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute value function loss"""
        
        if self.config.ppo_clip_range_vf is not None:
            # Clipped value loss (from PPO paper)
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.ppo_clip_range_vf,
                self.config.ppo_clip_range_vf
            )
            value_loss_1 = F.mse_loss(values, returns, reduction='none')
            value_loss_2 = F.mse_loss(values_clipped, returns, reduction='none')
            value_loss = torch.max(value_loss_1, value_loss_2)
        else:
            # Standard MSE loss
            value_loss = F.mse_loss(values, returns, reduction='none')
        
        # Apply attention mask
        if attention_mask is not None:
            value_loss = value_loss * attention_mask
            value_loss = value_loss.sum() / attention_mask.sum()
        else:
            value_loss = value_loss.mean()
            
        return value_loss
    
    def compute_kl_penalty(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence penalty between policy and reference model"""
        
        kl_div = ref_logprobs - logprobs  # KL(ref || policy)
        
        # Apply attention mask
        if attention_mask is not None:
            kl_div = kl_div * attention_mask
            kl_penalty = kl_div.sum() / attention_mask.sum()
        else:
            kl_penalty = kl_div.mean()
            
        return kl_penalty
    
    def train_step(self, batch: PPOBatch) -> Dict[str, float]:
        """Single PPO training step"""
        self.policy_model.train()
        
        metrics = {}
        
        # Mini-batch training (PPO epochs)
        for ppo_epoch in range(self.config.ppo_epochs):
            # Shuffle data for mini-batches
            batch_size = batch.queries.size(0)
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.config.ppo_mini_batch_size):
                end_idx = min(start_idx + self.config.ppo_mini_batch_size, batch_size)
                mini_batch_indices = indices[start_idx:end_idx]
                
                # Create mini-batch
                mini_queries = batch.queries[mini_batch_indices]
                mini_responses = batch.responses[mini_batch_indices]
                mini_old_logprobs = batch.logprobs[mini_batch_indices]
                mini_old_values = batch.values[mini_batch_indices]
                mini_advantages = batch.advantages[mini_batch_indices]
                mini_returns = batch.returns[mini_batch_indices]
                mini_attention_mask = batch.attention_mask[mini_batch_indices]
                
                # Forward pass through policy model
                full_input_ids = torch.cat([mini_queries, mini_responses], dim=1)
                policy_outputs = self.policy_model(input_ids=full_input_ids)
                logits = policy_outputs[0]
                
                # Get logits for response tokens only
                response_logits = logits[:, mini_queries.size(1):, :]  # [batch_size, response_len, vocab_size]
                
                # Compute log probabilities
                log_probs = F.log_softmax(response_logits, dim=-1)
                response_logprobs = log_probs.gather(-1, mini_responses.unsqueeze(-1)).squeeze(-1)
                
                # Get reference model logprobs
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=full_input_ids)
                    ref_logits = ref_outputs[0]
                    ref_response_logits = ref_logits[:, mini_queries.size(1):, :]
                    ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
                    ref_response_logprobs = ref_log_probs.gather(-1, mini_responses.unsqueeze(-1)).squeeze(-1)
                
                # Get current values from reward model
                reward_outputs = self.reward_model(
                    input_ids=full_input_ids,
                    return_values=True,
                    return_dict=True
                )
                current_values = reward_outputs['values'][:, mini_queries.size(1):]  # Response part only
                
                # Response attention mask
                response_mask = mini_attention_mask[:, mini_queries.size(1):]
                
                # Compute losses
                policy_loss = self.compute_policy_loss(
                    response_logprobs,
                    mini_old_logprobs,
                    mini_advantages,
                    response_mask
                )
                
                value_loss = self.compute_value_loss(
                    current_values,
                    mini_returns,
                    mini_old_values,
                    response_mask
                )
                
                kl_penalty = self.compute_kl_penalty(
                    response_logprobs,
                    ref_response_logprobs,
                    response_mask
                )
                
                # Entropy bonus
                entropy = -torch.sum(torch.exp(response_logprobs) * response_logprobs * response_mask) / response_mask.sum()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.ppo_value_loss_coeff * value_loss +
                    self.kl_controller.kl_coeff * kl_penalty -
                    self.config.ppo_entropy_coeff * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.ppo_max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Store metrics
                metrics.update({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'kl_penalty': kl_penalty.item(),
                    'entropy': entropy.item(),
                    'total_loss': total_loss.item(),
                    'kl_coeff': self.kl_controller.kl_coeff
                })
        
        # Update KL controller
        mean_kl = metrics.get('kl_penalty', 0.0)
        self.kl_controller.update(mean_kl)
        
        self.global_step += 1
        return metrics
    
    def evaluate_responses(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate generated responses using reward model"""
        
        with torch.no_grad():
            # Concatenate queries and responses
            full_sequences = torch.cat([queries, responses], dim=1)
            
            # Get rewards
            rewards = self.reward_model.get_rewards(full_sequences)
            
            # Get KL divergence with reference model
            policy_outputs = self.policy_model(input_ids=full_sequences)
            ref_outputs = self.ref_model(input_ids=full_sequences)
            
            policy_logits = policy_outputs[0][:, queries.size(1):, :]  # Response part
            ref_logits = ref_outputs[0][:, queries.size(1):, :]
            
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # KL divergence
            kl_div = torch.sum(torch.exp(ref_log_probs) * (ref_log_probs - policy_log_probs), dim=-1).mean()
            
            return {
                'mean_reward': rewards.mean().item(),
                'std_reward': rewards.std().item(),
                'mean_kl': kl_div.item()
            }


class AdaptiveKLController:
    """Adaptive KL coefficient controller for stable RLHF training"""
    
    def __init__(
        self,
        init_kl_coeff: float = 0.2,
        target_kl: float = 0.1,
        kl_coeff_max: float = 2.0,
        kl_coeff_min: float = 0.02,
        adaptation_factor: float = 1.5
    ):
        self.kl_coeff = init_kl_coeff
        self.target_kl = target_kl
        self.kl_coeff_max = kl_coeff_max
        self.kl_coeff_min = kl_coeff_min
        self.adaptation_factor = adaptation_factor
    
    def update(self, current_kl: float):
        """Update KL coefficient based on current KL divergence"""
        if current_kl > self.target_kl * self.adaptation_factor:
            # KL too high, increase penalty
            self.kl_coeff *= self.adaptation_factor
        elif current_kl < self.target_kl / self.adaptation_factor:
            # KL too low, decrease penalty
            self.kl_coeff /= self.adaptation_factor
        
        # Clamp to bounds
        self.kl_coeff = max(self.kl_coeff_min, min(self.kl_coeff_max, self.kl_coeff))


def create_ppo_batch(
    queries: torch.Tensor,
    responses: torch.Tensor,
    logprobs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95
) -> PPOBatch:
    """Create PPO batch with computed advantages and returns"""
    
    # Reshape rewards to match response length
    batch_size, response_len = responses.shape
    reward_per_step = torch.zeros(batch_size, response_len, device=rewards.device)
    reward_per_step[:, -1] = rewards  # Reward only at the end
    
    # Compute advantages and returns
    trainer = PPOTrainer(None, None, None, None, None)  # Dummy trainer for method access
    advantages, returns = trainer.compute_advantages(
        reward_per_step, values, gamma, gae_lambda
    )
    
    return PPOBatch(
        queries=queries,
        responses=responses,
        logprobs=logprobs,
        values=values,
        rewards=rewards,
        advantages=advantages,
        returns=returns,
        attention_mask=attention_mask
    )