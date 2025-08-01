"""
Main RLHF Training Pipeline - InstructGPT Style Implementation
Complete three-stage training: SFT -> Reward Model -> PPO
"""
import torch
import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import SLM components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from model import SLMModel
from tokenizer import SLMTokenizer
from slm_config import SLMConfig

# Import RLHF components
from rlhf_config import RLHFConfig, get_default_rlhf_config, get_fast_rlhf_config
from reward_model import RewardModel, RewardTrainer
from sft_trainer import SFTTrainer, create_sft_datasets
from ppo_trainer import PPOTrainer, create_ppo_batch
from data_utils import create_preference_dataset, create_ppo_dataset


class RLHFPipeline:
    """Complete RLHF training pipeline following InstructGPT methodology"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rlhf_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = None
        self.base_model = None
        self.sft_model = None
        self.reward_model = None
        self.ref_model = None  # Reference model for PPO (frozen SFT model)
        
        # Create output directories
        os.makedirs("sft_checkpoints", exist_ok=True)
        os.makedirs("reward_checkpoints", exist_ok=True)
        os.makedirs("ppo_checkpoints", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        self.logger.info("RLHF Pipeline initialized")
        
    def initialize_base_model(self, model_path: Optional[str] = None):
        """Initialize the base model and tokenizer"""
        
        self.logger.info("Initializing base model and tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = SLMTokenizer()
        
        if model_path and os.path.exists(model_path):
            # Load existing model
            self.logger.info(f"Loading model from {model_path}")
            self.base_model = SLMModel.from_pretrained(model_path, self.config.base_model_config)
        else:
            # Use best model from wikitext103 training if available
            best_model_path = "../xinSLM_v03_slm_llama_architecture/wikitext103_training/best_model"
            if os.path.exists(best_model_path):
                self.logger.info(f"Loading best WikiText-103 model from {best_model_path}")
                
                # Load the model state
                config_path = os.path.join(best_model_path, "config.json")
                model_path = os.path.join(best_model_path, "pytorch_model.bin")
                
                if os.path.exists(config_path) and os.path.exists(model_path):
                    # Load config
                    with open(config_path, 'r') as f:
                        model_config_dict = json.load(f)
                    
                    # Create model with loaded config
                    self.base_model = SLMModel(SLMConfig(**model_config_dict))
                    
                    # Load weights - handle different checkpoint formats
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    if 'model_state_dict' in checkpoint:
                        # Training checkpoint format
                        state_dict = checkpoint['model_state_dict']
                    else:
                        # Direct state dict format
                        state_dict = checkpoint
                    
                    # Handle key mapping if needed (remove "model." prefix)
                    if any(key.startswith('model.') for key in state_dict.keys()):
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith('model.'):
                                new_key = key[6:]  # Remove "model." prefix
                                new_state_dict[new_key] = value
                            else:
                                new_state_dict[key] = value
                        state_dict = new_state_dict
                    
                    self.base_model.load_state_dict(state_dict)
                    
                    self.logger.info("Successfully loaded pretrained WikiText-103 model")
                else:
                    self.logger.warning("WikiText-103 model files not found, initializing new model")
                    self.base_model = SLMModel(self.config.base_model_config)
            else:
                # Initialize new model
                self.logger.info("Initializing new base model")
                self.base_model = SLMModel(self.config.base_model_config)
        
        # Move to device
        self.base_model.to(self.device)
        
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.base_model.parameters() if p.requires_grad):,}")
    
    def stage1_supervised_finetuning(self):
        """Stage 1: Supervised Fine-tuning on instruction-response pairs"""
        
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: SUPERVISED FINE-TUNING")
        self.logger.info("=" * 60)
        
        # Clone base model for SFT
        self.sft_model = SLMModel(self.config.base_model_config)
        self.sft_model.load_state_dict(self.base_model.state_dict())
        self.sft_model.to(self.device)
        
        # Create datasets
        train_dataset, eval_dataset = create_sft_datasets(
            train_data_path=self.config.sft_data_path,
            eval_data_path=None,  # We'll create validation split if needed
            tokenizer=self.tokenizer,
            config=self.config
        )
        
        # Initialize SFT trainer
        sft_trainer = SFTTrainer(
            model=self.sft_model,
            tokenizer=self.tokenizer,
            config=self.config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Train
        self.logger.info("Starting SFT training...")
        sft_metrics = sft_trainer.train()
        
        # Test generation
        self.logger.info("Testing SFT model generation...")
        test_instructions = [
            "What is artificial intelligence?",
            "Explain how to cook pasta.",
            "Write a short story about a robot."
        ]
        
        for instruction in test_instructions:
            response = sft_trainer.generate_response(instruction)
            self.logger.info(f"Instruction: {instruction}")
            self.logger.info(f"Response: {response}")
            self.logger.info("-" * 40)
        
        # Save SFT metrics
        with open("sft_metrics.json", 'w') as f:
            json.dump(sft_metrics, f, indent=2)
        
        self.logger.info("SFT stage completed successfully")
        return sft_trainer
    
    def stage2_reward_model_training(self):
        """Stage 2: Train reward model on preference comparisons"""
        
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: REWARD MODEL TRAINING")
        self.logger.info("=" * 60)
        
        # Initialize reward model using SFT model as base
        self.reward_model = RewardModel(
            base_model=self.sft_model,
            config=self.config
        )
        self.reward_model.to(self.device)
        
        # Create preference dataset
        preference_dataset = create_preference_dataset(
            data_path=self.config.reward_data_path,
            tokenizer=self.tokenizer,
            config=self.config
        )
        
        # Initialize reward trainer
        reward_trainer = RewardTrainer(
            model=self.reward_model,
            config=self.config,
            tokenizer=self.tokenizer
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        reward_dataloader = DataLoader(
            preference_dataset,
            batch_size=self.config.reward_model_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=preference_dataset.collate_fn
        )
        
        # Training loop
        self.logger.info(f"Starting reward model training for {self.config.reward_model_epochs} epochs...")
        
        reward_metrics = []
        for epoch in range(self.config.reward_model_epochs):
            self.logger.info(f"Reward Model Epoch {epoch + 1}/{self.config.reward_model_epochs}")
            
            epoch_metrics = []
            progress_bar = tqdm(reward_dataloader, desc=f"Training Reward Model")
            
            for batch in progress_bar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Training step
                metrics = reward_trainer.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.3f}",
                    'margin': f"{metrics['reward_margin']:.3f}"
                })
            
            # Epoch metrics
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            reward_metrics.append(avg_metrics)
            
            self.logger.info(
                f"Epoch {epoch + 1} - Loss: {avg_metrics['loss']:.4f}, "
                f"Accuracy: {avg_metrics['accuracy']:.3f}, "
                f"Reward Margin: {avg_metrics['reward_margin']:.3f}"
            )
        
        # Save reward model
        self.reward_model.save_pretrained("reward_checkpoints/final_reward_model")
        
        # Save reward metrics
        with open("reward_metrics.json", 'w') as f:
            json.dump(reward_metrics, f, indent=2)
        
        self.logger.info("Reward model training completed successfully")
        return reward_trainer
    
    def stage3_ppo_training(self):
        """Stage 3: PPO training using reward model feedback"""
        
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: PPO TRAINING")
        self.logger.info("=" * 60)
        
        # Create reference model (frozen copy of SFT model)
        self.ref_model = SLMModel(self.config.base_model_config)
        self.ref_model.load_state_dict(self.sft_model.state_dict())
        self.ref_model.to(self.device)
        
        # Policy model starts as copy of SFT model
        policy_model = SLMModel(self.config.base_model_config)
        policy_model.load_state_dict(self.sft_model.state_dict())
        policy_model.to(self.device)
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            policy_model=policy_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            config=self.config
        )
        
        # Create PPO dataset (prompts for generation)
        ppo_dataset = create_ppo_dataset(
            data_path=self.config.ppo_prompts_path,
            tokenizer=self.tokenizer,
            config=self.config
        )
        
        # PPO training loop
        self.logger.info(f"Starting PPO training for {self.config.max_train_steps} steps...")
        
        ppo_metrics = []
        step = 0
        
        # Create data loader for prompts
        from torch.utils.data import DataLoader
        prompt_dataloader = DataLoader(
            ppo_dataset,
            batch_size=self.config.ppo_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        )
        
        while step < self.config.max_train_steps:
            for batch in prompt_dataloader:
                if step >= self.config.max_train_steps:
                    break
                
                # Move prompts to device
                queries = batch['input_ids'].to(self.device)
                
                # Generate responses
                responses, logprobs, values = ppo_trainer.generate_responses(
                    queries=queries,
                    max_length=self.config.generation_max_length,
                    temperature=self.config.generation_temperature,
                    top_p=self.config.generation_top_p,
                    do_sample=self.config.generation_do_sample
                )
                
                # Get rewards from reward model
                full_sequences = torch.cat([queries, responses], dim=1)
                rewards = self.reward_model.get_rewards(full_sequences)
                
                # Create attention mask
                attention_mask = torch.ones_like(full_sequences)
                
                # Create PPO batch
                ppo_batch = create_ppo_batch(
                    queries=queries,
                    responses=responses,
                    logprobs=logprobs,
                    values=values,
                    rewards=rewards,
                    attention_mask=attention_mask,
                    gamma=self.config.ppo_gamma,
                    gae_lambda=self.config.ppo_gae_lambda
                )
                
                # PPO training step
                step_metrics = ppo_trainer.train_step(ppo_batch)
                ppo_metrics.append(step_metrics)
                
                step += 1
                
                # Logging
                if step % self.config.logging_steps == 0:
                    self.logger.info(
                        f"Step {step}/{self.config.max_train_steps} - "
                        f"Policy Loss: {step_metrics['policy_loss']:.4f}, "
                        f"Value Loss: {step_metrics['value_loss']:.4f}, "
                        f"KL: {step_metrics['kl_penalty']:.4f}, "
                        f"Reward: {rewards.mean().item():.3f}"
                    )
                
                # Evaluation
                if step % self.config.eval_steps == 0:
                    eval_metrics = ppo_trainer.evaluate_responses(queries, responses)
                    self.logger.info(
                        f"Evaluation - Mean Reward: {eval_metrics['mean_reward']:.3f}, "
                        f"Mean KL: {eval_metrics['mean_kl']:.4f}"
                    )
                
                # Save checkpoint
                if step % self.config.save_steps == 0:
                    checkpoint_dir = f"ppo_checkpoints/checkpoint_step_{step}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    torch.save(
                        policy_model.state_dict(),
                        os.path.join(checkpoint_dir, 'pytorch_model.bin')
                    )
                    
                    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                        json.dump(self.config.to_dict(), f, indent=2)
                    
                    self.logger.info(f"Checkpoint saved at step {step}")
        
        # Save final model
        final_dir = "ppo_checkpoints/final_model"
        os.makedirs(final_dir, exist_ok=True)
        
        torch.save(policy_model.state_dict(), os.path.join(final_dir, 'pytorch_model.bin'))
        self.tokenizer.save_pretrained(final_dir)
        
        with open(os.path.join(final_dir, 'config.json'), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save PPO metrics
        with open("ppo_metrics.json", 'w') as f:
            json.dump(ppo_metrics, f, indent=2)
        
        self.logger.info("PPO training completed successfully")
        return policy_model
    
    def run_full_pipeline(self, model_path: Optional[str] = None):
        """Run the complete RLHF pipeline"""
        
        self.logger.info("Starting complete RLHF pipeline...")
        
        # Initialize base model
        self.initialize_base_model(model_path)
        
        # Stage 1: SFT
        sft_trainer = self.stage1_supervised_finetuning()
        
        # Stage 2: Reward Model
        reward_trainer = self.stage2_reward_model_training()
        
        # Stage 3: PPO
        final_model = self.stage3_ppo_training()
        
        self.logger.info("=" * 60)
        self.logger.info("RLHF PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 60)
        
        # Final evaluation
        self.logger.info("Testing final model...")
        test_instructions = [
            "Explain the concept of machine learning.",
            "Write a recipe for chocolate chip cookies.",
            "What are the benefits of renewable energy?",
            "How do you solve a Rubik's cube?",
            "Describe the water cycle."
        ]
        
        # Load final model for testing
        final_model.eval()
        
        for instruction in test_instructions:
            # Format prompt
            prompt = self.config.instruction_template + instruction + self.config.response_template
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            
            # Generate response
            with torch.no_grad():
                generated = []
                current_ids = input_ids
                
                for _ in range(256):  # Max response length
                    outputs = final_model(input_ids=current_ids)
                    logits = outputs[0]
                    next_token_logits = logits[0, -1, :] / 0.7  # Temperature
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    generated.append(next_token.item())
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
                    
                    # Simple stopping condition
                    if len(generated) > 10 and generated[-1] == generated[-2]:
                        break
            
            response = self.tokenizer.decode(generated)
            self.logger.info(f"Instruction: {instruction}")
            self.logger.info(f"Response: {response}")
            self.logger.info("-" * 50)
        
        return {
            'sft_model': self.sft_model,
            'reward_model': self.reward_model,
            'final_model': final_model
        }


def main():
    parser = argparse.ArgumentParser(description="RLHF Training Pipeline")
    parser.add_argument("--config", type=str, default="default", 
                       choices=["default", "fast"], help="Configuration preset")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model")
    parser.add_argument("--stage", type=str, default="all",
                       choices=["all", "sft", "reward", "ppo"],
                       help="Which stage to run")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "fast":
        config = get_fast_rlhf_config()
    else:
        config = get_default_rlhf_config()
    
    # Initialize pipeline
    pipeline = RLHFPipeline(config)
    
    if args.stage == "all":
        # Run full pipeline
        results = pipeline.run_full_pipeline(args.model_path)
    else:
        # Run specific stage
        pipeline.initialize_base_model(args.model_path)
        
        if args.stage == "sft":
            pipeline.stage1_supervised_finetuning()
        elif args.stage == "reward":
            # Need SFT model first
            pipeline.stage1_supervised_finetuning()
            pipeline.stage2_reward_model_training()
        elif args.stage == "ppo":
            # Need both SFT and reward models
            pipeline.stage1_supervised_finetuning()
            pipeline.stage2_reward_model_training()
            pipeline.stage3_ppo_training()


if __name__ == "__main__":
    main()