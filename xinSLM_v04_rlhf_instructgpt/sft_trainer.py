"""
Supervised Fine-tuning (SFT) Trainer for RLHF - InstructGPT Style
First stage of RLHF: Fine-tune base model on instruction-response pairs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
import json
import logging
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import SLM components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xinSLM_v03_slm_llama_architecture'))
from model import SLMModel
from tokenizer import SLMTokenizer


class InstructionDataset(Dataset):
    """Dataset for instruction-response pairs for SFT"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SLMTokenizer,
        max_length: int = 512,
        instruction_template: str = "Human: ",
        response_template: str = "\n\nAssistant: "
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        self.response_template = response_template
        
        # Load data
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load instruction-response pairs from JSONL file"""
        data = []
        
        if not os.path.exists(data_path):
            # Create sample data for testing
            print(f"Data file {data_path} not found. Creating sample data...")
            sample_data = [
                {
                    "instruction": "What is the capital of France?",
                    "response": "The capital of France is Paris."
                },
                {
                    "instruction": "Explain what machine learning is in simple terms.",
                    "response": "Machine learning is a type of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for each task."
                },
                {
                    "instruction": "Write a short poem about the ocean.",
                    "response": "Waves crash upon the sandy shore,\nEndless blue as far as can be,\nThe ocean's song forevermore,\nA symphony of mystery."
                },
                {
                    "instruction": "How do you make a simple sandwich?",
                    "response": "To make a simple sandwich: 1) Take two slices of bread, 2) Add your desired filling (like ham, cheese, or peanut butter), 3) Put the slices together, 4) Cut in half if desired. Enjoy!"
                },
                {
                    "instruction": "What are the benefits of exercise?",
                    "response": "Exercise has many benefits including: improved cardiovascular health, stronger muscles and bones, better mental health and mood, increased energy levels, better sleep quality, and reduced risk of chronic diseases."
                }
            ]
            
            # Create directory and save sample data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w') as f:
                for item in sample_data:
                    f.write(json.dumps(item) + '\n')
            
            return sample_data
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        item = self.data[idx]
        
        # Format the conversation
        instruction = item['instruction']
        response = item['response']
        
        # Create full conversation text
        full_text = (
            self.instruction_template + instruction + 
            self.response_template + response
        )
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input_ids and labels
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # For SFT, we only compute loss on the response part
        # Find where the response starts
        response_start_text = self.instruction_template + instruction + self.response_template
        response_start_tokens = self.tokenizer.encode(response_start_text)
        response_start_pos = len(response_start_tokens)
        
        # Create labels (-100 for tokens we don't want to compute loss on)
        labels = input_ids.clone()
        labels[:response_start_pos] = -100  # Ignore instruction part
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for padding sequences to same length"""
    
    # Find max length in batch
    max_length = max(item['input_ids'].size(0) for item in batch)
    
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    
    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']
        attention_mask = item['attention_mask']
        
        # Pad sequences
        pad_length = max_length - input_ids.size(0)
        if pad_length > 0:
            # Pad with zeros (assuming 0 is pad token)
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            labels = F.pad(labels, (0, pad_length), value=-100)  # -100 is ignore index
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_attention_mask.append(attention_mask)
    
    return {
        'input_ids': torch.stack(batch_input_ids),
        'labels': torch.stack(batch_labels),
        'attention_mask': torch.stack(batch_attention_mask)
    }


class SFTTrainer:
    """Supervised Fine-tuning Trainer for instruction following"""
    
    def __init__(
        self,
        model: SLMModel,
        tokenizer: SLMTokenizer,
        config,
        train_dataset: Optional[InstructionDataset] = None,
        eval_dataset: Optional[InstructionDataset] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Datasets
        if train_dataset is None:
            self.train_dataset = InstructionDataset(
                data_path=config.sft_data_path,
                tokenizer=tokenizer,
                max_length=config.sft_max_length,
                instruction_template=config.instruction_template,
                response_template=config.response_template
            )
        else:
            self.train_dataset = train_dataset
            
        self.eval_dataset = eval_dataset
        
        # Data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.sft_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_fn
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=config.dataloader_num_workers,
                collate_fn=collate_fn
            )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.sft_learning_rate,
            weight_decay=config.base_model_config.weight_decay,
            betas=(config.base_model_config.beta1, config.base_model_config.beta2),
            eps=config.base_model_config.eps
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_dataloader) * config.sft_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.sft_learning_rate * 0.1
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.config.device)
        labels = batch['labels'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs[0]  # [batch_size, seq_len, vocab_size]
        
        # Compute loss (only on response tokens)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for cross entropy
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Cross entropy loss (ignore_index=-100)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        
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
        self.scheduler.step()
        
        self.global_step += 1
        
        # Compute perplexity
        perplexity = torch.exp(loss).item()
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        if not self.eval_dataset:
            return {}
            
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs[0]
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='sum')
                
                # Count valid tokens
                valid_tokens = (shift_labels != -100).sum().item()
                
                total_loss += loss.item()
                total_tokens += valid_tokens
                num_batches += 1
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')  # Avoid overflow
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop"""
        
        print(f"Starting SFT training for {self.config.sft_epochs} epochs...")
        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Steps per epoch: {len(self.train_dataloader)}")
        
        # Training metrics
        train_losses = []
        train_perplexities = []
        eval_losses = []
        eval_perplexities = []
        
        for epoch in range(self.config.sft_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.sft_epochs}")
            
            # Training
            epoch_losses = []
            epoch_perplexities = []
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                epoch_losses.append(metrics['loss'])
                epoch_perplexities.append(metrics['perplexity'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}: loss={metrics['loss']:.4f}, "
                        f"perplexity={metrics['perplexity']:.2f}, "
                        f"lr={metrics['learning_rate']:.2e}"
                    )
            
            # Epoch metrics
            avg_train_loss = np.mean(epoch_losses)
            avg_train_perplexity = np.mean(epoch_perplexities)
            
            train_losses.append(avg_train_loss)
            train_perplexities.append(avg_train_perplexity)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train PPL: {avg_train_perplexity:.2f}")
            
            # Evaluation
            if self.eval_dataset:
                eval_metrics = self.evaluate()
                eval_losses.append(eval_metrics['eval_loss'])
                eval_perplexities.append(eval_metrics['eval_perplexity'])
                
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_metrics['eval_loss']:.4f}, Eval PPL: {eval_metrics['eval_perplexity']:.2f}")
                
                # Save best model
                if eval_metrics['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['eval_loss']
                    self.save_model(f"best_sft_model")
                    print(f"New best model saved with eval loss: {self.best_eval_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % max(1, self.config.sft_epochs // 5) == 0:
                self.save_model(f"sft_checkpoint_epoch_{epoch + 1}")
        
        # Save final model
        self.save_model("final_sft_model")
        
        return {
            'train_losses': train_losses,
            'train_perplexities': train_perplexities,
            'eval_losses': eval_losses,
            'eval_perplexities': eval_perplexities
        }
    
    def save_model(self, save_name: str):
        """Save model checkpoint"""
        save_dir = os.path.join("sft_checkpoints", save_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Model saved to {save_dir}")
    
    def generate_response(
        self,
        instruction: str,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate response for a given instruction"""
        self.model.eval()
        
        # Format prompt
        prompt = self.config.instruction_template + instruction + self.config.response_template
        
        # Tokenize
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        input_ids = input_ids.to(self.config.device)
        
        # Generate
        with torch.no_grad():
            generated = []
            current_ids = input_ids
            
            for _ in range(max_length):
                outputs = self.model(input_ids=current_ids)
                logits = outputs[0]
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(next_token_logits, keepdim=True)
                
                generated.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop at EOS token if available
                if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode response
        response = self.tokenizer.decode(generated)
        return response.strip()


def create_sft_datasets(
    train_data_path: str,
    eval_data_path: Optional[str],
    tokenizer: SLMTokenizer,
    config
) -> Tuple[InstructionDataset, Optional[InstructionDataset]]:
    """Create SFT training and evaluation datasets"""
    
    train_dataset = InstructionDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=config.sft_max_length,
        instruction_template=config.instruction_template,
        response_template=config.response_template
    )
    
    eval_dataset = None
    if eval_data_path:
        eval_dataset = InstructionDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_length=config.sft_max_length,
            instruction_template=config.instruction_template,
            response_template=config.response_template
        )
    
    return train_dataset, eval_dataset