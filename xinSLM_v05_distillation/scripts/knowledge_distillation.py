"""
Knowledge Distillation Framework for Training Small Language Models

This module implements the core knowledge distillation training loop
following the theoretical principles outlined in the research:
- Soft target matching via KL divergence
- Combined loss (CE + KL)
- Temperature-scaled teacher logits
- On-policy distillation support
"""

import os
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple, Union
import wandb
from tqdm import tqdm
import numpy as np

from models.distilled_llama import DistilledLlamaForCausalLM, DistilledLlamaConfig


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined loss function for knowledge distillation:
    Loss = α * CE(student_logits, true_labels) + β * KL(teacher_logits || student_logits)
    """
    
    def __init__(
        self,
        alpha: float = 0.3,  # Weight for ground truth loss
        beta: float = 0.7,   # Weight for distillation loss
        temperature: float = 2.0,  # Temperature for softening distributions
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction, log_target=False)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the combined distillation loss
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        
        # Flatten for loss computation
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Mask out padding tokens if attention_mask is provided
        if attention_mask is not None:
            # Create mask for valid positions
            mask = attention_mask.view(-1).bool()
            student_logits_flat = student_logits.view(-1, vocab_size)[mask]
            teacher_logits_flat = teacher_logits.view(-1, vocab_size)[mask]
            labels_flat = labels.view(-1)[mask]
        else:
            student_logits_flat = student_logits.view(-1, vocab_size)
            teacher_logits_flat = teacher_logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
        
        # Ground truth loss (standard cross-entropy)
        ce_loss = self.ce_loss(student_logits_flat, labels_flat)
        
        # Distillation loss (KL divergence)
        # Apply temperature scaling to soften distributions
        teacher_probs = F.softmax(teacher_logits_flat / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits_flat / self.temperature, dim=-1)
        
        # KL(teacher || student) = sum(teacher * log(teacher/student))
        kl_loss = self.kl_loss(student_log_probs, teacher_probs)
        
        # Scale KL loss by temperature^2 (standard practice)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + self.beta * kl_loss
        
        # Return loss components for logging
        loss_dict = {
            'ce_loss': ce_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'total_loss': total_loss.detach()
        }
        
        return total_loss, loss_dict


class DistillationDataset(torch.utils.data.Dataset):
    """Dataset for knowledge distillation training"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        teacher_model: Optional[nn.Module] = None,
        precompute_teacher: bool = False
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_model = teacher_model
        self.precompute_teacher = precompute_teacher
        
        # Precompute teacher logits if requested (memory intensive but faster training)
        if precompute_teacher and teacher_model is not None:
            print("Precomputing teacher logits...")
            self.teacher_logits = self._precompute_teacher_logits()
        else:
            self.teacher_logits = None
    
    def _precompute_teacher_logits(self):
        """Precompute teacher logits for all samples"""
        teacher_logits = []
        self.teacher_model.eval()
        
        for text in tqdm(self.texts, desc="Computing teacher logits"):
            with torch.no_grad():
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                outputs = self.teacher_model(**encoding)
                logits = outputs.logits.cpu()
                teacher_logits.append(logits)
        
        return teacher_logits
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (shifted input_ids for causal LM)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Add precomputed teacher logits if available
        if self.teacher_logits is not None:
            result['teacher_logits'] = self.teacher_logits[idx].squeeze(0)
        
        return result


class DistillationTrainer:
    """Main trainer class for knowledge distillation"""
    
    def __init__(
        self,
        student_model: DistilledLlamaForCausalLM,
        teacher_model: nn.Module,
        tokenizer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Initialize loss function
        self.distillation_loss = KnowledgeDistillationLoss(
            alpha=config.get('alpha', 0.3),
            beta=config.get('beta', 0.7),
            temperature=config.get('temperature', 2.0)
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging and wandb if configured"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'distillation'),
                config=self.config,
                name=self.config.get('run_name', 'distillation_run')
            )
    
    def prepare_datasets(self, train_texts: List[str], val_texts: List[str] = None):
        """Prepare training and validation datasets"""
        
        # Create training dataset
        self.train_dataset = DistillationDataset(
            texts=train_texts,
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_length', 512),
            teacher_model=self.teacher_model,
            precompute_teacher=self.config.get('precompute_teacher', False)
        )
        
        # Create validation dataset if provided
        if val_texts is not None:
            self.val_dataset = DistillationDataset(
                texts=val_texts,
                tokenizer=self.tokenizer,
                max_length=self.config.get('max_length', 512),
                teacher_model=self.teacher_model,
                precompute_teacher=self.config.get('precompute_teacher', False)
            )
        else:
            self.val_dataset = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.get('eval_batch_size', 8),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4)
            )
    
    def compute_teacher_logits(self, batch):
        """Compute teacher logits on-the-fly"""
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            return teacher_outputs.logits
    
    def train_step(self, batch):
        """Single training step"""
        self.student_model.train()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        student_logits = student_outputs[0]
        
        # Get teacher logits (precomputed or on-the-fly)
        if 'teacher_logits' in batch:
            teacher_logits = batch['teacher_logits']
        else:
            teacher_logits = self.compute_teacher_logits(batch)
        
        # Compute distillation loss
        loss, loss_dict = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=batch['labels'],
            attention_mask=batch['attention_mask']
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                self.config['max_grad_norm']
            )
        
        self.optimizer.step()
        
        return loss_dict
    
    def validate(self):
        """Validation step"""
        if self.val_dataset is None:
            return {}
        
        self.student_model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get student outputs
                student_outputs = self.student_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                student_logits = student_outputs[0]
                
                # Get teacher logits
                if 'teacher_logits' in batch:
                    teacher_logits = batch['teacher_logits']
                else:
                    teacher_logits = self.compute_teacher_logits(batch)
                
                # Compute loss
                loss, loss_dict = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=batch['labels'],
                    attention_mask=batch['attention_mask']
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_ce_loss += loss_dict['ce_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_ce_loss': total_ce_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches
        }
    
    def train(self, num_epochs: int):
        """Main training loop"""
        self.logger.info(f"Starting distillation training for {num_epochs} epochs")
        
        # Setup scheduler
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        step = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_kl_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                loss_dict = self.train_step(batch)
                self.scheduler.step()
                
                epoch_loss += loss_dict['total_loss'].item()
                epoch_ce_loss += loss_dict['ce_loss'].item()
                epoch_kl_loss += loss_dict['kl_loss'].item()
                step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss_dict['total_loss'].item(),
                    'ce': loss_dict['ce_loss'].item(),
                    'kl': loss_dict['kl_loss'].item()
                })
                
                # Log to wandb
                if self.config.get('use_wandb', False) and step % self.config.get('log_steps', 100) == 0:
                    wandb.log({
                        'train_loss': loss_dict['total_loss'].item(),
                        'train_ce_loss': loss_dict['ce_loss'].item(),
                        'train_kl_loss': loss_dict['kl_loss'].item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'step': step
                    })
            
            # Validation
            val_metrics = self.validate()
            
            # Log epoch metrics
            num_batches = len(self.train_loader)
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss / num_batches,
                'train_ce_loss': epoch_ce_loss / num_batches,
                'train_kl_loss': epoch_kl_loss / num_batches,
                **val_metrics
            }
            
            self.logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")
            
            if self.config.get('use_wandb', False):
                wandb.log(epoch_metrics)
            
            # Save checkpoint
            if self.config.get('save_steps', 0) > 0 and (epoch + 1) % self.config['save_steps'] == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, is_best=True)
                self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('output_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{epoch}')
        
        # Save student model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'config': self.config
        }, os.path.join(checkpoint_path, 'pytorch_model.bin'))
        
        # Save model config
        with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
            json.dump(self.student_model.config.__dict__, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_instruction_data(dataset_name: str, split: str = 'train', num_samples: int = None) -> List[str]:
    """Load instruction dataset for distillation"""
    
    if dataset_name == 'alpaca':
        # Load Alpaca-style instruction dataset
        dataset = load_dataset('tatsu-lab/alpaca', split=split)
        
        texts = []
        for example in dataset:
            if example['input']:
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            texts.append(text)
    
    elif dataset_name == 'dolly':
        # Load Dolly instruction dataset
        dataset = load_dataset('databricks/databricks-dolly-15k', split=split)
        
        texts = []
        for example in dataset:
            if example['context']:
                text = f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}\n\n### Response:\n{example['response']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
            texts.append(text)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit number of samples if specified
    if num_samples is not None:
        texts = texts[:num_samples]
    
    return texts


def main():
    """Example usage of the distillation framework"""
    
    # Configuration
    config = {
        'teacher_model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'student_vocab_size': 32000,
        'student_hidden_size': 1024,
        'student_num_layers': 24,
        'student_num_heads': 16,
        'student_num_kv_heads': 4,
        'max_length': 512,
        'batch_size': 4,
        'eval_batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 3,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'alpha': 0.3,  # Ground truth loss weight
        'beta': 0.7,   # Distillation loss weight
        'temperature': 2.0,
        'output_dir': './checkpoints',
        'save_steps': 1,
        'log_steps': 100,
        'use_wandb': False,
        'wandb_project': 'llama_distillation',
        'run_name': 'distilled_llama_1b'
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['teacher_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config['teacher_model_name'],
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    # Create student model
    print("Creating student model...")
    from models.distilled_llama import create_distilled_llama
    student_model = create_distilled_llama(
        vocab_size=config['student_vocab_size'],
        hidden_size=config['student_hidden_size'],
        num_layers=config['student_num_layers'],
        num_heads=config['student_num_heads'],
        num_kv_heads=config['student_num_kv_heads']
    )
    
    # Print model info
    info = student_model.get_model_info()
    print(f"Student model: {info['total_parameters']:,} parameters ({info['parameter_size_mb']:.1f} MB)")
    
    # Load training data
    print("Loading training data...")
    train_texts = load_instruction_data('alpaca', split='train', num_samples=10000)
    val_texts = load_instruction_data('alpaca', split='train', num_samples=1000)[-1000:]  # Use last 1k for validation
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Initialize trainer
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Prepare datasets
    trainer.prepare_datasets(train_texts, val_texts)
    
    # Start training
    trainer.train(config['num_epochs'])


if __name__ == "__main__":
    main()