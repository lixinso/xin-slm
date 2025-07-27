"""
Training script for Small Language Model with knowledge distillation support
Based on Llama 3.2 training procedures with AdamW, cosine scheduling, and distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import asdict
import time
from tqdm import tqdm

from slm_config import SLMConfig
from model import SLMForCausalLM
from tokenizer import SLMTokenizer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: SLMTokenizer,
        max_length: int = 8192,
        stride: int = 4096
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts and create sliding windows
        self.examples = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            
            # Create sliding windows
            for i in range(0, len(token_ids) - max_length + 1, stride):
                window = token_ids[i:i + max_length]
                if len(window) == max_length:
                    self.examples.append(window)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining student loss and teacher matching"""
    
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits (optional) [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            
        Returns:
            Total loss and loss components
        """
        # Standard cross-entropy loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        student_loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        loss_dict = {"student_loss": student_loss.item()}
        
        if teacher_logits is not None:
            # Distillation loss
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            
            # Apply temperature scaling
            student_probs = F.log_softmax(shift_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
            
            # KL divergence loss
            distillation_loss = self.kl_loss(
                student_probs.view(-1, student_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1))
            ) * (self.temperature ** 2)
            
            # Combined loss
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
            loss_dict.update({
                "distillation_loss": distillation_loss.item(),
                "total_loss": total_loss.item()
            })
        else:
            total_loss = student_loss
            loss_dict["total_loss"] = student_loss.item()
        
        return total_loss, loss_dict


class SLMTrainer:
    """Trainer class for Small Language Model with distillation support"""
    
    def __init__(
        self,
        config: SLMConfig,
        model: SLMForCausalLM,
        tokenizer: SLMTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        teacher_model: Optional[SLMForCausalLM] = None,
        output_dir: str = "./checkpoints"
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.teacher_model = teacher_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            # Freeze teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Calculate total training steps
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        total_steps = len(self.train_dataloader) * 10  # Assume 10 epochs for now
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0.0
        )
        
        # Setup loss function
        self.loss_fn = DistillationLoss(
            alpha=config.distillation_alpha,
            temperature=config.distillation_temperature
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup logging
        self.train_metrics = []
        self.eval_metrics = []
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        student_logits = outputs[1]  # logits are second output
        
        # Get teacher logits if available
        teacher_logits = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs[1] if len(teacher_outputs) > 1 else teacher_outputs[0]
        
        # Compute loss
        loss, loss_dict = self.loss_fn(student_logits, teacher_logits, input_ids)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update global step
        self.global_step += 1
        
        # Add learning rate to metrics
        loss_dict["learning_rate"] = self.scheduler.get_last_lr()[0]
        
        return loss_dict
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on eval dataset"""
        if self.eval_dataset is None:
            return {}
        
        self.model.eval()
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        total_loss = 0.0
        total_student_loss = 0.0
        total_distillation_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                student_logits = outputs[1]
                
                # Get teacher logits if available
                teacher_logits = None
                if self.teacher_model is not None:
                    teacher_outputs = self.teacher_model(input_ids=input_ids)
                    teacher_logits = teacher_outputs[1] if len(teacher_outputs) > 1 else teacher_outputs[0]
                
                # Compute loss
                loss, loss_dict = self.loss_fn(student_logits, teacher_logits, input_ids)
                
                total_loss += loss_dict["total_loss"]
                total_student_loss += loss_dict["student_loss"]
                if "distillation_loss" in loss_dict:
                    total_distillation_loss += loss_dict["distillation_loss"]
                num_batches += 1
        
        eval_metrics = {
            "eval_total_loss": total_loss / num_batches,
            "eval_student_loss": total_student_loss / num_batches,
            "eval_perplexity": math.exp(total_student_loss / num_batches)
        }
        
        if total_distillation_loss > 0:
            eval_metrics["eval_distillation_loss"] = total_distillation_loss / num_batches
        
        return eval_metrics
    
    def save_checkpoint(self, checkpoint_dir: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': asdict(self.config),
            'best_eval_loss': self.best_eval_loss
        }
        
        torch.save(model_state, checkpoint_path / "pytorch_model.bin")
        
        # Save config
        with open(checkpoint_path / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_path))
        
        if is_best:
            # Copy to best model directory
            best_dir = checkpoint_path.parent / "best_model"
            best_dir.mkdir(exist_ok=True)
            
            torch.save(model_state, best_dir / "pytorch_model.bin")
            with open(best_dir / "config.json", 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            self.tokenizer.save_pretrained(str(best_dir))
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, num_epochs: int = 10, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {len(self.train_dataloader) * num_epochs}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training loop
            self.model.train()
            epoch_metrics = []
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False
            )
            
            for step, batch in enumerate(progress_bar):
                step_metrics = self.train_step(batch)
                epoch_metrics.append(step_metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['total_loss']:.4f}",
                    'lr': f"{step_metrics['learning_rate']:.2e}"
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_metrics = {
                        key: sum(m[key] for m in epoch_metrics[-self.config.logging_steps:]) / 
                        min(len(epoch_metrics), self.config.logging_steps)
                        for key in step_metrics.keys()
                    }
                    self.train_metrics.append({
                        'step': self.global_step,
                        'epoch': epoch,
                        **avg_metrics
                    })
                    
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={avg_metrics['total_loss']:.4f}, "
                        f"lr={avg_metrics['learning_rate']:.2e}"
                    )
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        eval_metrics['step'] = self.global_step
                        eval_metrics['epoch'] = epoch
                        self.eval_metrics.append(eval_metrics)
                        
                        logger.info(
                            f"Eval at step {self.global_step}: "
                            f"loss={eval_metrics['eval_total_loss']:.4f}, "
                            f"ppl={eval_metrics['eval_perplexity']:.2f}"
                        )
                        
                        # Save best model
                        if eval_metrics['eval_total_loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_total_loss']
                            self.save_checkpoint(
                                str(self.output_dir / f"checkpoint-{self.global_step}"),
                                is_best=True
                            )
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(str(self.output_dir / f"checkpoint-{self.global_step}"))
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Final checkpoint
        self.save_checkpoint(str(self.output_dir / "final_checkpoint"))
        
        # Save training metrics
        with open(self.output_dir / "train_metrics.json", 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
        
        with open(self.output_dir / "eval_metrics.json", 'w') as f:
            json.dump(self.eval_metrics, f, indent=2)
        
        logger.info("Training completed!")


def create_sample_dataset(tokenizer: SLMTokenizer, num_samples: int = 1000) -> List[str]:
    """Create a sample dataset for demonstration"""
    import random
    
    # Sample texts for training
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning models require large datasets.",
        "Natural language processing involves understanding text.",
        "Deep learning uses neural networks with multiple layers.",
        "Artificial intelligence aims to simulate human intelligence.",
        "Data science combines statistics and programming.",
        "Large language models can generate human-like text.",
        "Transformer architectures have revolutionized NLP.",
        "Knowledge distillation transfers knowledge from teacher to student models.",
    ]
    
    # Generate more diverse samples
    dataset = []
    for _ in range(num_samples):
        # Combine random sentences
        num_sentences = random.randint(5, 20)
        text = " ".join(random.choices(sample_texts, k=num_sentences))
        dataset.append(text)
    
    return dataset


def main():
    """Main training function"""
    # Create configuration
    config = SLMConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=128000,
        learning_rate=1e-3,
        batch_size=4,  # Small batch for demo
        sequence_length=512,  # Shorter sequence for demo
        gradient_accumulation_steps=1,
        warmup_steps=100,
        save_steps=500,
        eval_steps=250,
        logging_steps=50,
        use_distillation=False  # Set to True if you have a teacher model
    )
    
    # Create tokenizer
    tokenizer = SLMTokenizer(vocab_size=config.vocab_size)
    
    # Create model
    model = SLMForCausalLM(config)
    
    # Create datasets
    train_texts = create_sample_dataset(tokenizer, num_samples=1000)
    eval_texts = create_sample_dataset(tokenizer, num_samples=100)
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.sequence_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=config.sequence_length)
    
    # Create trainer
    trainer = SLMTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        teacher_model=None,  # No teacher for this demo
        output_dir="./checkpoints"
    )
    
    # Start training
    trainer.train(num_epochs=3)


if __name__ == "__main__":
    main()