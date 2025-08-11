"""
Training Script for GPT-OSS MoE Model on Mac Mini (16GB)
Optimized for memory efficiency with gradient accumulation and checkpointing

Features:
- Memory-efficient training with gradient accumulation
- Mixed precision training (FP16)
- Model quantization during training
- Mac Mini optimizations with MPS support
- Comprehensive logging and monitoring
- Knowledge distillation support
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    default_data_collator
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
import logging
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import GPTOSSForCausalLM, GPTOSSMoEConfig, create_gpt_oss_moe
from models.quantization import ModelQuantizer, QuantizationConfig

# Import multi-dataset loader
try:
    from multi_dataset_loader import MultiDatasetLoader
    MULTI_DATASET_AVAILABLE = True
except ImportError:
    MULTI_DATASET_AVAILABLE = False

# Import resource monitoring
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent / "xinSLM_v05_distillation" / "scripts"))
try:
    from resource_monitor import MacResourceMonitor, start_resource_monitoring, stop_resource_monitoring, get_current_stats
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    print("Warning: Resource monitoring not available")


class MacMiniTrainer:
    """Training class optimized for Mac Mini constraints"""
    
    def __init__(self, config_path: str):
        # Load configurations
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_device()
        
        # Initialize model, tokenizer, and data
        self.model = None
        self.tokenizer = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Memory monitoring
        self.memory_stats = []
        self.resource_monitor = None
        if RESOURCE_MONITORING_AVAILABLE:
            self.resource_monitor = MacResourceMonitor(log_interval=30)
            self.logger.info("Resource monitoring enabled")
        else:
            self.logger.warning("Resource monitoring not available")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.get('monitoring', {}).get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Weights & Biases if enabled
        if self.config.get('monitoring', {}).get('use_wandb', False):
            wandb.init(
                project=self.config['monitoring'].get('wandb_project', 'xinslm-v06-gpt-oss'),
                entity=self.config['monitoring'].get('wandb_entity'),
                name=self.config['monitoring'].get('wandb_run_name'),
                config=self.config
            )
    
    def setup_device(self):
        """Setup device (MPS for Mac Mini)"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using Metal Performance Shaders (MPS)")
        else:
            self.device = torch.device("cpu")
            self.logger.info("MPS not available, using CPU")
        
        # Set memory management for MPS
        if self.device.type == "mps":
            # Configure MPS memory management
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    def setup_model(self):
        """Initialize model with Mac Mini optimizations"""
        self.logger.info("Setting up GPT-OSS MoE model...")
        
        # Load model configuration
        model_config = self.config['training']
        variant_config = self.config.get('model_variants', {}).get(
            model_config.get('model_variant', 'standard'), {}
        )
        
        # Create model with optimized parameters
        self.model = create_gpt_oss_moe(
            vocab_size=50257,
            hidden_size=variant_config.get('hidden_size', 768),
            num_layers=variant_config.get('num_hidden_layers', 20),
            num_heads=variant_config.get('num_attention_heads', 12),
            num_kv_heads=variant_config.get('num_key_value_heads', 4),
            max_seq_len=model_config.get('max_seq_length', 1024),
            num_experts=variant_config.get('num_experts', 32),
            num_experts_per_tok=variant_config.get('num_experts_per_tok', 2),
            reasoning_effort=variant_config.get('reasoning_effort', 'medium'),
            use_quantization=self.config.get('quantization', {}).get('enable_quantization', True)
        )
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model created with {model_info['active_parameters']:,} active parameters")
        self.logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"Estimated active memory: {model_info['active_size_mb']:.1f} MB")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.get('mac_optimizations', {}).get('gradient_checkpointing', True):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
            else:
                self.logger.info("Gradient checkpointing not available for this model")
        
        # Apply quantization if configured
        if self.config.get('quantization', {}).get('enable_quantization', True):
            self._apply_quantization()
    
    def _apply_quantization(self):
        """Apply quantization to model for memory efficiency"""
        self.logger.info("Applying model quantization...")
        
        quant_cfg = self.config.get('quantization', {})
        # If config explicitly disables quantization, exit early
        if not quant_cfg.get('enable_quantization', True):
            self.logger.info("Quantization disabled by config")
            return
        
        quantization_config = QuantizationConfig(
            bits=quant_cfg.get('bits', 4),
            group_size=quant_cfg.get('group_size', 32),
            quantize_moe_experts=quant_cfg.get('quantize_moe_weights', True),
            quantize_attention=quant_cfg.get('quantize_attention', False),
            use_mps_kernels=self.config.get('mac_optimizations', {}).get('use_metal_performance_shaders', True)
        )
        
        quantizer = ModelQuantizer(quantization_config)
        self.model = quantizer.quantize_model(self.model)
        
        # Log quantization stats
        stats = quantizer.get_model_compression_stats(self.model)
        self.logger.info(f"Quantization complete - Compression: {stats['compression_ratio']:.2f}x")
        self.logger.info(f"Memory savings: {stats['memory_savings_mb']:.1f} MB")
    
    def setup_tokenizer(self):
        """Setup tokenizer"""
        tokenizer_name = self.config['data'].get('tokenizer_path', 'gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info(f"Tokenizer loaded: {tokenizer_name}")
    
    def setup_data(self):
        """Setup training and evaluation data with multi-dataset support"""
        self.logger.info("Setting up datasets...")
        
        # Check if multi-dataset mode is enabled
        use_multi_datasets = self.config['training'].get('use_multi_datasets', False)
        
        if use_multi_datasets and MULTI_DATASET_AVAILABLE:
            self.logger.info("Using multi-dataset loader")
            self._setup_multi_datasets()
        else:
            self.logger.info("Using single dataset loader")
            self._setup_single_dataset()
    
    def _setup_multi_datasets(self):
        """Setup multiple datasets using MultiDatasetLoader"""
        # Initialize multi-dataset loader
        dataset_loader = MultiDatasetLoader(self.config)
        
        # Get dataset configurations
        train_datasets = self.config['training'].get('train_datasets', [])
        eval_datasets = self.config['training'].get('eval_datasets', [])
        
        if not train_datasets:
            # Fallback to single dataset
            self.logger.warning("No train_datasets specified, falling back to single dataset")
            self._setup_single_dataset()
            return
        
        # Create datasets
        max_seq_length = self.config['training'].get('max_seq_length', 1024)
        train_dataset, eval_dataset = dataset_loader.create_dataloaders(
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length
        )
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['training'].get('per_device_train_batch_size', 1),
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=self.config.get('mac_optimizations', {}).get('num_workers', 2),
            pin_memory=False
        )
        
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config['training'].get('per_device_eval_batch_size', 1),
                shuffle=False,
                collate_fn=default_data_collator,
                num_workers=self.config.get('mac_optimizations', {}).get('num_workers', 2),
                pin_memory=False
            )
        else:
            self.eval_dataloader = None
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            self.logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    def _setup_single_dataset(self):
        """Setup single dataset (original method)"""
        # Load dataset
        dataset_name = self.config['training'].get('dataset_name', 'wikitext-2')
        if dataset_name == 'wikitext-2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        else:
            dataset = load_dataset(dataset_name)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config['training'].get('max_seq_length', 1024)
            )
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        # Group texts for language modeling
        block_size = self.config['training'].get('max_seq_length', 1024)
        
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            
            result = {
                k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        processed_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            desc="Grouping texts"
        )
        
        # Create data loaders
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['training'].get('per_device_train_batch_size', 1),
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=self.config.get('mac_optimizations', {}).get('num_workers', 2),
            pin_memory=False  # Avoid pin_memory on Mac
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config['training'].get('per_device_eval_batch_size', 1),
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=self.config.get('mac_optimizations', {}).get('num_workers', 2),
            pin_memory=False
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Create optimizer
        optimizer_config = self.config['training']
        
        # Separate parameters for different learning rates (if needed)
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if p.requires_grad],
                'lr': float(optimizer_config.get('learning_rate', 3e-4))
            }
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=float(optimizer_config.get('learning_rate', 3e-4)),
            betas=(float(optimizer_config.get('beta1', 0.9)), float(optimizer_config.get('beta2', 0.95))),
            weight_decay=float(optimizer_config.get('weight_decay', 0.01)),
            eps=float(optimizer_config.get('epsilon', 1e-8))
        )
        
        # Setup scheduler
        total_steps = len(self.train_dataloader) * optimizer_config.get('num_train_epochs', 3)
        total_steps //= optimizer_config.get('gradient_accumulation_steps', 1)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=optimizer_config.get('warmup_steps', 1000),
            num_training_steps=total_steps
        )
        
        self.logger.info(f"Optimizer created with LR: {optimizer_config.get('learning_rate', 3e-4)}")
        self.logger.info(f"Total training steps: {total_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_moe_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs[0]  # Main language modeling loss
            
            # Scale loss for gradient accumulation
            gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Extract MoE auxiliary losses
            moe_loss = 0
            if len(outputs) > 2:  # Check if auxiliary losses are present
                aux_losses = outputs[-1]
                if isinstance(aux_losses, (list, tuple)):
                    for layer_aux_losses in aux_losses:
                        if isinstance(layer_aux_losses, dict):
                            for aux_loss_name, aux_loss_value in layer_aux_losses.items():
                                moe_loss += aux_loss_value.item()
                        elif hasattr(layer_aux_losses, 'item'):
                            moe_loss += layer_aux_losses.item()
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_moe_loss += moe_loss
            num_batches += 1
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['training'].get('logging_steps', 50) == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # Get memory stats
                    memory_info = ""
                    if self.resource_monitor:
                        try:
                            stats = self.resource_monitor.get_system_stats()
                            if stats:
                                memory_info = f", Mem={stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%)"
                                
                                # Check for memory pressure
                                if stats['memory_percent'] > 85:
                                    self.logger.warning(f"High memory usage: {stats['memory_percent']:.1f}%")
                                    torch.mps.empty_cache() if self.device.type == "mps" else torch.cuda.empty_cache()
                                
                        except Exception as e:
                            self.logger.warning(f"Memory monitoring error: {e}")
                    
                    self.logger.info(
                        f"Step {self.global_step}: Loss={total_loss/num_batches:.4f}, "
                        f"MoE_Loss={total_moe_loss/num_batches:.4f}, LR={current_lr:.2e}{memory_info}"
                    )
                    
                    # Log to wandb
                    if self.config.get('monitoring', {}).get('use_wandb', False):
                        log_data = {
                            'train_loss': total_loss / num_batches,
                            'train_moe_loss': total_moe_loss / num_batches,
                            'learning_rate': current_lr,
                            'global_step': self.global_step
                        }
                        
                        # Add memory stats to wandb
                        if self.resource_monitor:
                            try:
                                stats = self.resource_monitor.get_system_stats()
                                if stats:
                                    log_data.update({
                                        'memory_percent': stats['memory_percent'],
                                        'memory_used_gb': stats['memory_used_gb'],
                                        'process_memory_mb': stats['process_memory_mb']
                                    })
                            except:
                                pass
                        
                        wandb.log(log_data)
            
            # Memory cleanup for MPS - more aggressive
            cleanup_interval = self.config.get('mac_optimizations', {}).get('empty_cache_interval', 25)
            if self.device.type == "mps" and step % cleanup_interval == 0:
                torch.mps.empty_cache()
                
                # Additional memory monitoring and cleanup
                if self.resource_monitor and step % (cleanup_interval * 2) == 0:
                    try:
                        stats = self.resource_monitor.get_system_stats()
                        if stats and stats['memory_percent'] > 80:
                            self.logger.warning(f"Memory usage high ({stats['memory_percent']:.1f}%), forcing cleanup")
                            import gc
                            gc.collect()
                            torch.mps.empty_cache()
                    except Exception as e:
                        self.logger.debug(f"Memory cleanup error: {e}")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'moe_loss': f"{total_moe_loss/num_batches:.4f}"
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_moe_loss': total_moe_loss / num_batches
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        total_moe_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs[0]
                
                # Extract MoE losses
                moe_loss = 0
                if len(outputs) > 2:
                    aux_losses = outputs[-1]
                    if isinstance(aux_losses, (list, tuple)):
                        for layer_aux_losses in aux_losses:
                            if isinstance(layer_aux_losses, dict):
                                for aux_loss_name, aux_loss_value in layer_aux_losses.items():
                                    moe_loss += aux_loss_value.item()
                            elif hasattr(layer_aux_losses, 'item'):
                                moe_loss += layer_aux_losses.item()
                
                total_loss += loss.item()
                total_moe_loss += moe_loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_moe_loss = total_moe_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'eval_loss': avg_loss,
            'eval_moe_loss': avg_moe_loss,
            'perplexity': perplexity
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['training'].get('output_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and optimizer state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint-{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with eval_loss: {metrics['eval_loss']:.4f}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Start resource monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
            self.logger.info("Resource monitoring started")
        
        try:
            # Setup all components
            self.setup_tokenizer()
            self.setup_data()
            self.setup_model()
            self.setup_optimizer()
            
            # Training loop
            num_epochs = self.config['training'].get('num_train_epochs', 3)
            eval_steps = self.config['training'].get('eval_steps', 500)
            save_steps = self.config['training'].get('save_steps', 1000)
            
            for epoch in range(num_epochs):
                self.epoch = epoch
                self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Evaluate
                eval_metrics = self.evaluate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **eval_metrics}
                
                # Log metrics with memory info
                memory_info = ""
                if self.resource_monitor:
                    try:
                        stats = self.resource_monitor.get_system_stats()
                        if stats:
                            memory_info = f", Memory: {stats['memory_percent']:.1f}%"
                    except:
                        pass
                
                self.logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                    f"Perplexity: {eval_metrics['perplexity']:.2f}{memory_info}"
                )
                
                # Log to wandb
                if self.config.get('monitoring', {}).get('use_wandb', False):
                    wandb.log({**all_metrics, 'epoch': epoch})
                
                # Save checkpoint
                is_best = eval_metrics['eval_loss'] < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = eval_metrics['eval_loss']
                
                self.save_checkpoint(all_metrics, is_best)
            
            self.logger.info("Training completed!")
            self.logger.info(f"Best eval loss: {self.best_eval_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Stop resource monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
                self.logger.info("Resource monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Train GPT-OSS MoE model on Mac Mini")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MacMiniTrainer(args.config)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
    except Exception as e:
        trainer.logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()