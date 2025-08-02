#!/usr/bin/env python3
"""
Main training script for knowledge distillation

This script orchestrates the complete distillation training pipeline:
- Load configuration
- Initialize teacher and student models
- Prepare datasets
- Run distillation training
- Evaluate and save results
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.distilled_llama import create_distilled_llama
from scripts.knowledge_distillation import DistillationTrainer, load_instruction_data
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = config.get('logging', {}).get('log_level', 'info').upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                config.get('logging', {}).get('logging_dir', './logs') + '/training.log'
            )
        ]
    )
    
    return logging.getLogger(__name__)


def load_teacher_model(config: dict, logger):
    """Load and prepare teacher model"""
    teacher_config = config['teacher']
    
    logger.info(f"Loading teacher model: {teacher_config['model_name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        'torch_dtype': getattr(torch, teacher_config.get('torch_dtype', 'float16')),
        'device_map': teacher_config.get('device_map', 'auto'),
        'trust_remote_code': teacher_config.get('trust_remote_code', False)
    }
    
    # Add quantization if specified
    if teacher_config.get('load_in_4bit', False):
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif teacher_config.get('load_in_8bit', False):
        model_kwargs['load_in_8bit'] = True
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_config['model_name'],
        **model_kwargs
    )
    
    # Freeze teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_model.eval()
    
    logger.info(f"Teacher model loaded successfully")
    return teacher_model, tokenizer


def create_student_model(config: dict, logger):
    """Create and initialize student model"""
    student_config = config['student']
    
    logger.info("Creating student model")
    
    # Load model configuration
    model_config_path = student_config.get('config_file', 'configs/model_config.yaml')
    with open(model_config_path, 'r') as f:
        model_yaml = yaml.safe_load(f)
    
    # Get variant configuration
    variant = student_config.get('variant', 'default')
    if variant != 'default' and variant in model_yaml.get('model_variants', {}):
        variant_config = model_yaml['model_variants'][variant]
        # Update base config with variant
        for key, value in variant_config.items():
            model_yaml['model'][key] = value
    
    model_config = model_yaml['model']
    
    # Create student model
    student_model = create_distilled_llama(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_hidden_layers'],
        num_heads=model_config['num_attention_heads'],
        num_kv_heads=model_config['num_key_value_heads'],
        max_seq_len=model_config['max_position_embeddings']
    )
    
    # Initialize from pretrained if specified
    if student_config.get('pretrained_path'):
        logger.info(f"Loading student from pretrained: {student_config['pretrained_path']}")
        # Implementation would depend on the pretrained model format
        pass
    
    # Print model info
    info = student_model.get_model_info()
    logger.info(f"Student model created: {info['total_parameters']:,} parameters ({info['parameter_size_mb']:.1f} MB)")
    
    return student_model


def prepare_training_data(config: dict, logger):
    """Prepare training and validation datasets"""
    data_config = config['data']
    
    logger.info("Preparing training data")
    
    # Load training datasets
    train_texts = []
    for dataset_config in data_config['train_datasets']:
        dataset_name = dataset_config['name']
        num_samples = dataset_config.get('num_samples', 10000)
        weight = dataset_config.get('weight', 1.0)
        
        logger.info(f"Loading {dataset_name} dataset ({num_samples} samples, weight: {weight})")
        
        texts = load_instruction_data(dataset_name, split='train', num_samples=num_samples)
        
        # Apply weighting by repeating samples
        if weight != 1.0:
            repeat_count = max(1, int(weight * len(texts)))
            texts = (texts * repeat_count)[:repeat_count]
        
        train_texts.extend(texts)
    
    # Load validation dataset
    val_texts = []
    if 'val_dataset' in data_config:
        val_config = data_config['val_dataset']
        val_texts = load_instruction_data(
            val_config['name'],
            split='train',  # We'll use a subset of train for validation
            num_samples=val_config.get('num_samples', 1000)
        )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    return train_texts, val_texts


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train distilled LLaMA model")
    parser.add_argument(
        "--config",
        default="configs/distillation_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (reduced dataset size)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.resume:
        config['checkpointing']['resume_from_checkpoint'] = args.resume
    
    if args.debug:
        config['experiment']['debug'] = True
        config['experiment']['max_steps'] = 100
        # Reduce dataset sizes for debugging
        for dataset in config['data']['train_datasets']:
            dataset['num_samples'] = min(dataset.get('num_samples', 1000), 100)
        if 'val_dataset' in config['data']:
            config['data']['val_dataset']['num_samples'] = 50
    
    # Setup logging
    os.makedirs(config.get('logging', {}).get('logging_dir', './logs'), exist_ok=True)
    logger = setup_logging(config)
    
    logger.info("Starting distillation training")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Set random seed for reproducibility
        if config.get('experiment', {}).get('seed'):
            import random
            import numpy as np
            seed = config['experiment']['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Load teacher model and tokenizer
        teacher_model, tokenizer = load_teacher_model(config, logger)
        
        # Create student model
        student_model = create_student_model(config, logger)
        
        # Prepare training data
        train_texts, val_texts = prepare_training_data(config, logger)
        
        # Initialize trainer
        logger.info("Initializing distillation trainer")
        
        training_config = {
            **config['training'],
            **config['loss'],
            **config['evaluation'],
            **config['checkpointing'],
            **config['logging'],
            'max_length': config['data']['max_length'],
            'precompute_teacher': False,  # Compute on-the-fly to save memory
        }
        
        trainer = DistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            config=training_config
        )
        
        # Prepare datasets
        logger.info("Preparing datasets")
        trainer.prepare_datasets(train_texts, val_texts if val_texts else None)
        
        # Run sanity check if enabled
        if config.get('experiment', {}).get('sanity_check', True):
            logger.info("Running sanity check")
            try:
                # Try a few training steps
                trainer.train(num_epochs=1)
                logger.info("Sanity check passed")
                
                # Reset for actual training
                trainer = DistillationTrainer(
                    student_model=student_model,
                    teacher_model=teacher_model,
                    tokenizer=tokenizer,
                    config=training_config
                )
                trainer.prepare_datasets(train_texts, val_texts if val_texts else None)
                
            except Exception as e:
                logger.error(f"Sanity check failed: {e}")
                return
        
        # Start training
        num_epochs = config['training']['num_epochs']
        if config.get('experiment', {}).get('max_steps'):
            # Override epochs for debugging
            steps_per_epoch = len(trainer.train_loader)
            max_steps = config['experiment']['max_steps']
            num_epochs = max(1, max_steps // steps_per_epoch)
            logger.info(f"Debug mode: limiting to {max_steps} steps ({num_epochs} epochs)")
        
        logger.info(f"Starting training for {num_epochs} epochs")
        trainer.train(num_epochs)
        
        logger.info("Training completed successfully!")
        
        # Final evaluation
        logger.info("Running final evaluation")
        final_metrics = trainer.validate()
        logger.info(f"Final validation metrics: {final_metrics}")
        
        # Save final model info
        final_info = {
            'config': config,
            'final_metrics': final_metrics,
            'model_info': student_model.get_model_info(),
            'training_completed': True
        }
        
        output_dir = config['checkpointing']['output_dir']
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            import json
            json.dump(final_info, f, indent=2, default=str)
        
        logger.info(f"Training artifacts saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()