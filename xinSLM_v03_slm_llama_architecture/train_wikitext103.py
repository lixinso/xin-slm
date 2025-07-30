"""
Training script for SLM model using WikiText-103 dataset
Optimized for larger dataset training on Mac Mini M4
"""
import torch
import os
import time
from pathlib import Path

# Import SLM components
from slm_config import SLMConfig
from model import SLMForCausalLM
from tokenizer import SLMTokenizer
from train import SLMTrainer, TextDataset

def download_wikitext103():
    """Download WikiText-103 dataset using datasets library"""
    try:
        from datasets import load_dataset
        print("Downloading WikiText-103 dataset...")
        print("Note: WikiText-103 is ~500MB, this may take a few minutes...")
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        print("✓ WikiText-103 dataset downloaded successfully!")
        return dataset
    except ImportError:
        print("Error: 'datasets' library not found. Installing...")
        os.system("pip install datasets")
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        return dataset

def prepare_wikitext103_data(dataset, tokenizer, max_length=1024):
    """Prepare WikiText-103 data for training with optimized processing"""
    print("Preparing WikiText-103 data...")
    
    # Extract text from dataset splits
    train_texts = []
    eval_texts = []
    
    # Process training data with filtering for quality - FASTER processing
    print("Processing training data (limited for speed)...")
    for i, text in enumerate(dataset['train']['text']):
        if i >= 50000:  # Limit to first 50k examples for speed
            break
        if text.strip() and len(text.strip()) > 50:  # Lower threshold for speed
            train_texts.append(text.strip())
        
        # Progress indicator
        if i > 0 and i % 20000 == 0:
            print(f"  Processed {i:,} training texts, kept {len(train_texts):,}")
    
    # Process validation data - LIMITED
    print("Processing validation data (limited for speed)...")
    for i, text in enumerate(dataset['validation']['text']):
        if i >= 5000:  # Further limit validation data
            break
        if text.strip() and len(text.strip()) > 50:
            eval_texts.append(text.strip())
    
    print(f"Original training texts: {len(train_texts):,}")
    print(f"Original validation texts: {len(eval_texts):,}")
    
    # Aggressive data limitation for 8-hour target
    train_dataset = TextDataset(train_texts[:15000], tokenizer, max_length=max_length, stride=max_length)  # Half the data
    eval_dataset = TextDataset(eval_texts[:1000], tokenizer, max_length=max_length, stride=max_length)  # Minimal eval
    
    print(f"Training samples (after windowing): {len(train_dataset):,}")
    print(f"Evaluation samples (after windowing): {len(eval_dataset):,}")
    
    return train_dataset, eval_dataset

def create_optimized_slm_config():
    """Create an optimized SLM configuration for WikiText-103 training"""
    config = SLMConfig(
        # Model architecture - further optimized for 8-hour target
        vocab_size=32000,           # Same vocabulary size
        hidden_size=384,            # Further reduced for speed
        intermediate_size=1536,     # 4x hidden size
        num_hidden_layers=6,        # Even fewer layers
        num_attention_heads=6,      # Fewer attention heads
        num_key_value_heads=2,      # GQA: 6 heads -> 2 KV heads
        max_position_embeddings=512,   # Much shorter context
        
        # Training parameters optimized for speed
        learning_rate=5e-4,         # Higher learning rate for faster convergence
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        gradient_clip_norm=1.0,
        warmup_steps=500,           # Fewer warmup steps
        
        # Batch and sequence settings - aggressive optimization
        batch_size=16,              # Much larger batch size
        sequence_length=256,        # Much shorter sequences
        gradient_accumulation_steps=2,  # Effective batch size = 16 * 2 = 32
        
        # Logging and saving - more frequent for monitoring
        save_steps=1000,            # Save checkpoints
        eval_steps=500,             # Evaluate periodically
        logging_steps=25,           # Log frequently for monitoring
        
        # Regularization
        attention_dropout=0.1,
        hidden_dropout=0.1,
        
        # Architecture specifics
        use_cache=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        output_attentions=False,
        output_hidden_states=False,
        
        # No distillation for this training
        use_distillation=False,
    )
    
    return config

def main():
    """Main training function"""
    print("=" * 70)
    print("SLM TRAINING WITH WIKITEXT-103 DATASET")
    print("=" * 70)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("./wikitext103_training")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download dataset
    print("\n1. DOWNLOADING DATASET")
    print("-" * 30)
    dataset = download_wikitext103()
    
    # Print dataset statistics
    print(f"Train set size: {len(dataset['train']):,} examples")
    print(f"Validation set size: {len(dataset['validation']):,} examples")
    print(f"Test set size: {len(dataset['test']):,} examples")
    
    # Step 2: Create configuration
    print("\n2. CREATING MODEL CONFIGURATION")
    print("-" * 35)
    config = create_optimized_slm_config()
    print(f"Model config: {config.hidden_size}d, {config.num_hidden_layers} layers")
    print(f"Attention: {config.num_attention_heads} heads -> {config.num_key_value_heads} KV heads")
    print(f"Vocabulary: {config.vocab_size:,} tokens")
    print(f"Max context: {config.max_position_embeddings:,} tokens")
    print(f"Sequence length: {config.sequence_length:,} tokens")
    
    # Step 3: Create tokenizer and model
    print("\n3. INITIALIZING MODEL")
    print("-" * 25)
    tokenizer = SLMTokenizer(vocab_size=config.vocab_size)
    model = SLMForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Step 4: Prepare data
    print("\n4. PREPARING TRAINING DATA")
    print("-" * 30)
    train_dataset, eval_dataset = prepare_wikitext103_data(
        dataset, tokenizer, max_length=config.sequence_length
    )
    
    if len(train_dataset) == 0:
        print("❌ No training data available. Check dataset preparation.")
        return
    
    # Step 5: Create trainer
    print("\n5. SETTING UP TRAINER")
    print("-" * 25)
    trainer = SLMTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir)
    )
    
    # Step 6: Start training
    print("\n6. STARTING TRAINING")
    print("-" * 25)
    num_epochs = 1  # Single epoch for 8-hour target
    print("Training configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Sequence length: {config.sequence_length}")
    print(f"  - Training samples: {len(train_dataset):,}")
    print(f"  - Eval samples: {len(eval_dataset):,}")
    
    estimated_steps = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * num_epochs
    estimated_time_hours = estimated_steps * 2.0 / 3600  # Updated estimate based on optimizations
    print(f"  - Estimated total steps: {estimated_steps:,}")
    print(f"  - Estimated training time: ~{estimated_time_hours:.1f} hours")
    
    start_time = time.time()
    
    try:
        trainer.train(num_epochs=num_epochs)
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"Total training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Save detailed training time metrics
        save_training_time_summary(training_time, estimated_steps, config, output_dir, model)
        
        # Test generation
        print("\n7. TESTING GENERATION")
        print("-" * 25)
        test_generation(model, tokenizer, device)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

def save_training_time_summary(training_time, estimated_steps, config, output_dir, model):
    """Save detailed training time summary for reporting"""
    from datetime import timedelta
    import json
    
    time_summary = {
        'total_training_time_seconds': training_time,
        'total_training_time_hours': training_time / 3600,
        'total_training_time_formatted': str(timedelta(seconds=int(training_time))),
        'estimated_steps': estimated_steps,
        'average_time_per_step': training_time / estimated_steps if estimated_steps > 0 else 0,
        'steps_per_hour': 3600 / (training_time / estimated_steps) if estimated_steps > 0 and training_time > 0 else 0,
        'training_config': {
            'batch_size': config.batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'effective_batch_size': config.batch_size * config.gradient_accumulation_steps,
            'sequence_length': config.sequence_length,
            'model_parameters': sum(p.numel() for p in model.parameters()),
        },
        'hardware_info': {
            'device': 'Mac Mini M4',
            'accelerator': 'MPS',
            'memory_optimization': 'Grouped-Query Attention',
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to training directory
    time_file = output_dir / 'training_time_summary.json'
    with open(time_file, 'w') as f:
        json.dump(time_summary, f, indent=2)
    
    print(f"\n⏱️ TRAINING TIME SUMMARY:")
    print(f"   Total time: {time_summary['total_training_time_formatted']}")
    print(f"   Average per step: {time_summary['average_time_per_step']:.2f} seconds")
    print(f"   Training speed: {time_summary['steps_per_hour']:.1f} steps/hour")
    print(f"   Time summary saved to: {time_file}")

def test_generation(model, tokenizer, device):
    """Test text generation with the trained model"""
    model.eval()
    model.to(device)
    
    # Ensure device is set correctly - handle MPS issues
    if device.type == "mps":
        device = torch.device("cpu")  # Use CPU for generation to avoid MPS issues
        model.to(device)
    
    test_prompts = [
        "The quick brown fox",
        "In the beginning",
        "Machine learning is",
        "The future of artificial intelligence",
        "Natural language processing",
        "Deep learning has revolutionized"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        
        # Generate
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(20):  # Generate 20 tokens
                outputs = model(torch.tensor([generated_ids]).to(device))
                logits = outputs[1]
                
                # Get next token (with some randomness)
                next_token_logits = logits[0, -1, :] / 0.8  # Temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(next_token_probs, 1).item()
                
                generated_ids.append(next_token_id)
                
                # Stop if we hit special tokens
                if next_token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    break
        
        # Decode
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")

if __name__ == "__main__":
    main()