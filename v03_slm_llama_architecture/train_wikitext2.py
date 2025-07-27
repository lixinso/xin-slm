"""
Training script for SLM model using WikiText-2 dataset
Optimized for Mac Mini M4 with practical configurations
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

def download_wikitext2():
    """Download WikiText-2 dataset using datasets library"""
    try:
        from datasets import load_dataset
        print("Downloading WikiText-2 dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        print("✓ WikiText-2 dataset downloaded successfully!")
        return dataset
    except ImportError:
        print("Error: 'datasets' library not found. Installing...")
        os.system("pip install datasets")
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        return dataset

def prepare_wikitext2_data(dataset, tokenizer, max_length=512):
    """Prepare WikiText-2 data for training"""
    print("Preparing WikiText-2 data...")
    
    # Extract text from dataset splits
    train_texts = []
    eval_texts = []
    
    # Process training data
    for text in dataset['train']['text']:
        if text.strip() and len(text.strip()) > 50:  # Filter out empty and very short texts
            train_texts.append(text.strip())
    
    # Process validation data
    for text in dataset['validation']['text']:
        if text.strip() and len(text.strip()) > 50:
            eval_texts.append(text.strip())
    
    print(f"Original training texts: {len(train_texts)}")
    print(f"Original validation texts: {len(eval_texts)}")
    
    # Create datasets with sliding window
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length, stride=max_length//2)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=max_length, stride=max_length//2)
    
    print(f"Training samples (after windowing): {len(train_dataset)}")
    print(f"Evaluation samples (after windowing): {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def create_small_slm_config():
    """Create a smaller SLM configuration suitable for WikiText-2 training on Mac Mini M4"""
    config = SLMConfig(
        # Model architecture - smaller for faster training
        vocab_size=32000,           # Reduced vocabulary for faster training
        hidden_size=512,            # Smaller hidden size
        intermediate_size=2048,     # 4x hidden size
        num_hidden_layers=8,        # Fewer layers
        num_attention_heads=16,     # Reduced attention heads
        num_key_value_heads=4,      # GQA: 16 heads -> 4 KV heads
        max_position_embeddings=2048,  # Shorter context for faster training
        
        # Training parameters optimized for Mac Mini M4
        learning_rate=5e-4,         # Slightly lower learning rate
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        gradient_clip_norm=1.0,
        warmup_steps=500,           # Fewer warmup steps
        
        # Batch and sequence settings
        batch_size=4,               # Small batch size for Mac Mini
        sequence_length=512,        # Shorter sequences for faster training
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        
        # Logging and saving
        save_steps=200,             # Save more frequently
        eval_steps=100,             # Evaluate more frequently
        logging_steps=20,           # Log more frequently
        
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
    print("=" * 60)
    print("SLM TRAINING WITH WIKITEXT-2 DATASET")
    print("=" * 60)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("./wikitext2_training")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download dataset
    print("\n1. DOWNLOADING DATASET")
    print("-" * 30)
    dataset = download_wikitext2()
    
    # Step 2: Create configuration
    print("\n2. CREATING MODEL CONFIGURATION")
    print("-" * 35)
    config = create_small_slm_config()
    print(f"Model config: {config.hidden_size}d, {config.num_hidden_layers} layers")
    print(f"Attention: {config.num_attention_heads} heads -> {config.num_key_value_heads} KV heads")
    print(f"Vocabulary: {config.vocab_size:,} tokens")
    print(f"Max context: {config.max_position_embeddings:,} tokens")
    
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
    train_dataset, eval_dataset = prepare_wikitext2_data(
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
    print("Training configuration:")
    print(f"  - Epochs: 3")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Sequence length: {config.sequence_length}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Eval samples: {len(eval_dataset)}")
    
    estimated_steps = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * 3
    print(f"  - Estimated total steps: {estimated_steps}")
    
    start_time = time.time()
    
    try:
        trainer.train(num_epochs=3)
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Test generation
        print("\n7. TESTING GENERATION")
        print("-" * 25)
        test_generation(model, tokenizer, device)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_generation(model, tokenizer, device):
    """Test text generation with the trained model"""
    model.eval()
    model.to(device)
    
    test_prompts = [
        "The quick brown fox",
        "In the beginning",
        "Machine learning is",
        "The future of"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids]).to(device)
        
        # Generate
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(15):  # Generate 15 tokens
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