"""
Example usage of the Small Language Model implementation
Demonstrates training, inference, and various configuration options
"""
import torch
import json
from pathlib import Path

from slm_config import SLMConfig
from model import SLMForCausalLM
from tokenizer import SLMTokenizer
from train import SLMTrainer, TextDataset, create_sample_dataset


def basic_training_example():
    """Basic training example with default configuration"""
    print("=== Basic Training Example ===")
    
    # Create configuration for a smaller model (for demo purposes)
    config = SLMConfig(
        vocab_size=32000,  # Smaller vocab for demo
        hidden_size=512,   # Smaller model
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        learning_rate=5e-4,
        batch_size=2,
        sequence_length=512,
        gradient_accumulation_steps=2,
        save_steps=100,
        eval_steps=50,
        logging_steps=10
    )
    
    # Initialize tokenizer and model
    tokenizer = SLMTokenizer(vocab_size=config.vocab_size)
    model = SLMForCausalLM(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample datasets
    train_texts = create_sample_dataset(tokenizer, num_samples=100)
    eval_texts = create_sample_dataset(tokenizer, num_samples=20)
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.sequence_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=config.sequence_length)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Create trainer
    trainer = SLMTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./demo_checkpoints"
    )
    
    # Train for a few steps
    print("Starting training...")
    trainer.train(num_epochs=1)
    print("Training completed!")


def inference_example():
    """Example of using a trained model for inference"""
    print("\n=== Inference Example ===")
    
    # Create a simple model for demonstration
    config = SLMConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=4
    )
    
    tokenizer = SLMTokenizer(vocab_size=config.vocab_size)
    model = SLMForCausalLM(config)
    
    # Put model in eval mode
    model.eval()
    
    # Sample text generation
    prompts = [
        "The future of artificial intelligence",
        "Deep learning is",
        "Python programming language"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids])
        
        # Generate (simple greedy decoding)
        with torch.no_grad():
            for _ in range(20):  # Generate 20 tokens
                outputs = model(input_ids=input_tensor)
                logits = outputs[1]  # Get logits
                next_token_id = torch.argmax(logits[0, -1, :]).item()
                
                # Append to sequence
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token_id]])
                ], dim=1)
                
                # Stop if we hit end token
                if next_token_id == tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_ids = input_tensor[0].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")


def rope_scaling_example():
    """Example of using RoPE scaling for different context lengths"""
    print("\n=== RoPE Scaling Example ===")
    
    from rope import RotaryEmbedding, RoPEScaledRotaryEmbedding
    
    head_dim = 64
    max_pos = 8192
    
    # Standard RoPE
    rope_standard = RotaryEmbedding(head_dim, max_pos)
    
    # Scaled RoPE for longer contexts
    rope_scaled = RoPEScaledRotaryEmbedding(
        head_dim,
        max_pos,
        scaling_type="linear",
        scaling_factor=2.0  # 2x scaling
    )
    
    # Test with different sequence lengths
    for seq_len in [1024, 4096, 16384]:
        print(f"\nSequence length: {seq_len}")
        
        # Create dummy input
        x = torch.randn(1, 1, seq_len, head_dim)
        
        try:
            cos_std, sin_std = rope_standard(x, seq_len)
            cos_scaled, sin_scaled = rope_scaled(x, seq_len)
            
            print(f"  Standard RoPE: cos shape {cos_std.shape}, sin shape {sin_std.shape}")
            print(f"  Scaled RoPE: cos shape {cos_scaled.shape}, sin shape {sin_scaled.shape}")
            
        except Exception as e:
            print(f"  Error: {e}")


def tokenizer_multilingual_example():
    """Example of tokenizer with multilingual text"""
    print("\n=== Multilingual Tokenizer Example ===")
    
    tokenizer = SLMTokenizer(vocab_size=128256)
    
    # Test with different languages
    texts = [
        "Hello, how are you?",  # English
        "Hola, ¿cómo estás?",  # Spanish
        "Bonjour, comment allez-vous?",  # French
        "Guten Tag, wie geht es Ihnen?",  # German
        "你好，你怎么样？",  # Chinese
        "print('Hello, World!')",  # Code
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"  # More code
    ]
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"\nOriginal: {text}")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Token IDs ({len(token_ids)}): {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"Decoded: {decoded}")


def model_architecture_analysis():
    """Analyze model architecture and memory usage"""
    print("\n=== Model Architecture Analysis ===")
    
    # Different model sizes
    configs = [
        ("Tiny", SLMConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2)),
        ("Small", SLMConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=16, num_key_value_heads=4)),
        ("Medium", SLMConfig(hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=4)),
        ("Large", SLMConfig(hidden_size=2048, num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8)),
    ]
    
    for name, config in configs:
        model = SLMForCausalLM(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough calculation)
        param_memory_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
        
        print(f"\n{name} Model:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads} (KV heads: {config.num_key_value_heads})")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Estimated model memory: {param_memory_mb:.1f} MB")


def save_and_load_example():
    """Example of saving and loading models"""
    print("\n=== Save and Load Example ===")
    
    # Create model and tokenizer
    config = SLMConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2
    )
    
    tokenizer = SLMTokenizer(vocab_size=config.vocab_size)
    model = SLMForCausalLM(config)
    
    save_dir = "./test_model"
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    
    # Save model config (exclude computed attributes)
    config_path = Path(save_dir) / "config.json"
    config_dict = {k: v for k, v in config.__dict__.items() 
                   if k not in ['head_dim', 'num_queries_per_kv']}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Save model weights
    model_path = Path(save_dir) / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
    
    # Load everything back
    loaded_tokenizer = SLMTokenizer.from_pretrained(save_dir)
    
    with open(config_path, 'r') as f:
        loaded_config_dict = json.load(f)
    loaded_config = SLMConfig(**loaded_config_dict)
    
    loaded_model = SLMForCausalLM(loaded_config)
    loaded_model.load_state_dict(torch.load(model_path))
    
    print("Model and tokenizer loaded successfully!")
    
    # Verify they work
    test_text = "This is a test"
    original_tokens = tokenizer.encode(test_text)
    loaded_tokens = loaded_tokenizer.encode(test_text)
    
    print(f"Original tokenization: {original_tokens}")
    print(f"Loaded tokenization: {loaded_tokens}")
    print(f"Tokenizations match: {original_tokens == loaded_tokens}")


def main():
    """Run all examples"""
    print("Small Language Model Implementation Examples")
    print("=" * 50)
    
    # Run examples
    try:
        model_architecture_analysis()
        tokenizer_multilingual_example()
        rope_scaling_example()
        save_and_load_example()
        inference_example()
        
        # Comment out training example to avoid long execution time
        # Uncomment to run actual training
        # basic_training_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()