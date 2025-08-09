#!/usr/bin/env python3
"""
Quick Test Script for Trained GPT-OSS MoE Model
Tests the best model with a simple prompt to verify it's working
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import GPTOSSForCausalLM, create_gpt_oss_moe

def load_trained_model(checkpoint_path: str, device: str = "mps"):
    """Load the trained model from checkpoint"""
    print(f"🔄 Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_variant = config['training'].get('model_variant', 'micro')
        variant_config = config.get('model_variants', {}).get(model_variant, {})
        
        print(f"📋 Model variant: {model_variant}")
        print(f"📊 Configuration: {variant_config}")
    else:
        # Fallback to micro config if no config in checkpoint
        variant_config = {
            'hidden_size': 384,
            'intermediate_size': 1024,
            'num_hidden_layers': 8,
            'num_attention_heads': 6,
            'num_key_value_heads': 1,
            'num_experts': 4,
            'num_experts_per_tok': 1,
            'reasoning_effort': 'low'
        }
        print("⚠️  Using default micro configuration")
    
    # Create model with the same configuration used during training
    model = create_gpt_oss_moe(
        vocab_size=50257,
        hidden_size=variant_config.get('hidden_size', 384),
        num_layers=variant_config.get('num_hidden_layers', 8),
        num_heads=variant_config.get('num_attention_heads', 6),
        num_kv_heads=variant_config.get('num_key_value_heads', 1),
        max_seq_len=512,  # Match training config
        num_experts=variant_config.get('num_experts', 4),
        num_experts_per_tok=variant_config.get('num_experts_per_tok', 1),
        reasoning_effort=variant_config.get('reasoning_effort', 'low'),
        use_quantization=False
    )
    
    # Load the trained weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully (direct state dict)")
    
    # Move to device and set eval mode
    model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Total parameters: {total_params:,}")
    
    return model

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9, device: str = "mps"):
    """Generate text from the model"""
    print(f"🎯 Prompt: '{prompt}'")
    print(f"🔧 Settings: max_length={max_length}, temperature={temperature}, top_p={top_p}")
    print("📝 Generated text:")
    print("-" * 50)
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        # Generate tokens one by one for streaming effect
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.logits
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Decode and print new token
            new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(new_token_text, end='', flush=True)
            
            # Check for end of sequence
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        print()  # New line after generation
    
    end_time = time.time()
    
    # Decode full response
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response_text = generated_text[len(prompt):].strip()
    
    print("-" * 50)
    print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
    print(f"📊 Tokens generated: {generated_ids.shape[1] - input_ids.shape[1]}")
    print(f"🚀 Speed: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.1f} tokens/sec")
    
    return response_text

def main():
    parser = argparse.ArgumentParser(description="Test trained GPT-OSS MoE model")
    parser.add_argument("--checkpoint", default="checkpoints_ultra_safe/best_model.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", default="The future of artificial intelligence is", 
                       help="Prompt to test the model with")
    parser.add_argument("--max-length", type=int, default=50, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, 
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"],
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    print("🚀 GPT-OSS MoE Model Testing")
    print("=" * 50)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("💡 Available checkpoints:")
        checkpoint_dir = Path(args.checkpoint).parent
        if checkpoint_dir.exists():
            for file in checkpoint_dir.glob("*.pt"):
                print(f"   - {file}")
        return
    
    # Setup device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"🖥️  Using device: {args.device}")
    
    try:
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Load model
        model = load_trained_model(args.checkpoint, args.device)
        
        # Test generation
        print("\n🎯 Testing model generation...")
        response = generate_text(
            model=model,
            tokenizer=tokenizer, 
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📝 Full response: '{response}'")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()