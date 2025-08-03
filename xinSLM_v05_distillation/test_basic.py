#!/usr/bin/env python3
"""
Basic test of the distillation framework components
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_student_model():
    """Test student model creation and basic functionality"""
    print("Testing student model...")
    
    from models.distilled_llama import create_distilled_llama
    
    # Create a very small model for testing
    model = create_distilled_llama(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=128
    )
    
    print(f"Model created successfully")
    info = model.get_model_info()
    print(f"Parameters: {info['total_parameters']:,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0]
        print(f"Forward pass successful: {logits.shape}")
    
    return model

def test_teacher_model():
    """Test teacher model loading"""
    print("\nTesting teacher model...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"  # Even smaller than medium
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"Teacher model loaded: {model_name}")
        
        # Test tokenization
        text = "Hello, how are you?"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"Tokenization successful: {tokens['input_ids'].shape}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**tokens)
            print(f"Teacher forward pass successful: {outputs.logits.shape}")
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Teacher model test failed: {e}")
        return None, None

def test_distillation_loss():
    """Test the distillation loss function"""
    print("\nTesting distillation loss...")
    
    from scripts.knowledge_distillation import KnowledgeDistillationLoss
    
    # Create loss function
    loss_fn = KnowledgeDistillationLoss(alpha=0.5, beta=0.5, temperature=2.0)
    
    # Create dummy data
    vocab_size = 1000
    seq_len = 10
    batch_size = 2
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Compute loss
    loss, loss_dict = loss_fn(student_logits, teacher_logits, labels)
    
    print(f"Loss computation successful: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    return loss_fn

def main():
    """Run all tests"""
    print("🧪 Running xinSLM v05 distillation basic tests\n")
    
    try:
        # Test 1: Student model
        student_model = test_student_model()
        
        # Test 2: Teacher model
        teacher_model, tokenizer = test_teacher_model()
        
        # Test 3: Distillation loss
        loss_fn = test_distillation_loss()
        
        if student_model and teacher_model and tokenizer and loss_fn:
            print("\n✅ All basic tests passed!")
            print("The distillation framework components are working correctly.")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()