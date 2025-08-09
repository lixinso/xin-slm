"""
Comprehensive Test Suite for GPT-OSS MoE Model
Tests model functionality, quantization, and Mac Mini compatibility

Features:
- Unit tests for model components
- Integration tests for training and inference
- Memory usage validation
- Performance benchmarking
- Quantization testing
- Mac Mini specific tests
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import unittest
from pathlib import Path
from typing import Dict, Any
import warnings
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import (
    GPTOSSMoEConfig, GPTOSSForCausalLM, create_gpt_oss_moe,
    RMSNorm, RotaryEmbedding, GroupedQueryAttention, MoELayer, Expert
)
from models.quantization import QuantizedLinear, ModelQuantizer, QuantizationConfig


class TestGPTOSSComponents(unittest.TestCase):
    """Test individual model components"""
    
    def setUp(self):
        """Setup test environment"""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.config = GPTOSSMoEConfig(
            vocab_size=1000,  # Small vocab for testing
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            pad_token_id=999,  # Within vocab range
            bos_token_id=999,
            eos_token_id=999
        )
        
    def test_rms_norm(self):
        """Test RMSNorm layer"""
        rms_norm = RMSNorm(self.config.hidden_size).to(self.device)
        
        # Test input
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # Forward pass
        output = rms_norm(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check normalization (approximately unit variance)
        variance = output.pow(2).mean(dim=-1)
        self.assertTrue(torch.allclose(variance, torch.ones_like(variance), atol=0.1))
        
        print("✅ RMSNorm test passed")
    
    def test_rotary_embedding(self):
        """Test RotaryEmbedding"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        rope = RotaryEmbedding(head_dim, max_position_embeddings=self.config.max_position_embeddings).to(self.device)
        
        seq_len = 32
        x = torch.randn(2, seq_len, head_dim).to(self.device)
        
        cos, sin = rope(x, seq_len)
        
        # Check output shapes
        self.assertEqual(cos.shape, (seq_len, head_dim))
        self.assertEqual(sin.shape, (seq_len, head_dim))
        
        print("✅ RotaryEmbedding test passed")
    
    def test_expert(self):
        """Test Expert layer"""
        expert = Expert(self.config).to(self.device)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        output = expert(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that output is different from input (expert should transform)
        self.assertFalse(torch.allclose(output, x))
        
        print("✅ Expert test passed")
    
    def test_moe_layer(self):
        """Test MoE layer"""
        moe = MoELayer(self.config).to(self.device)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        output, aux_losses = moe(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check auxiliary losses
        self.assertIsInstance(aux_losses, dict)
        self.assertIn("router_z_loss", aux_losses)
        self.assertIn("router_aux_loss", aux_losses)
        
        print("✅ MoE layer test passed")
    
    def test_grouped_query_attention(self):
        """Test GroupedQueryAttention"""
        attention = GroupedQueryAttention(self.config).to(self.device)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # Create attention mask
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len).to(self.device)
        attention_mask[:, :, :, :] = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
        outputs = attention(x, attention_mask=attention_mask)
        output = outputs[0]
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        print("✅ GroupedQueryAttention test passed")


class TestQuantization(unittest.TestCase):
    """Test quantization functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.config = QuantizationConfig(
            bits=4,
            group_size=32,
            use_mps_kernels=True,
            quantize_moe_experts=True
        )
    
    def test_quantized_linear(self):
        """Test QuantizedLinear layer"""
        # Create quantized linear layer
        layer = QuantizedLinear(256, 128, bias=True, config=self.config).to(self.device)
        
        # Test input
        batch_size, seq_len, hidden_size = 2, 16, 256
        x = torch.randn(batch_size, seq_len, hidden_size).to(self.device)
        
        # Forward pass before quantization
        output_original = layer(x)
        original_stats = layer.get_memory_stats()
        
        # Quantize
        layer.quantize_weights()
        
        # Forward pass after quantization
        output_quantized = layer(x)
        quantized_stats = layer.get_memory_stats()
        
        # Check shapes
        self.assertEqual(output_original.shape, output_quantized.shape)
        self.assertEqual(output_original.shape, (batch_size, seq_len, 128))
        
        # Check compression
        self.assertTrue(layer.is_quantized)
        self.assertIn("compression_ratio", quantized_stats)
        self.assertGreater(quantized_stats["compression_ratio"], 1.0)
        
        print(f"✅ Quantization test passed - Compression: {quantized_stats['compression_ratio']:.2f}x")
    
    def test_model_quantizer(self):
        """Test full model quantization"""
        # Create small model for testing
        config = GPTOSSMoEConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            pad_token_id=999,
            bos_token_id=999,
            eos_token_id=999
        )
        
        model = GPTOSSForCausalLM(config).to(self.device)
        
        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters())
        
        # Quantize model
        quantizer = ModelQuantizer(self.config)
        quantized_model = quantizer.quantize_model(model)
        
        # Get compression stats
        stats = quantizer.get_model_compression_stats(quantized_model)
        
        # Check stats
        self.assertGreater(stats["quantized_parameters"], 0)
        self.assertGreater(stats["compression_ratio"], 1.0)
        
        print(f"✅ Model quantization test passed")
        print(f"   Quantized parameters: {stats['quantized_parameters']:,}")
        print(f"   Compression ratio: {stats['compression_ratio']:.2f}x")


class TestModelFunctionality(unittest.TestCase):
    """Test full model functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model = create_gpt_oss_moe(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=512,
            num_experts=8,
            num_experts_per_tok=2,
            use_quantization=False,  # Test without quantization first
            pad_token_id=999,
            bos_token_id=999,
            eos_token_id=999
        ).to(self.device)
        
    def test_forward_pass(self):
        """Test model forward pass"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs[0]
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 1000)
        self.assertEqual(logits.shape, expected_shape)
        
        # Check that logits are not all zeros or NaN
        self.assertFalse(torch.allclose(logits, torch.zeros_like(logits)))
        self.assertFalse(torch.isnan(logits).any())
        
        print("✅ Forward pass test passed")
    
    def test_generation(self):
        """Test text generation"""
        self.model.eval()
        
        # Input prompt
        input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
        
        with torch.no_grad():
            # Generate a few tokens
            for _ in range(5):
                outputs = self.model(input_ids)
                logits = outputs[0]
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Check that sequence length increased
        self.assertEqual(input_ids.shape[1], 15)
        
        print("✅ Generation test passed")
    
    def test_training_step(self):
        """Test training step"""
        self.model.train()
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        labels = input_ids.clone()
        
        # Forward pass with loss
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        
        # Check loss properties
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
        
        # Test backward pass
        loss.backward()
        
        # Check that gradients exist
        param_with_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                param_with_grad = True
                break
        
        self.assertTrue(param_with_grad)
        
        print("✅ Training step test passed")


class TestMacMiniCompatibility(unittest.TestCase):
    """Test Mac Mini specific functionality"""
    
    def setUp(self):
        """Setup Mac Mini environment"""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Testing on device: {self.device}")
        
    def test_mps_availability(self):
        """Test MPS availability and functionality"""
        if torch.backends.mps.is_available():
            print("✅ MPS is available")
            
            # Test basic tensor operations
            x = torch.randn(100, 100).to("mps")
            y = torch.randn(100, 100).to("mps")
            z = torch.mm(x, y)
            
            self.assertEqual(z.device.type, "mps")
            print("✅ MPS tensor operations work")
        else:
            print("⚠️  MPS not available, using CPU")
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        # Create model optimized for Mac Mini
        model = create_gpt_oss_moe(
            vocab_size=50257,
            hidden_size=512,  # Smaller for memory test
            num_layers=12,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=1024,
            num_experts=16,
            num_experts_per_tok=2,
            use_quantization=True
        ).to(self.device)
        
        # Get model info
        model_info = model.get_model_info()
        estimated_memory = model_info["active_size_mb"]
        
        print(f"✅ Model memory test passed")
        print(f"   Estimated active memory: {estimated_memory:.1f} MB")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Active parameters: {model_info['active_parameters']:,}")
        
        # Check that memory usage is reasonable for 16GB Mac Mini
        self.assertLess(estimated_memory, 8000)  # Less than 8GB for model
    
    def test_inference_performance(self):
        """Test inference performance"""
        model = create_gpt_oss_moe(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=512,
            num_experts=8,
            num_experts_per_tok=2,
            use_quantization=True,
            pad_token_id=999,
            bos_token_id=999,
            eos_token_id=999
        ).to(self.device)
        
        model.eval()
        
        batch_size, seq_len = 1, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Time inference
        num_runs = 10
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(input_ids)
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Cleanup
            if self.device.type == "mps":
                torch.mps.empty_cache()
        
        avg_time = np.mean(times)
        tokens_per_second = (batch_size * seq_len) / avg_time
        
        print(f"✅ Performance test passed")
        print(f"   Average inference time: {avg_time*1000:.2f} ms")
        print(f"   Tokens per second: {tokens_per_second:.1f}")
        
        # Check reasonable performance (should be > 10 tokens/second)
        self.assertGreater(tokens_per_second, 10)


def run_smoke_tests():
    """Run quick smoke tests"""
    print("Running smoke tests...")
    
    # Test 1: Model creation
    try:
        model = create_gpt_oss_moe(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            pad_token_id=999,
            bos_token_id=999,
            eos_token_id=999
        )
        print("✅ Model creation smoke test passed")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 2: Forward pass
    try:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = model.to(device)
        input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print("✅ Forward pass smoke test passed")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 3: Quantization
    try:
        config = QuantizationConfig()
        quantizer = ModelQuantizer(config)
        quantized_model = quantizer.quantize_model(model)
        print("✅ Quantization smoke test passed")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False
    
    print("🎉 All smoke tests passed!")
    return True


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    print("=" * 60)
    print("GPT-OSS MoE Model Test Suite")
    print("=" * 60)
    
    # Run smoke tests first
    if not run_smoke_tests():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Running Full Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGPTOSSComponents,
        TestQuantization,
        TestModelFunctionality,
        TestMacMiniCompatibility
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 All tests passed! Model is ready for Mac Mini deployment.")
        sys.exit(0)
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed.")
        sys.exit(1)