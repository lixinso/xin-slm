"""
Inference Script for GPT-OSS MoE Model on Mac Mini
Optimized for 16GB RAM with quantization and efficient generation

Features:
- Quantized model loading for memory efficiency
- Configurable reasoning effort levels
- Streaming generation support
- Metal Performance Shaders optimization
- Memory monitoring and optimization
- Interactive chat interface
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
import warnings
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import GPTOSSForCausalLM, create_gpt_oss_moe
from models.quantization import ModelQuantizer, QuantizationConfig, load_quantized_model


class GPTOSSInference:
    """Inference engine for GPT-OSS MoE model on Mac Mini"""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_device()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        # Generation settings
        self.generation_config = self.config.get('inference', {})
        
        # Performance monitoring
        self.generation_stats = []
        
        # Load model and tokenizer
        if checkpoint_path:
            self.load_model(checkpoint_path)
        else:
            self.create_model()
        
        self.load_tokenizer()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device for inference"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using Metal Performance Shaders (MPS)")
            
            # Configure MPS memory management
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        else:
            self.device = torch.device("cpu")
            self.logger.info("MPS not available, using CPU")
    
    def create_model(self):
        """Create a new model with default configuration"""
        self.logger.info("Creating new GPT-OSS MoE model...")
        
        # Get model variant configuration
        model_variant = self.config.get('model', {}).get('model_variant', 'standard')
        variant_config = self.config.get('model_variants', {}).get(model_variant, {})
        
        # Create model
        self.model = create_gpt_oss_moe(
            vocab_size=self.config['model'].get('vocab_size', 50257),
            hidden_size=variant_config.get('hidden_size', 768),
            num_layers=variant_config.get('num_hidden_layers', 20),
            num_heads=variant_config.get('num_attention_heads', 12),
            num_kv_heads=variant_config.get('num_key_value_heads', 4),
            max_seq_len=self.config['model'].get('max_position_embeddings', 2048),
            num_experts=variant_config.get('num_experts', 32),
            num_experts_per_tok=variant_config.get('num_experts_per_tok', 2),
            reasoning_effort=variant_config.get('reasoning_effort', 'medium'),
            use_quantization=self.config.get('quantization', {}).get('enable_quantization', True)
        )
        
        # Apply quantization
        if self.config.get('quantization', {}).get('enable_quantization', True):
            self._apply_quantization()
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model created with {model_info['active_parameters']:,} active parameters")
        self.logger.info(f"Estimated memory usage: {model_info['active_size_mb']:.1f} MB")
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract configuration from checkpoint
        if 'config' in checkpoint:
            self.model_config = checkpoint['config']
        
        # Create model with same configuration as checkpoint
        self.create_model()  # This will create the model structure
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.logger.info("Model loaded successfully")
    
    def _apply_quantization(self):
        """Apply quantization to model"""
        self.logger.info("Applying quantization...")
        
        quantization_config = QuantizationConfig(
            bits=self.config['quantization'].get('bits', 4),
            group_size=self.config['quantization'].get('group_size', 32),
            quantize_moe_experts=self.config['quantization'].get('quantize_moe_weights', True),
            quantize_attention=self.config['quantization'].get('quantize_attention', False),
            use_mps_kernels=self.config.get('mac_optimizations', {}).get('use_metal_performance_shaders', True)
        )
        
        quantizer = ModelQuantizer(quantization_config)
        self.model = quantizer.quantize_model(self.model)
        
        # Log quantization stats
        stats = quantizer.get_model_compression_stats(self.model)
        self.logger.info(f"Quantization applied - Compression: {stats['compression_ratio']:.2f}x")
        self.logger.info(f"Memory usage: {stats['total_memory_mb']:.1f} MB")
    
    def load_tokenizer(self):
        """Load tokenizer"""
        tokenizer_name = self.config.get('model', {}).get('tokenizer_name', 'gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info(f"Tokenizer loaded: {tokenizer_name}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None,
        do_sample: bool = True,
        stream: bool = False,
        reasoning_effort: Optional[str] = None
    ) -> str:
        """Generate text from prompt"""
        
        # Use config defaults if not provided
        max_new_tokens = max_new_tokens or self.generation_config.get('max_new_tokens', 512)
        temperature = temperature or self.generation_config.get('temperature', 0.7)
        top_p = top_p or self.generation_config.get('top_p', 0.9)
        top_k = top_k or self.generation_config.get('top_k', 50)
        repetition_penalty = repetition_penalty or self.generation_config.get('repetition_penalty', 1.1)
        
        # Adjust reasoning effort if specified
        if reasoning_effort and hasattr(self.model.config, 'reasoning_effort'):
            original_effort = self.model.config.reasoning_effort
            self.model.config.reasoning_effort = reasoning_effort
        
        start_time = time.time()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_tokens = 0
        generated_text = ""
        
        if stream:
            # Streaming generation
            for token in self._generate_streaming(
                input_ids, attention_mask, max_new_tokens, temperature, 
                top_p, top_k, repetition_penalty, do_sample
            ):
                generated_text += token
                generated_tokens += 1
                yield token
        else:
            # Non-streaming generation
            with torch.no_grad():
                generated_ids = self._generate_batch(
                    input_ids, attention_mask, max_new_tokens, temperature,
                    top_p, top_k, repetition_penalty, do_sample
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0, input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            generated_tokens = generated_ids.shape[1] - input_ids.shape[1]
        
        # Calculate performance stats
        generation_time = time.time() - start_time
        tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
        
        # Log stats
        stats = {
            'prompt_length': input_ids.shape[1],
            'generated_tokens': generated_tokens,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'temperature': temperature,
            'top_p': top_p,
            'reasoning_effort': reasoning_effort or self.model.config.reasoning_effort
        }
        
        self.generation_stats.append(stats)
        self.logger.info(
            f"Generated {generated_tokens} tokens in {generation_time:.2f}s "
            f"({tokens_per_second:.1f} tokens/s)"
        )
        
        # Restore original reasoning effort
        if reasoning_effort and hasattr(self.model.config, 'reasoning_effort'):
            self.model.config.reasoning_effort = original_effort
        
        if not stream:
            return generated_text
    
    def _generate_streaming(
        self, input_ids, attention_mask, max_new_tokens, temperature,
        top_p, top_k, repetition_penalty, do_sample
    ) -> Generator[str, None, None]:
        """Generate text with streaming output"""
        
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        past_key_values = None
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs[0][:, -1, :]  # Get logits for last token
                past_key_values = outputs[1] if len(outputs) > 1 else None
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(current_input_ids[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(1, indices_to_remove.unsqueeze(0), float('-inf'))
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            
            # Decode token and yield
            token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            yield token_text
            
            # Update input for next iteration
            current_input_ids = next_token_id
            current_attention_mask = torch.ones_like(next_token_id)
            
            # Memory cleanup for MPS
            if self.device.type == "mps":
                torch.mps.empty_cache()
    
    def _generate_batch(
        self, input_ids, attention_mask, max_new_tokens, temperature,
        top_p, top_k, repetition_penalty, do_sample
    ) -> torch.Tensor:
        """Generate text in batch mode (non-streaming)"""
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs[0][:, -1, :]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            
            # Apply sampling parameters
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(generated_ids.shape[0]):
                    for token_id in set(generated_ids[batch_idx].tolist()):
                        if logits[batch_idx, token_id] < 0:
                            logits[batch_idx, token_id] *= repetition_penalty
                        else:
                            logits[batch_idx, token_id] /= repetition_penalty
            
            # Sample next token
            if do_sample:
                # Apply top-k and top-p
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            ], dim=-1)
            
            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        return generated_ids
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("GPT-OSS MoE Interactive Chat")
        print("Type 'quit' to exit, 'stats' to see performance statistics")
        print("Commands:")
        print("  /temp <value>    - Set temperature (0.1-2.0)")
        print("  /top_p <value>   - Set top_p (0.1-1.0)")
        print("  /effort <level>  - Set reasoning effort (low/medium/high)")
        print("  /stream          - Toggle streaming mode")
        print("-" * 50)
        
        # Chat settings
        temperature = 0.7
        top_p = 0.9
        reasoning_effort = "medium"
        stream_mode = True
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._print_performance_stats()
                    continue
                elif user_input.startswith('/'):
                    # Handle commands
                    parts = user_input.split()
                    command = parts[0][1:]  # Remove '/'
                    
                    if command == 'temp' and len(parts) > 1:
                        temperature = max(0.1, min(2.0, float(parts[1])))
                        print(f"Temperature set to {temperature}")
                    elif command == 'top_p' and len(parts) > 1:
                        top_p = max(0.1, min(1.0, float(parts[1])))
                        print(f"Top-p set to {top_p}")
                    elif command == 'effort' and len(parts) > 1:
                        if parts[1] in ['low', 'medium', 'high']:
                            reasoning_effort = parts[1]
                            print(f"Reasoning effort set to {reasoning_effort}")
                    elif command == 'stream':
                        stream_mode = not stream_mode
                        print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    else:
                        print("Unknown command")
                    continue
                
                if not user_input:
                    continue
                
                print("Assistant: ", end="", flush=True)
                
                if stream_mode:
                    # Streaming generation
                    for token in self.generate(
                        user_input,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_effort=reasoning_effort,
                        stream=True
                    ):
                        print(token, end="", flush=True)
                    print()  # New line after generation
                else:
                    # Non-streaming generation
                    response = self.generate(
                        user_input,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_effort=reasoning_effort,
                        stream=False
                    )
                    print(response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if not self.generation_stats:
            print("No generation statistics available")
            return
        
        # Calculate averages
        avg_tokens_per_second = sum(s['tokens_per_second'] for s in self.generation_stats) / len(self.generation_stats)
        avg_generated_tokens = sum(s['generated_tokens'] for s in self.generation_stats) / len(self.generation_stats)
        total_tokens = sum(s['generated_tokens'] for s in self.generation_stats)
        
        print(f"\nPerformance Statistics ({len(self.generation_stats)} generations):")
        print(f"  Average tokens/second: {avg_tokens_per_second:.1f}")
        print(f"  Average tokens per generation: {avg_generated_tokens:.1f}")
        print(f"  Total tokens generated: {total_tokens}")
        
        # Show memory info if available
        if self.device.type == "mps":
            try:
                allocated = torch.mps.current_allocated_memory() / 1024**2
                print(f"  Current MPS memory: {allocated:.1f} MB")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS MoE Inference on Mac Mini")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference_engine = GPTOSSInference(args.config, args.checkpoint)
        
        if args.interactive:
            # Start interactive chat
            inference_engine.interactive_chat()
        elif args.prompt:
            # Single prompt generation
            print(f"Prompt: {args.prompt}")
            print("Generated text:")
            print("-" * 40)
            
            if args.stream:
                for token in inference_engine.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=True
                ):
                    print(token, end="", flush=True)
                print()
            else:
                response = inference_engine.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=False
                )
                print(response)
        else:
            print("No prompt provided. Use --prompt or --interactive")
    
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()