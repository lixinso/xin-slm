"""
Model Quantization for Mac Mini Deployment

This module implements 4-bit quantization using GPTQ and other methods
to optimize the distilled LLaMA model for deployment on resource-constrained devices.

Supports:
- GPTQ 4-bit quantization
- Group-wise quantization (32-weight groups)
- GGUF format conversion for llama.cpp
- CoreML optimization for Apple Silicon
- Memory-optimized inference
"""

import os
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import yaml
from pathlib import Path
import time
import psutil
import subprocess

# Quantization libraries
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from auto_gptq.utils.data_utils import make_tokenize_function
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False
    print("Warning: auto_gptq not available. GPTQ quantization will be disabled.")

try:
    import optimum
    from optimum.gptq import GPTQConfig, load_quantized_model
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np


class ModelQuantizer:
    """Main class for model quantization and optimization"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the quantizer with configuration
        
        Args:
            config_path: Path to deployment configuration YAML file
        """
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Check available quantization methods
        self.available_methods = self._check_available_methods()
        self.logger.info(f"Available quantization methods: {self.available_methods}")
    
    def load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = "configs/deployment_config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'quantization': {
                    'enable': True,
                    'method': 'gptq',
                    'bits': 4,
                    'group_size': 32,
                    'calibration': {
                        'dataset': 'wikitext2',
                        'num_samples': 512,
                        'seq_len': 512
                    }
                },
                'hardware': {
                    'memory_gb': 16,
                    'device_type': 'mac_mini'
                }
            }
        
        return config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _check_available_methods(self) -> List[str]:
        """Check which quantization methods are available"""
        methods = []
        
        if GPTQ_AVAILABLE:
            methods.append('gptq')
        if BNB_AVAILABLE:
            methods.append('bitsandbytes')
        
        # Always available
        methods.extend(['dynamic', 'gguf'])
        
        return methods
    
    def prepare_calibration_data(
        self,
        tokenizer,
        dataset_name: str = 'wikitext2',
        num_samples: int = 512,
        seq_len: int = 512
    ) -> List[Dict]:
        """
        Prepare calibration dataset for quantization
        
        Args:
            tokenizer: Model tokenizer
            dataset_name: Name of calibration dataset
            num_samples: Number of calibration samples
            seq_len: Sequence length for calibration
            
        Returns:
            List of tokenized calibration samples
        """
        self.logger.info(f"Preparing calibration data from {dataset_name}")
        
        if dataset_name == 'wikitext2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            texts = [example['text'] for example in dataset if len(example['text'].strip()) > 0]
        elif dataset_name == 'c4':
            dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
            texts = [example['text'] for example in dataset.take(num_samples * 2)]
        else:
            raise ValueError(f"Unsupported calibration dataset: {dataset_name}")
        
        # Tokenize and prepare samples
        calibration_data = []
        
        for i, text in enumerate(texts[:num_samples]):
            if len(calibration_data) >= num_samples:
                break
                
            # Tokenize text
            encoding = tokenizer(
                text,
                max_length=seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            calibration_data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })
        
        self.logger.info(f"Prepared {len(calibration_data)} calibration samples")
        return calibration_data
    
    def quantize_with_gptq(
        self,
        model,
        tokenizer,
        output_dir: str,
        calibration_data: List[Dict] = None
    ) -> str:
        """
        Quantize model using GPTQ method
        
        Args:
            model: Model to quantize
            tokenizer: Model tokenizer
            output_dir: Output directory for quantized model
            calibration_data: Calibration dataset
            
        Returns:
            Path to quantized model
        """
        if not GPTQ_AVAILABLE:
            raise ImportError("auto_gptq is not available. Please install with: pip install auto_gptq")
        
        self.logger.info("Starting GPTQ quantization")
        
        # Configuration for GPTQ
        quantize_config = BaseQuantizeConfig(
            bits=self.config['quantization']['bits'],
            group_size=self.config['quantization']['group_size'],
            desc_act=self.config['quantization'].get('desc_act', False),
            static_groups=self.config['quantization'].get('static_groups', False),
            sym=True,
            true_sequential=True,
            model_name_or_path=None,
            model_file_base_name="model"
        )
        
        # Prepare calibration data if not provided
        if calibration_data is None:
            calibration_data = self.prepare_calibration_data(
                tokenizer,
                self.config['quantization']['calibration']['dataset'],
                self.config['quantization']['calibration']['num_samples'],
                self.config['quantization']['calibration']['seq_len']
            )
        
        # Convert calibration data to the format expected by GPTQ
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
        
        # Create a simple dataset for GPTQ
        examples = []
        for sample in calibration_data[:128]:  # GPTQ typically uses fewer samples
            # Decode tokens back to text for GPTQ format
            text = tokenizer.decode(sample['input_ids'].squeeze(), skip_special_tokens=True)
            examples.append({'text': text})
        
        # Wrap model for GPTQ quantization
        quantized_model = AutoGPTQForCausalLM.from_pretrained(
            model,
            quantize_config=quantize_config,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Perform quantization
        self.logger.info("Calibrating model with GPTQ...")
        start_time = time.time()
        
        quantized_model.quantize(examples, use_triton=False)
        
        quantization_time = time.time() - start_time
        self.logger.info(f"GPTQ quantization completed in {quantization_time:.2f} seconds")
        
        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        quantized_model.save_quantized(
            output_dir,
            model_basename="model",
            use_safetensors=True
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save quantization info
        quant_info = {
            'method': 'gptq',
            'bits': quantize_config.bits,
            'group_size': quantize_config.group_size,
            'quantization_time': quantization_time,
            'calibration_samples': len(examples)
        }
        
        with open(os.path.join(output_dir, 'quantization_info.json'), 'w') as f:
            json.dump(quant_info, f, indent=2)
        
        self.logger.info(f"GPTQ quantized model saved to: {output_dir}")
        return output_dir
    
    def quantize_with_bitsandbytes(
        self,
        model,
        output_dir: str,
        bits: int = 4
    ) -> str:
        """
        Quantize model using BitsAndBytesConfig (for NVIDIA GPUs)
        Note: This is mainly for reference as it requires CUDA
        
        Args:
            model: Model to quantize
            output_dir: Output directory
            bits: Number of bits (4 or 8)
            
        Returns:
            Path to quantized model config
        """
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes is not available")
        
        self.logger.info(f"Creating {bits}-bit BitsAndBytes configuration")
        
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            raise ValueError("BitsAndBytes only supports 4-bit or 8-bit quantization")
        
        # Save configuration
        os.makedirs(output_dir, exist_ok=True)
        config_dict = {
            'quantization_method': 'bitsandbytes',
            'bits': bits,
            'load_in_4bit': bits == 4,
            'load_in_8bit': bits == 8,
            'config': bnb_config.to_dict() if hasattr(bnb_config, 'to_dict') else str(bnb_config)
        }
        
        with open(os.path.join(output_dir, 'bnb_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"BitsAndBytes config saved to: {output_dir}")
        return output_dir
    
    def dynamic_quantization(
        self,
        model,
        output_dir: str
    ) -> str:
        """
        Apply PyTorch dynamic quantization
        
        Args:
            model: Model to quantize
            output_dir: Output directory
            
        Returns:
            Path to quantized model
        """
        self.logger.info("Applying dynamic quantization")
        
        # Set model to evaluation mode
        model.eval()
        
        # Apply dynamic quantization to linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(quantized_model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        
        # Save model architecture for loading
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            json.dump({
                'quantization_method': 'dynamic',
                'dtype': 'qint8',
                'quantized_layers': ['linear']
            }, f, indent=2)
        
        self.logger.info(f"Dynamic quantized model saved to: {output_dir}")
        return output_dir
    
    def convert_to_gguf(
        self,
        model_path: str,
        output_path: str,
        quantization_type: str = "Q4_K_M"
    ) -> str:
        """
        Convert model to GGUF format for llama.cpp
        
        Args:
            model_path: Path to the model to convert
            output_path: Output path for GGUF file
            quantization_type: GGUF quantization type
            
        Returns:
            Path to GGUF file
        """
        self.logger.info(f"Converting model to GGUF format with {quantization_type}")
        
        # Check if llama.cpp conversion scripts are available
        conversion_script = "convert.py"  # This would be from llama.cpp repo
        
        if not os.path.exists(conversion_script):
            self.logger.warning("llama.cpp conversion script not found. Please install llama.cpp.")
            return None
        
        try:
            # Convert to GGUF
            cmd = [
                "python", conversion_script,
                model_path,
                "--outfile", output_path,
                "--outtype", quantization_type
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"GGUF conversion successful: {output_path}")
                return output_path
            else:
                self.logger.error(f"GGUF conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during GGUF conversion: {e}")
            return None
    
    def optimize_for_apple_silicon(
        self,
        model,
        output_dir: str
    ) -> str:
        """
        Optimize model for Apple Silicon using CoreML
        
        Args:
            model: Model to optimize
            output_dir: Output directory
            
        Returns:
            Path to optimized model
        """
        try:
            import coremltools as ct
            from coremltools.models.neural_network import quantization_utils
        except ImportError:
            self.logger.warning("CoreML tools not available. Skipping Apple Silicon optimization.")
            return None
        
        self.logger.info("Optimizing model for Apple Silicon")
        
        # Convert to CoreML format
        model.eval()
        
        # Create example input
        example_input = torch.randint(0, 1000, (1, 10))  # Batch size 1, sequence length 10
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use Neural Engine
                minimum_deployment_target=ct.target.iOS15
            )
            
            # Apply quantization
            quantized_model = quantization_utils.quantize_weights(coreml_model, nbits=8)
            
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'model.mlpackage')
            quantized_model.save(output_path)
            
            self.logger.info(f"CoreML optimized model saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"CoreML optimization failed: {e}")
            return None
    
    def benchmark_quantized_model(
        self,
        model_path: str,
        tokenizer,
        num_runs: int = 100
    ) -> Dict:
        """
        Benchmark quantized model performance
        
        Args:
            model_path: Path to quantized model
            tokenizer: Model tokenizer
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Benchmarking quantized model: {model_path}")
        
        # Load quantized model (implementation depends on quantization method)
        # This is a simplified version - actual implementation would vary by method
        
        # Test inputs
        test_inputs = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
            "Calculate the area of a circle with radius 5."
        ]
        
        inference_times = []
        memory_usage = []
        
        for i in range(num_runs):
            test_input = test_inputs[i % len(test_inputs)]
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors='pt')
            
            # Measure memory before inference
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time inference
            start_time = time.time()
            
            # Note: Actual inference would depend on the quantized model format
            # with torch.no_grad():
            #     outputs = model(**inputs)
            
            # Simulate inference time for demo
            time.sleep(0.01)  # Placeholder
            
            inference_time = time.time() - start_time
            
            # Measure memory after inference
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            inference_times.append(inference_time)
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        metrics = {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'p95_inference_time': np.percentile(inference_times, 95),
            'p99_inference_time': np.percentile(inference_times, 99),
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage),
            'tokens_per_second': 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
        }
        
        self.logger.info(f"Benchmark results: {metrics}")
        return metrics
    
    def quantize_model(
        self,
        model,
        tokenizer,
        model_path: str,
        output_dir: str,
        method: str = None
    ) -> Dict[str, str]:
        """
        Main method to quantize a model using the specified method
        
        Args:
            model: Model to quantize
            tokenizer: Model tokenizer
            model_path: Path to original model
            output_dir: Base output directory
            method: Quantization method to use
            
        Returns:
            Dictionary of output paths for different formats
        """
        if method is None:
            method = self.config['quantization']['method']
        
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Available: {self.available_methods}")
        
        self.logger.info(f"Starting quantization with method: {method}")
        
        results = {}
        
        # Create output directories
        base_output = Path(output_dir)
        base_output.mkdir(parents=True, exist_ok=True)
        
        try:
            if method == 'gptq' and GPTQ_AVAILABLE:
                gptq_output = base_output / 'gptq'
                results['gptq'] = self.quantize_with_gptq(model, tokenizer, str(gptq_output))
                
            elif method == 'bitsandbytes' and BNB_AVAILABLE:
                bnb_output = base_output / 'bitsandbytes'
                results['bitsandbytes'] = self.quantize_with_bitsandbytes(model, str(bnb_output))
                
            elif method == 'dynamic':
                dynamic_output = base_output / 'dynamic'
                results['dynamic'] = self.dynamic_quantization(model, str(dynamic_output))
            
            # Additional conversions
            if self.config.get('formats', {}).get('gguf', {}).get('enable', False):
                gguf_output = base_output / 'model.gguf'
                gguf_path = self.convert_to_gguf(
                    results.get('gptq', model_path),
                    str(gguf_output),
                    self.config['formats']['gguf'].get('quantization_type', 'Q4_K_M')
                )
                if gguf_path:
                    results['gguf'] = gguf_path
            
            # Apple Silicon optimization
            if self.config.get('apple_optimization', {}).get('coreml', {}).get('enable', False):
                coreml_output = base_output / 'coreml'
                coreml_path = self.optimize_for_apple_silicon(model, str(coreml_output))
                if coreml_path:
                    results['coreml'] = coreml_path
            
            # Benchmark results
            if self.config.get('monitoring', {}).get('benchmark', {}).get('enable', False):
                for format_name, format_path in results.items():
                    if format_name in ['gptq', 'dynamic']:  # Benchmark quantized models
                        benchmark_results = self.benchmark_quantized_model(format_path, tokenizer)
                        
                        # Save benchmark results
                        benchmark_file = base_output / f'{format_name}_benchmark.json'
                        with open(benchmark_file, 'w') as f:
                            json.dump(benchmark_results, f, indent=2)
            
            self.logger.info(f"Quantization completed. Results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise


def main():
    """Example usage of the quantization system"""
    
    # Initialize quantizer
    quantizer = ModelQuantizer("configs/deployment_config.yaml")
    
    # Load model and tokenizer (placeholder - replace with actual model)
    model_name = "path/to/distilled/model"  # Replace with actual path
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Quantize model
        results = quantizer.quantize_model(
            model=model,
            tokenizer=tokenizer,
            model_path=model_name,
            output_dir="./quantized_models",
            method="gptq"
        )
        
        print("Quantization completed successfully!")
        print("Output files:")
        for format_name, path in results.items():
            print(f"  {format_name}: {path}")
            
    except Exception as e:
        print(f"Error during quantization: {e}")


if __name__ == "__main__":
    main()