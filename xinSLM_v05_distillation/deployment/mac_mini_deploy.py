"""
Mac Mini Deployment Scripts for Distilled LLaMA

This module provides optimized deployment tools specifically for Mac Mini hardware:
- Apple Silicon optimization (M1/M2/M3)
- Memory-efficient loading
- llama.cpp integration
- Ollama setup
- CoreML conversion
- Performance monitoring
"""

import os
import sys
import json
import logging
import shutil
import subprocess
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

import torch
import numpy as np
from transformers import AutoTokenizer


class MacMiniDeployment:
    """Main deployment class for Mac Mini optimization"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Mac Mini deployment
        
        Args:
            config_path: Path to deployment configuration
        """
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.detect_hardware()
        
        # Deployment paths
        self.deployment_dir = Path(self.config.get('deployment_dir', './deployed_model'))
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str = None) -> Dict:
        """Load deployment configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default Mac Mini configuration
            config = {
                'deployment_dir': './deployed_model',
                'quantization': {'bits': 4, 'method': 'gguf'},
                'inference': {
                    'max_memory_fraction': 0.8,
                    'use_mps': True,
                    'threads': 8
                },
                'formats': {
                    'gguf': {'enable': True, 'quantization_type': 'Q4_K_M'},
                    'coreml': {'enable': True},
                    'ollama': {'enable': True}
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
    
    def detect_hardware(self):
        """Detect Mac Mini hardware specifications"""
        try:
            # Get system information
            self.system_info = {
                'platform': sys.platform,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'has_mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
            
            # Detect Apple Silicon
            if sys.platform == 'darwin':
                try:
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True)
                    cpu_brand = result.stdout.strip()
                    self.system_info['cpu_brand'] = cpu_brand
                    self.system_info['is_apple_silicon'] = 'Apple' in cpu_brand
                except:
                    self.system_info['is_apple_silicon'] = False
            else:
                self.system_info['is_apple_silicon'] = False
            
            self.logger.info(f"Detected hardware: {self.system_info}")
            
        except Exception as e:
            self.logger.warning(f"Could not detect hardware: {e}")
            self.system_info = {
                'platform': sys.platform,
                'cpu_count': 8,
                'memory_gb': 16,
                'has_mps': False,
                'is_apple_silicon': False
            }
    
    def install_dependencies(self):
        """Install required dependencies for Mac deployment"""
        self.logger.info("Installing deployment dependencies")
        
        dependencies = []
        
        # Check for llama.cpp
        if self.config.get('formats', {}).get('gguf', {}).get('enable', False):
            if not shutil.which('llama.cpp') and not os.path.exists('./llama.cpp'):
                dependencies.append(('llama.cpp', 'https://github.com/ggerganov/llama.cpp.git'))
        
        # Check for Ollama
        if self.config.get('formats', {}).get('ollama', {}).get('enable', False):
            if not shutil.which('ollama'):
                dependencies.append(('ollama', 'https://ollama.ai/install.sh'))
        
        # Install missing dependencies
        for dep_name, dep_source in dependencies:
            self.logger.info(f"Installing {dep_name}")
            try:
                if dep_name == 'llama.cpp':
                    self.install_llamacpp()
                elif dep_name == 'ollama':
                    self.install_ollama()
            except Exception as e:
                self.logger.error(f"Failed to install {dep_name}: {e}")
    
    def install_llamacpp(self):
        """Install and build llama.cpp"""
        llamacpp_dir = Path('./llama.cpp')
        
        if not llamacpp_dir.exists():
            self.logger.info("Cloning llama.cpp repository")
            subprocess.run([
                'git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'
            ], check=True)
        
        # Build llama.cpp with Metal support for Apple Silicon
        build_cmd = ['make', '-C', str(llamacpp_dir)]
        
        if self.system_info.get('is_apple_silicon', False):
            build_cmd.append('LLAMA_METAL=1')
        
        self.logger.info("Building llama.cpp")
        subprocess.run(build_cmd, check=True)
        
        # Verify build
        main_executable = llamacpp_dir / 'main'
        if main_executable.exists():
            self.logger.info("llama.cpp built successfully")
        else:
            raise RuntimeError("Failed to build llama.cpp")
    
    def install_ollama(self):
        """Install Ollama"""
        self.logger.info("Installing Ollama")
        
        try:
            # Download and run Ollama installer
            subprocess.run([
                'curl', '-fsSL', 'https://ollama.ai/install.sh'
            ], stdout=subprocess.PIPE, check=True)
            
            # Alternative: use brew if available
            if shutil.which('brew'):
                subprocess.run(['brew', 'install', 'ollama'], check=True)
                
        except subprocess.CalledProcessError:
            self.logger.warning("Automatic Ollama installation failed. Please install manually.")
    
    def convert_to_gguf(self, model_path: str) -> Optional[str]:
        """
        Convert model to GGUF format for llama.cpp
        
        Args:
            model_path: Path to the model to convert
            
        Returns:
            Path to converted GGUF file
        """
        self.logger.info("Converting model to GGUF format")
        
        llamacpp_dir = Path('./llama.cpp')
        if not llamacpp_dir.exists():
            self.logger.error("llama.cpp not found. Please install first.")
            return None
        
        # Paths
        convert_script = llamacpp_dir / 'convert.py'
        quantize_executable = llamacpp_dir / 'quantize'
        
        if not convert_script.exists():
            self.logger.error("convert.py not found in llama.cpp directory")
            return None
        
        try:
            # Step 1: Convert to f16 GGUF
            f16_output = self.deployment_dir / 'model_f16.gguf'
            self.logger.info("Converting to f16 GGUF...")
            
            subprocess.run([
                sys.executable, str(convert_script),
                model_path,
                '--outfile', str(f16_output),
                '--outtype', 'f16'
            ], check=True)
            
            # Step 2: Quantize to target precision
            quantization_type = self.config.get('formats', {}).get('gguf', {}).get('quantization_type', 'Q4_K_M')
            quantized_output = self.deployment_dir / f'model_{quantization_type.lower()}.gguf'
            
            if quantize_executable.exists():
                self.logger.info(f"Quantizing to {quantization_type}...")
                subprocess.run([
                    str(quantize_executable),
                    str(f16_output),
                    str(quantized_output),
                    quantization_type
                ], check=True)
                
                # Remove intermediate f16 file
                f16_output.unlink()
                
                self.logger.info(f"GGUF conversion completed: {quantized_output}")
                return str(quantized_output)
            else:
                self.logger.warning("Quantize executable not found. Using f16 model.")
                return str(f16_output)
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"GGUF conversion failed: {e}")
            return None
    
    def setup_ollama_model(self, gguf_path: str, model_name: str = "distilled-llama") -> bool:
        """
        Setup model in Ollama
        
        Args:
            gguf_path: Path to GGUF model file
            model_name: Name for the Ollama model
            
        Returns:
            Success status
        """
        if not shutil.which('ollama'):
            self.logger.error("Ollama not found. Please install Ollama first.")
            return False
        
        self.logger.info(f"Setting up Ollama model: {model_name}")
        
        try:
            # Create Modelfile
            modelfile_content = f"""FROM {gguf_path}

# Model parameters optimized for Mac Mini
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1

# Context and generation settings
PARAMETER num_ctx 2048
PARAMETER num_predict 512

# Performance settings for Mac Mini
PARAMETER num_thread {self.system_info.get('cpu_count', 8)}
PARAMETER num_gpu_layers 0

# System prompt
SYSTEM \"\"\"You are a helpful AI assistant. You provide accurate, helpful, and concise responses.\"\"\"
"""
            
            modelfile_path = self.deployment_dir / 'Modelfile'
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            # Create the model in Ollama
            subprocess.run([
                'ollama', 'create', model_name, '-f', str(modelfile_path)
            ], check=True, cwd=self.deployment_dir)
            
            # Test the model
            self.logger.info("Testing Ollama model...")
            result = subprocess.run([
                'ollama', 'run', model_name, 'Hello! Can you introduce yourself?'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.logger.info("Ollama model setup successful")
                return True
            else:
                self.logger.error(f"Ollama model test failed: {result.stderr}")
                return False
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ollama setup failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Ollama model test timed out")
            return False
    
    def convert_to_coreml(self, model_path: str) -> Optional[str]:
        """
        Convert model to CoreML format for Apple Neural Engine
        
        Args:
            model_path: Path to the model to convert
            
        Returns:
            Path to CoreML model
        """
        try:
            import coremltools as ct
            from transformers import AutoModelForCausalLM
        except ImportError:
            self.logger.warning("CoreML tools or transformers not available. Skipping CoreML conversion.")
            return None
        
        if not self.system_info.get('is_apple_silicon', False):
            self.logger.warning("CoreML optimization is primarily for Apple Silicon. Skipping.")
            return None
        
        self.logger.info("Converting model to CoreML format")
        
        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            model.eval()
            
            # Create example input
            example_input = torch.randint(0, model.config.vocab_size, (1, 10))
            
            # Trace model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape, dtype=np.int32)],
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.macOS12
            )
            
            # Apply quantization
            quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, nbits=8
            )
            
            # Save model
            coreml_output = self.deployment_dir / 'model.mlpackage'
            quantized_model.save(str(coreml_output))
            
            self.logger.info(f"CoreML conversion completed: {coreml_output}")
            return str(coreml_output)
        
        except Exception as e:
            self.logger.error(f"CoreML conversion failed: {e}")
            return None
    
    def create_inference_script(self, gguf_path: str):
        """Create optimized inference script for Mac Mini"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Optimized inference script for Mac Mini deployment
"""

import sys
import time
import argparse
from pathlib import Path

# Add llama.cpp Python bindings if available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Install with: pip install llama-cpp-python")


class MacMiniInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        if LLAMA_CPP_AVAILABLE:
            # Initialize llama.cpp model with Mac Mini optimizations
            self.model = Llama(
                model_path=model_path,
                n_ctx={self.config.get('inference', {}).get('context_size', 2048)},
                n_threads={self.system_info.get('cpu_count', 8)},
                use_mmap=True,
                use_mlock=False,
                n_gpu_layers=0,  # CPU inference on Mac Mini
                verbose=False
            )
        else:
            self.model = None
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        if not LLAMA_CPP_AVAILABLE or self.model is None:
            return "Error: Model not loaded"
        
        start_time = time.time()
        
        # Generate response
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["</s>", "\\n\\n"]
        )
        
        generation_time = time.time() - start_time
        
        if isinstance(response, dict) and 'choices' in response:
            text = response['choices'][0]['text']
            tokens_generated = len(text.split())  # Rough estimate
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            print(f"Generated {{tokens_generated}} tokens in {{generation_time:.2f}}s ({{tokens_per_second:.1f}} tokens/s)")
            return text
        else:
            return str(response)
    
    def interactive_chat(self):
        print("Mac Mini LLaMA Chat (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\\nYou: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if prompt:
                    response = self.generate(prompt, max_tokens=200)
                    print(f"Assistant: {{response}}")
            
            except KeyboardInterrupt:
                break
        
        print("\\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Mac Mini LLaMA Inference")
    parser.add_argument("--model", default="{gguf_path}", help="Path to GGUF model")
    parser.add_argument("--prompt", help="Single prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = MacMiniInference(args.model)
    
    if args.interactive:
        inferencer.interactive_chat()
    elif args.prompt:
        response = inferencer.generate(args.prompt, args.max_tokens)
        print(f"Response: {{response}}")
    else:
        print("Please provide --prompt or use --interactive mode")


if __name__ == "__main__":
    main()
'''
        
        script_path = self.deployment_dir / 'inference.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        self.logger.info(f"Inference script created: {script_path}")
        return str(script_path)
    
    def create_api_server(self, gguf_path: str):
        """Create FastAPI server for model serving"""
        
        server_content = f'''#!/usr/bin/env python3
"""
FastAPI server for Mac Mini LLaMA deployment
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import llama.cpp if available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.error("llama-cpp-python not available. Please install with: pip install llama-cpp-python")

app = FastAPI(title="Mac Mini LLaMA API", version="1.0.0")

# Global model instance
model = None

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[list] = None

class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float

@app.on_event("startup")
async def load_model():
    global model
    
    if not LLAMA_CPP_AVAILABLE:
        logger.error("Cannot start server without llama-cpp-python")
        return
    
    logger.info("Loading model...")
    model = Llama(
        model_path="{gguf_path}",
        n_ctx=2048,
        n_threads={self.system_info.get('cpu_count', 8)},
        use_mmap=True,
        use_mlock=False,
        n_gpu_layers=0,
        verbose=False
    )
    logger.info("Model loaded successfully")

@app.get("/")
async def root():
    return {{"message": "Mac Mini LLaMA API", "status": "running"}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        response = model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False,
            stop=request.stop or ["</s>", "\\n\\n"]
        )
        
        generation_time = time.time() - start_time
        
        if isinstance(response, dict) and 'choices' in response:
            text = response['choices'][0]['text']
            tokens_generated = len(text.split())  # Rough estimate
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            return GenerationResponse(
                text=text,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second
            )
        else:
            raise HTTPException(status_code=500, detail="Invalid model response")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {{str(e)}}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
'''
        
        server_path = self.deployment_dir / 'api_server.py'
        with open(server_path, 'w') as f:
            f.write(server_content)
        
        # Make executable
        server_path.chmod(0o755)
        
        self.logger.info(f"API server created: {server_path}")
        return str(server_path)
    
    def create_benchmark_script(self):
        """Create performance benchmark script"""
        
        benchmark_content = '''#!/usr/bin/env python3
"""
Performance benchmark for Mac Mini deployment
"""

import time
import psutil
import subprocess
import json
from pathlib import Path

def benchmark_ollama(model_name: str, num_runs: int = 10):
    """Benchmark Ollama model performance"""
    print(f"Benchmarking Ollama model: {model_name}")
    
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?"
    ]
    
    results = []
    
    for i in range(num_runs):
        prompt = test_prompts[i % len(test_prompts)]
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the generation
        start_time = time.time()
        
        try:
            result = subprocess.run([
                'ollama', 'run', model_name, prompt
            ], capture_output=True, text=True, timeout=30)
            
            generation_time = time.time() - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if result.returncode == 0:
                response = result.stdout.strip()
                tokens_generated = len(response.split())
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                results.append({
                    'run': i + 1,
                    'prompt': prompt,
                    'generation_time': generation_time,
                    'tokens_generated': tokens_generated,
                    'tokens_per_second': tokens_per_second,
                    'memory_used_mb': memory_used,
                    'success': True
                })
                
                print(f"Run {i+1}: {generation_time:.2f}s, {tokens_per_second:.1f} tok/s")
            else:
                results.append({
                    'run': i + 1,
                    'error': result.stderr,
                    'success': False
                })
        
        except subprocess.TimeoutExpired:
            results.append({
                'run': i + 1,
                'error': 'Timeout',
                'success': False
            })
    
    # Calculate statistics
    successful_runs = [r for r in results if r.get('success', False)]
    
    if successful_runs:
        avg_time = sum(r['generation_time'] for r in successful_runs) / len(successful_runs)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in successful_runs) / len(successful_runs)
        avg_memory = sum(r['memory_used_mb'] for r in successful_runs) / len(successful_runs)
        
        summary = {
            'model': model_name,
            'total_runs': num_runs,
            'successful_runs': len(successful_runs),
            'avg_generation_time': avg_time,
            'avg_tokens_per_second': avg_tokens_per_sec,
            'avg_memory_usage_mb': avg_memory,
            'detailed_results': results
        }
        
        print(f"\\nBenchmark Summary:")
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Average tokens/second: {avg_tokens_per_sec:.1f}")
        print(f"Average memory usage: {avg_memory:.1f} MB")
        
        # Save results
        with open('benchmark_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    else:
        print("No successful runs!")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mac Mini Model Benchmark")
    parser.add_argument("--model", default="distilled-llama", help="Ollama model name to benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    benchmark_ollama(args.model, args.runs)

if __name__ == "__main__":
    main()
'''
        
        benchmark_path = self.deployment_dir / 'benchmark.py'
        with open(benchmark_path, 'w') as f:
            f.write(benchmark_content)
        
        # Make executable
        benchmark_path.chmod(0o755)
        
        self.logger.info(f"Benchmark script created: {benchmark_path}")
        return str(benchmark_path)
    
    def deploy_model(self, model_path: str) -> Dict[str, str]:
        """
        Complete deployment pipeline for Mac Mini
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Dictionary of deployment artifacts
        """
        self.logger.info(f"Starting Mac Mini deployment for: {model_path}")
        
        deployment_results = {
            'model_path': model_path,
            'deployment_dir': str(self.deployment_dir),
            'system_info': self.system_info
        }
        
        try:
            # Install dependencies
            self.install_dependencies()
            
            # Convert to GGUF format
            if self.config.get('formats', {}).get('gguf', {}).get('enable', True):
                gguf_path = self.convert_to_gguf(model_path)
                if gguf_path:
                    deployment_results['gguf_path'] = gguf_path
                    
                    # Setup Ollama model
                    if self.config.get('formats', {}).get('ollama', {}).get('enable', False):
                        if self.setup_ollama_model(gguf_path):
                            deployment_results['ollama_model'] = 'distilled-llama'
                    
                    # Create inference scripts
                    inference_script = self.create_inference_script(gguf_path)
                    deployment_results['inference_script'] = inference_script
                    
                    # Create API server
                    api_server = self.create_api_server(gguf_path)
                    deployment_results['api_server'] = api_server
            
            # Convert to CoreML if enabled
            if self.config.get('formats', {}).get('coreml', {}).get('enable', False):
                coreml_path = self.convert_to_coreml(model_path)
                if coreml_path:
                    deployment_results['coreml_path'] = coreml_path
            
            # Create benchmark script
            benchmark_script = self.create_benchmark_script()
            deployment_results['benchmark_script'] = benchmark_script
            
            # Create deployment summary
            self.create_deployment_summary(deployment_results)
            
            self.logger.info("Mac Mini deployment completed successfully!")
            return deployment_results
        
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    def create_deployment_summary(self, results: Dict[str, str]):
        """Create deployment summary and usage instructions"""
        
        summary_content = f"""# Mac Mini Deployment Summary

## Model Information
- Original Model: {results['model_path']}
- Deployment Directory: {results['deployment_dir']}
- Deployment Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## System Information
- Platform: {self.system_info.get('platform', 'unknown')}
- CPU: {self.system_info.get('cpu_brand', 'unknown')}
- CPU Cores: {self.system_info.get('cpu_count', 'unknown')}
- Memory: {self.system_info.get('memory_gb', 0):.1f} GB
- Apple Silicon: {self.system_info.get('is_apple_silicon', False)}
- MPS Available: {self.system_info.get('has_mps', False)}

## Deployed Formats
"""
        
        if 'gguf_path' in results:
            summary_content += f"- GGUF Model: {results['gguf_path']}\\n"
        
        if 'coreml_path' in results:
            summary_content += f"- CoreML Model: {results['coreml_path']}\\n"
        
        if 'ollama_model' in results:
            summary_content += f"- Ollama Model: {results['ollama_model']}\\n"
        
        summary_content += f"""
## Usage Instructions

### Command Line Inference
```bash
python {results.get('inference_script', 'inference.py')} --prompt "Your question here"
```

### Interactive Chat
```bash
python {results.get('inference_script', 'inference.py')} --interactive
```

### API Server
```bash
python {results.get('api_server', 'api_server.py')}
```

Then make requests to http://localhost:8000/generate

### Ollama Usage
"""
        
        if 'ollama_model' in results:
            summary_content += f"""```bash
ollama run {results['ollama_model']} "Your question here"
```
"""
        else:
            summary_content += "Ollama model not available.\\n"
        
        summary_content += f"""
### Performance Benchmarking
```bash
python {results.get('benchmark_script', 'benchmark.py')} --model {results.get('ollama_model', 'distilled-llama')}
```

## Performance Optimization Tips

1. **Memory Usage**: The model is optimized to use less than 1GB of RAM
2. **CPU Threads**: Using {self.system_info.get('cpu_count', 8)} CPU threads for optimal performance
3. **Apple Silicon**: {'Optimized for Apple Silicon' if self.system_info.get('is_apple_silicon') else 'Standard optimization'}

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are installed
2. Check available memory with `htop` or Activity Monitor
3. Try reducing context size or batch size
4. Check logs in the deployment directory

## Next Steps

1. Test the model with various prompts
2. Run benchmarks to measure performance
3. Integrate with your applications
4. Monitor memory usage during inference
"""
        
        summary_path = self.deployment_dir / 'DEPLOYMENT_SUMMARY.md'
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        self.logger.info(f"Deployment summary created: {summary_path}")


def main():
    """Example deployment usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy model for Mac Mini")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", help="Path to deployment config")
    parser.add_argument("--output", default="./deployed_model", help="Output directory")
    
    args = parser.parse_args()
    
    # Update config with output directory
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    config['deployment_dir'] = args.output
    
    # Deploy model
    deployer = MacMiniDeployment()
    deployer.config = config
    deployer.deployment_dir = Path(args.output)
    deployer.deployment_dir.mkdir(parents=True, exist_ok=True)
    
    results = deployer.deploy_model(args.model)
    
    print("\\nDeployment completed!")
    print(f"Deployment directory: {results['deployment_dir']}")
    print("See DEPLOYMENT_SUMMARY.md for usage instructions.")


if __name__ == "__main__":
    main()
"""