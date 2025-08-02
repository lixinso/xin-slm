"""
Comprehensive Evaluation and Benchmarking Suite for Distilled Models

This module implements evaluation across multiple dimensions:
- Language modeling perplexity
- Downstream task performance (HellaSwag, TruthfulQA, etc.)
- Efficiency metrics (speed, memory usage)
- Teacher-student comparison
- Mac Mini specific performance profiling
"""

import os
import json
import time
import logging
import torch
import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

# Evaluation libraries
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import evaluate
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    model_path: str
    tokenizer_path: str = None
    output_dir: str = "./benchmark_results"
    
    # Evaluation tasks
    eval_perplexity: bool = True
    eval_downstream: bool = True
    eval_generation: bool = True
    eval_efficiency: bool = True
    
    # Task-specific settings
    perplexity_datasets: List[str] = None
    downstream_tasks: List[str] = None
    generation_prompts: List[str] = None
    
    # Performance settings
    batch_size: int = 1
    max_length: int = 512
    num_samples: int = 1000
    device: str = "auto"
    
    # Efficiency benchmarking
    benchmark_memory: bool = True
    benchmark_speed: bool = True
    warmup_runs: int = 10
    benchmark_runs: int = 100
    
    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        
        if self.perplexity_datasets is None:
            self.perplexity_datasets = ['wikitext2', 'ptb']
        
        if self.downstream_tasks is None:
            self.downstream_tasks = ['hellaswag', 'truthfulqa', 'mmlu']
        
        if self.generation_prompts is None:
            self.generation_prompts = [
                "The capital of France is",
                "Explain quantum computing in simple terms:",
                "Write a short story about a robot:",
                "What are the benefits of renewable energy?",
                "How does photosynthesis work?"
            ]


class PerplexityEvaluator:
    """Evaluates model perplexity on various datasets"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        num_samples: int = 1000,
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Evaluate perplexity on a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            num_samples: Number of samples to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity metrics
        """
        logging.info(f"Evaluating perplexity on {dataset_name}")
        
        # Load dataset
        if dataset_name == 'wikitext2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            texts = [example['text'] for example in dataset if len(example['text'].strip()) > 50]
        elif dataset_name == 'ptb':
            dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
            texts = [example['sentence'] for example in dataset]
        elif dataset_name == 'lambada':
            dataset = load_dataset('lambada', split='test')
            texts = [example['text'] for example in dataset]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit number of samples
        texts = texts[:num_samples]
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc=f"Evaluating {dataset_name}"):
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                input_ids = encoding['input_ids']
                attention_mask = encoding.get('attention_mask', None)
                
                # Skip very short sequences
                if input_ids.size(1) < 10:
                    continue
                
                # Forward pass
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss
                    
                    # Count actual tokens (excluding padding)
                    if attention_mask is not None:
                        num_tokens = attention_mask.sum().item()
                    else:
                        num_tokens = input_ids.size(1)
                    
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    
                except Exception as e:
                    logging.warning(f"Error processing text: {e}")
                    continue
        
        # Calculate perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        else:
            perplexity = float('inf')
        
        results = {
            'dataset': dataset_name,
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_samples': len(texts)
        }
        
        logging.info(f"{dataset_name} results: PPL={perplexity:.2f}, Loss={avg_loss:.4f}")
        return results


class DownstreamTaskEvaluator:
    """Evaluates model on downstream tasks using lm-evaluation-harness"""
    
    def __init__(self, model_path: str, device='cpu'):
        self.model_path = model_path
        self.device = device
    
    def evaluate_tasks(
        self,
        tasks: List[str],
        num_fewshot: int = 0,
        batch_size: int = 1
    ) -> Dict[str, Dict]:
        """
        Evaluate model on downstream tasks
        
        Args:
            tasks: List of task names
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with task results
        """
        if not LM_EVAL_AVAILABLE:
            logging.warning("lm-eval not available. Skipping downstream task evaluation.")
            return {}
        
        logging.info(f"Evaluating downstream tasks: {tasks}")
        
        try:
            # Initialize model for lm-eval
            lm = HFLM(pretrained=self.model_path, device=self.device)
            
            # Run evaluation
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=self.device
            )
            
            return results
            
        except Exception as e:
            logging.error(f"Error in downstream evaluation: {e}")
            return {}


class GenerationEvaluator:
    """Evaluates text generation quality and diversity"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_generation(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Union[List, float]]:
        """
        Evaluate generation quality and diversity
        
        Args:
            prompts: List of prompts to generate from
            max_new_tokens: Maximum tokens to generate
            num_return_sequences: Number of sequences per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with generation metrics
        """
        logging.info("Evaluating text generation")
        
        all_generations = []
        generation_times = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating text"):
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                # Generate
                start_time = time.time()
                
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                # Decode generations
                prompt_length = inputs['input_ids'].size(1)
                generated_texts = []
                
                for output in outputs:
                    generated_tokens = output[prompt_length:]
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    generated_texts.append(generated_text)
                
                all_generations.extend(generated_texts)
        
        # Calculate metrics
        avg_generation_time = np.mean(generation_times)
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in all_generations)
        tokens_per_second = total_tokens / sum(generation_times) if sum(generation_times) > 0 else 0
        
        # Calculate diversity (unique n-grams)
        def calculate_diversity(texts, n=2):
            all_ngrams = set()
            total_ngrams = 0
            
            for text in texts:
                tokens = self.tokenizer.encode(text)
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    all_ngrams.add(ngram)
                    total_ngrams += 1
            
            return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0
        
        diversity_2gram = calculate_diversity(all_generations, 2)
        diversity_3gram = calculate_diversity(all_generations, 3)
        
        results = {
            'prompts': prompts,
            'generations': all_generations,
            'avg_generation_time': avg_generation_time,
            'tokens_per_second': tokens_per_second,
            'diversity_2gram': diversity_2gram,
            'diversity_3gram': diversity_3gram,
            'total_generated_tokens': total_tokens
        }
        
        logging.info(f"Generation results: {tokens_per_second:.2f} tokens/sec, "
                    f"diversity-2: {diversity_2gram:.3f}, diversity-3: {diversity_3gram:.3f}")
        
        return results


class EfficiencyBenchmark:
    """Benchmarks model efficiency (speed, memory, etc.)"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def benchmark_inference_speed(
        self,
        input_lengths: List[int] = [10, 50, 100, 200, 500],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Dict]:
        """
        Benchmark inference speed across different input lengths
        
        Args:
            input_lengths: List of input sequence lengths to test
            num_runs: Number of benchmark runs per length
            warmup_runs: Number of warmup runs
            
        Returns:
            Speed benchmark results
        """
        logging.info("Benchmarking inference speed")
        
        results = {}
        
        for length in input_lengths:
            logging.info(f"Benchmarking length {length}")
            
            # Create test input
            input_ids = torch.randint(
                0, self.tokenizer.vocab_size, 
                (1, length), 
                device=self.device
            )
            attention_mask = torch.ones_like(input_ids)
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Benchmark runs
            inference_times = []
            
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
            
            # Calculate statistics
            results[f"length_{length}"] = {
                'input_length': length,
                'avg_time': np.mean(inference_times),
                'std_time': np.std(inference_times),
                'min_time': np.min(inference_times),
                'max_time': np.max(inference_times),
                'p95_time': np.percentile(inference_times, 95),
                'p99_time': np.percentile(inference_times, 99),
                'tokens_per_second': length / np.mean(inference_times)
            }
        
        return results
    
    def benchmark_memory_usage(
        self,
        input_lengths: List[int] = [10, 50, 100, 200, 500]
    ) -> Dict[str, Dict]:
        """
        Benchmark memory usage across different input lengths
        
        Args:
            input_lengths: List of input sequence lengths to test
            
        Returns:
            Memory benchmark results
        """
        logging.info("Benchmarking memory usage")
        
        results = {}
        process = psutil.Process()
        
        for length in input_lengths:
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create test input
            input_ids = torch.randint(
                0, self.tokenizer.vocab_size,
                (1, length),
                device=self.device
            )
            attention_mask = torch.ones_like(input_ids)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Get model memory on GPU if applicable
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                gpu_memory = torch.cuda.memory_allocated(self.device) / 1024 / 1024  # MB
                gpu_peak = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024  # MB
            else:
                gpu_memory = 0
                gpu_peak = 0
            
            results[f"length_{length}"] = {
                'input_length': length,
                'cpu_memory_mb': memory_used,
                'gpu_memory_mb': gpu_memory,
                'gpu_peak_mb': gpu_peak,
                'memory_per_token': memory_used / length if length > 0 else 0
            }
            
            # Clean up
            del input_ids, attention_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory usage (assuming float32)
        param_memory_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory_mb,
            'model_size_gb': param_memory_mb / 1024
        }


class ComprehensiveBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.setup_logging()
        self.load_model()
        
        # Results storage
        self.results = {
            'config': config.__dict__,
            'model_info': {},
            'perplexity': {},
            'downstream': {},
            'generation': {},
            'efficiency': {}
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model from: {self.config.model_path}")
        
        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "cpu" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Get model info
        efficiency_benchmark = EfficiencyBenchmark(self.model, self.tokenizer, self.device)
        self.results['model_info'] = efficiency_benchmark.get_model_size()
        
        self.logger.info(f"Model loaded: {self.results['model_info']['total_parameters']:,} parameters")
    
    def run_perplexity_evaluation(self):
        """Run perplexity evaluation"""
        if not self.config.eval_perplexity:
            return
        
        self.logger.info("Starting perplexity evaluation")
        evaluator = PerplexityEvaluator(self.model, self.tokenizer, self.device)
        
        for dataset in self.config.perplexity_datasets:
            try:
                results = evaluator.evaluate_dataset(
                    dataset,
                    self.config.num_samples,
                    self.config.max_length
                )
                self.results['perplexity'][dataset] = results
            except Exception as e:
                self.logger.error(f"Error evaluating {dataset}: {e}")
    
    def run_downstream_evaluation(self):
        """Run downstream task evaluation"""
        if not self.config.eval_downstream:
            return
        
        self.logger.info("Starting downstream task evaluation")
        evaluator = DownstreamTaskEvaluator(self.config.model_path, self.device)
        
        try:
            results = evaluator.evaluate_tasks(
                self.config.downstream_tasks,
                batch_size=self.config.batch_size
            )
            self.results['downstream'] = results
        except Exception as e:
            self.logger.error(f"Error in downstream evaluation: {e}")
    
    def run_generation_evaluation(self):
        """Run text generation evaluation"""
        if not self.config.eval_generation:
            return
        
        self.logger.info("Starting generation evaluation")
        evaluator = GenerationEvaluator(self.model, self.tokenizer, self.device)
        
        try:
            results = evaluator.evaluate_generation(self.config.generation_prompts)
            self.results['generation'] = results
        except Exception as e:
            self.logger.error(f"Error in generation evaluation: {e}")
    
    def run_efficiency_evaluation(self):
        """Run efficiency benchmarks"""
        if not self.config.eval_efficiency:
            return
        
        self.logger.info("Starting efficiency evaluation")
        benchmark = EfficiencyBenchmark(self.model, self.tokenizer, self.device)
        
        try:
            if self.config.benchmark_speed:
                speed_results = benchmark.benchmark_inference_speed(
                    num_runs=self.config.benchmark_runs,
                    warmup_runs=self.config.warmup_runs
                )
                self.results['efficiency']['speed'] = speed_results
            
            if self.config.benchmark_memory:
                memory_results = benchmark.benchmark_memory_usage()
                self.results['efficiency']['memory'] = memory_results
                
        except Exception as e:
            self.logger.error(f"Error in efficiency evaluation: {e}")
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        self.generate_summary_report(output_dir)
        
        # Generate visualizations
        self.generate_visualizations(output_dir)
        
        self.logger.info(f"Benchmark report saved to: {output_dir}")
    
    def generate_summary_report(self, output_dir: Path):
        """Generate human-readable summary report"""
        
        summary = []
        summary.append("# Distilled LLaMA Benchmark Report\n")
        summary.append(f"Model: {self.config.model_path}\n")
        summary.append(f"Device: {self.device}\n")
        summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model info
        model_info = self.results['model_info']
        summary.append("## Model Information\n")
        summary.append(f"- Total Parameters: {model_info.get('total_parameters', 0):,}\n")
        summary.append(f"- Model Size: {model_info.get('model_size_gb', 0):.2f} GB\n")
        summary.append(f"- Parameter Memory: {model_info.get('parameter_memory_mb', 0):.1f} MB\n\n")
        
        # Perplexity results
        if self.results['perplexity']:
            summary.append("## Perplexity Results\n")
            for dataset, results in self.results['perplexity'].items():
                ppl = results.get('perplexity', float('inf'))
                summary.append(f"- {dataset}: {ppl:.2f}\n")
            summary.append("\n")
        
        # Downstream task results
        if self.results['downstream']:
            summary.append("## Downstream Task Results\n")
            for task, results in self.results['downstream'].get('results', {}).items():
                if isinstance(results, dict) and 'acc' in results:
                    acc = results['acc']
                    summary.append(f"- {task}: {acc:.3f}\n")
            summary.append("\n")
        
        # Generation results
        if self.results['generation']:
            gen_results = self.results['generation']
            summary.append("## Generation Results\n")
            summary.append(f"- Tokens/sec: {gen_results.get('tokens_per_second', 0):.2f}\n")
            summary.append(f"- Avg generation time: {gen_results.get('avg_generation_time', 0):.3f}s\n")
            summary.append(f"- Diversity (2-gram): {gen_results.get('diversity_2gram', 0):.3f}\n")
            summary.append(f"- Diversity (3-gram): {gen_results.get('diversity_3gram', 0):.3f}\n\n")
        
        # Efficiency results
        if self.results['efficiency']:
            summary.append("## Efficiency Results\n")
            
            if 'speed' in self.results['efficiency']:
                speed_results = self.results['efficiency']['speed']
                summary.append("### Speed Benchmarks\n")
                for length_key, metrics in speed_results.items():
                    length = metrics['input_length']
                    avg_time = metrics['avg_time']
                    tps = metrics['tokens_per_second']
                    summary.append(f"- Length {length}: {avg_time:.4f}s ({tps:.1f} tokens/s)\n")
                summary.append("\n")
            
            if 'memory' in self.results['efficiency']:
                memory_results = self.results['efficiency']['memory']
                summary.append("### Memory Usage\n")
                for length_key, metrics in memory_results.items():
                    length = metrics['input_length']
                    cpu_mem = metrics['cpu_memory_mb']
                    summary.append(f"- Length {length}: {cpu_mem:.1f} MB\n")
                summary.append("\n")
        
        # Save summary
        with open(output_dir / 'benchmark_summary.md', 'w') as f:
            f.writelines(summary)
    
    def generate_visualizations(self, output_dir: Path):
        """Generate benchmark visualizations"""
        
        try:
            # Speed vs sequence length plot
            if 'speed' in self.results.get('efficiency', {}):
                self.plot_speed_benchmarks(output_dir)
            
            # Memory usage plot
            if 'memory' in self.results.get('efficiency', {}):
                self.plot_memory_usage(output_dir)
            
            # Perplexity comparison
            if self.results.get('perplexity'):
                self.plot_perplexity_results(output_dir)
                
        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")
    
    def plot_speed_benchmarks(self, output_dir: Path):
        """Plot speed benchmark results"""
        speed_data = self.results['efficiency']['speed']
        
        lengths = [metrics['input_length'] for metrics in speed_data.values()]
        avg_times = [metrics['avg_time'] for metrics in speed_data.values()]
        tokens_per_sec = [metrics['tokens_per_second'] for metrics in speed_data.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average inference time
        ax1.plot(lengths, avg_times, 'bo-')
        ax1.set_xlabel('Input Length (tokens)')
        ax1.set_ylabel('Average Inference Time (s)')
        ax1.set_title('Inference Time vs Input Length')
        ax1.grid(True)
        
        # Tokens per second
        ax2.plot(lengths, tokens_per_sec, 'ro-')
        ax2.set_xlabel('Input Length (tokens)')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Throughput vs Input Length')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'speed_benchmarks.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_usage(self, output_dir: Path):
        """Plot memory usage results"""
        memory_data = self.results['efficiency']['memory']
        
        lengths = [metrics['input_length'] for metrics in memory_data.values()]
        cpu_memory = [metrics['cpu_memory_mb'] for metrics in memory_data.values()]
        
        plt.figure(figsize=(8, 6))
        plt.plot(lengths, cpu_memory, 'go-')
        plt.xlabel('Input Length (tokens)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Input Length')
        plt.grid(True)
        plt.savefig(output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_perplexity_results(self, output_dir: Path):
        """Plot perplexity comparison across datasets"""
        perplexity_data = self.results['perplexity']
        
        datasets = list(perplexity_data.keys())
        perplexities = [perplexity_data[dataset]['perplexity'] for dataset in datasets]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(datasets, perplexities)
        plt.xlabel('Dataset')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Results by Dataset')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, ppl in zip(bars, perplexities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{ppl:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'perplexity_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        self.logger.info("Starting comprehensive benchmark")
        
        start_time = time.time()
        
        # Run all evaluations
        self.run_perplexity_evaluation()
        self.run_downstream_evaluation()
        self.run_generation_evaluation()
        self.run_efficiency_evaluation()
        
        # Record total time
        total_time = time.time() - start_time
        self.results['benchmark_duration'] = total_time
        
        # Generate report
        self.generate_report()
        
        self.logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        
        return self.results


def main():
    """Example usage"""
    
    # Configuration
    config = BenchmarkConfig(
        model_path="./checkpoints/best_model",
        output_dir="./benchmark_results",
        num_samples=100,  # Reduced for example
        benchmark_runs=10,
        perplexity_datasets=['wikitext2'],
        downstream_tasks=['hellaswag'],
        eval_downstream=False  # Disable for this example
    )
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    print("Benchmark completed!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()