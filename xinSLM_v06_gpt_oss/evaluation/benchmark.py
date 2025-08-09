"""
Comprehensive Evaluation and Benchmarking for GPT-OSS MoE Model
Optimized for Mac Mini with memory-efficient evaluation protocols

Features:
- Standard NLP benchmarks (HellaSwag, LAMBADA, etc.)
- Perplexity evaluation on various datasets
- Memory usage and performance benchmarking
- Reasoning capability assessment
- Model comparison utilities
- Automated reporting
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import warnings

# Standard evaluation datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.gpt_oss_moe import GPTOSSForCausalLM, create_gpt_oss_moe
from scripts.inference import GPTOSSInference


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    score: float
    samples: int
    metrics: Dict[str, Any]
    execution_time: float
    memory_usage: float


class PerplexityEvaluator:
    """Evaluate model perplexity on various datasets"""
    
    def __init__(self, model, tokenizer, device, max_length: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
    
    def evaluate_dataset(self, dataset_name: str, split: str = "validation") -> Dict[str, float]:
        """Evaluate perplexity on a dataset"""
        self.logger.info(f"Evaluating perplexity on {dataset_name} ({split})")
        
        # Load dataset
        if dataset_name == "wikitext-2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif dataset_name == "wikitext-103":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        elif dataset_name == "ptb":
            dataset = load_dataset("ptb_text_only", split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        
        # Process dataset in chunks
        for i in tqdm(range(0, min(len(dataset), 1000), 10), desc=f"Evaluating {dataset_name}"):
            batch_texts = []
            for j in range(i, min(i + 10, len(dataset))):
                text = dataset[j]["text"] if "text" in dataset[j] else str(dataset[j])
                if text and len(text.strip()) > 0:
                    batch_texts.append(text.strip())
            
            if not batch_texts:
                continue
            
            # Tokenize batch
            batch_encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = batch_encodings.input_ids.to(self.device)
            attention_mask = batch_encodings.attention_mask.to(self.device)
            
            # Evaluate
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs[0]
                
                # Count valid tokens (exclude padding)
                valid_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
                num_batches += 1
            
            # Memory cleanup
            if self.device.type == "mps":
                torch.mps.empty_cache()
        
        if total_tokens == 0:
            return {"perplexity": float("inf"), "loss": float("inf"), "tokens": 0}
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "loss": avg_loss,
            "tokens": total_tokens,
            "batches": num_batches
        }


class HellaSwagEvaluator:
    """Evaluate on HellaSwag commonsense reasoning benchmark"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate on HellaSwag dataset"""
        self.logger.info(f"Evaluating on HellaSwag (max {num_samples} samples)")
        
        # Load dataset
        dataset = load_dataset("hellaswag", split="validation")
        
        correct = 0
        total = 0
        
        for i, example in enumerate(tqdm(dataset.select(range(min(len(dataset), num_samples))), desc="HellaSwag")):
            # Format the context and choices
            context = example["ctx_a"] + " " + example["ctx_b"]
            choices = example["endings"]
            correct_answer = int(example["label"])
            
            # Calculate likelihood for each choice
            choice_probs = []
            for choice in choices:
                full_text = context + " " + choice
                prob = self._calculate_likelihood(full_text, len(context) + 1)
                choice_probs.append(prob)
            
            # Predict answer
            predicted_answer = np.argmax(choice_probs)
            
            if predicted_answer == correct_answer:
                correct += 1
            total += 1
            
            # Memory cleanup
            if self.device.type == "mps" and i % 50 == 0:
                torch.mps.empty_cache()
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _calculate_likelihood(self, text: str, start_pos: int) -> float:
        """Calculate likelihood of text completion"""
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoding.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
            
            # Calculate log likelihood of completion
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Focus on the completion part
            completion_start = min(start_pos, shift_labels.size(-1))
            completion_logits = shift_logits[:, completion_start:]
            completion_labels = shift_labels[:, completion_start:]
            
            if completion_labels.size(-1) == 0:
                return 0.0
            
            # Calculate cross entropy
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(completion_logits.view(-1, completion_logits.size(-1)), 
                            completion_labels.view(-1))
            
            # Average negative log likelihood
            avg_loss = losses.mean().item()
            return -avg_loss


class LAMBADAEvaluator:
    """Evaluate on LAMBADA reading comprehension benchmark"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate on LAMBADA dataset"""
        self.logger.info(f"Evaluating on LAMBADA (max {num_samples} samples)")
        
        # Load dataset
        dataset = load_dataset("lambada", split="validation")
        
        correct = 0
        total = 0
        
        for i, example in enumerate(tqdm(dataset.select(range(min(len(dataset), num_samples))), desc="LAMBADA")):
            text = example["text"]
            
            # Split into context and target word
            words = text.split()
            if len(words) < 2:
                continue
                
            context = " ".join(words[:-1])
            target_word = words[-1]
            
            # Generate prediction
            predicted_word = self._predict_next_word(context)
            
            # Check if prediction matches (case insensitive)
            if predicted_word.lower().strip() == target_word.lower().strip():
                correct += 1
            total += 1
            
            # Memory cleanup
            if self.device.type == "mps" and i % 50 == 0:
                torch.mps.empty_cache()
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _predict_next_word(self, context: str) -> str:
        """Predict the next word given context"""
        encoding = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=500)
        input_ids = encoding.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Get predictions for the next token
            next_token_logits = logits[:, -1, :]
            predicted_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Decode the predicted token
            predicted_word = self.tokenizer.decode(predicted_token_id, skip_special_tokens=True)
            return predicted_word.strip()


class WinograndeEvaluator:
    """Evaluate on Winogrande commonsense reasoning benchmark"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, num_samples: int = 500) -> Dict[str, float]:
        """Evaluate on Winogrande dataset"""
        self.logger.info(f"Evaluating on Winogrande (max {num_samples} samples)")
        
        # Load dataset
        dataset = load_dataset("winogrande", "winogrande_l", split="validation")
        
        correct = 0
        total = 0
        
        for i, example in enumerate(tqdm(dataset.select(range(min(len(dataset), num_samples))), desc="Winogrande")):
            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]
            answer = example["answer"]  # "1" or "2"
            
            # Replace placeholder with each option
            sentence1 = sentence.replace("_", option1)
            sentence2 = sentence.replace("_", option2)
            
            # Calculate likelihood for each sentence
            prob1 = self._calculate_sentence_likelihood(sentence1)
            prob2 = self._calculate_sentence_likelihood(sentence2)
            
            # Predict answer
            predicted_answer = "1" if prob1 > prob2 else "2"
            
            if predicted_answer == answer:
                correct += 1
            total += 1
            
            # Memory cleanup
            if self.device.type == "mps" and i % 50 == 0:
                torch.mps.empty_cache()
        
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _calculate_sentence_likelihood(self, sentence: str) -> float:
        """Calculate likelihood of a complete sentence"""
        encoding = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
        input_ids = encoding.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs[0].item()
            return -loss  # Higher likelihood = lower loss


class PerformanceBenchmark:
    """Benchmark model performance and memory usage"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def benchmark_inference(self, sequence_lengths: List[int] = None, 
                          batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark inference performance"""
        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024]
        if batch_sizes is None:
            batch_sizes = [1]  # Mac Mini constraint
        
        self.logger.info("Running performance benchmark...")
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                self.logger.info(f"Benchmarking {key}")
                
                # Generate random input
                input_ids = torch.randint(0, self.tokenizer.vocab_size, (batch_size, seq_len)).to(self.device)
                attention_mask = torch.ones_like(input_ids)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Benchmark
                times = []
                memory_usage = []
                
                for _ in range(10):
                    # Memory before
                    if self.device.type == "mps":
                        torch.mps.synchronize()
                        memory_before = torch.mps.current_allocated_memory() if hasattr(torch, 'mps') else 0
                    else:
                        memory_before = 0
                    
                    # Timing
                    start_time = time.time()
                    
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    if self.device.type == "mps":
                        torch.mps.synchronize()
                    
                    end_time = time.time()
                    
                    # Memory after
                    if self.device.type == "mps":
                        memory_after = torch.mps.current_allocated_memory() if hasattr(torch, 'mps') else 0
                        memory_diff = (memory_after - memory_before) / 1024**2  # MB
                    else:
                        memory_diff = 0
                    
                    times.append(end_time - start_time)
                    memory_usage.append(memory_diff)
                    
                    # Cleanup
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_memory = np.mean(memory_usage)
                tokens_per_second = (batch_size * seq_len) / avg_time
                
                results[key] = {
                    "avg_time_ms": avg_time * 1000,
                    "std_time_ms": std_time * 1000,
                    "avg_memory_mb": avg_memory,
                    "tokens_per_second": tokens_per_second,
                    "batch_size": batch_size,
                    "sequence_length": seq_len
                }
        
        return results


class ModelEvaluator:
    """Main evaluation orchestrator"""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize model
        if checkpoint_path:
            self.inference_engine = GPTOSSInference(config_path, checkpoint_path)
            self.model = self.inference_engine.model
            self.tokenizer = self.inference_engine.tokenizer
            self.device = self.inference_engine.device
        else:
            # Create new model for testing
            self.setup_model_and_tokenizer()
        
        # Initialize evaluators
        self.perplexity_evaluator = PerplexityEvaluator(self.model, self.tokenizer, self.device)
        self.hellaswag_evaluator = HellaSwagEvaluator(self.model, self.tokenizer, self.device)
        self.lambada_evaluator = LAMBADAEvaluator(self.model, self.tokenizer, self.device)
        self.winogrande_evaluator = WinograndeEvaluator(self.model, self.tokenizer, self.device)
        self.performance_benchmark = PerformanceBenchmark(self.model, self.tokenizer, self.device)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for evaluation"""
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Create model
        model_variant = self.config.get('model', {}).get('model_variant', 'standard')
        variant_config = self.config.get('model_variants', {}).get(model_variant, {})
        
        self.model = create_gpt_oss_moe(
            vocab_size=50257,
            hidden_size=variant_config.get('hidden_size', 768),
            num_layers=variant_config.get('num_hidden_layers', 20),
            num_heads=variant_config.get('num_attention_heads', 12),
            num_kv_heads=variant_config.get('num_key_value_heads', 4),
            max_seq_len=2048,
            num_experts=variant_config.get('num_experts', 32),
            num_experts_per_tok=variant_config.get('num_experts_per_tok', 2),
            reasoning_effort=variant_config.get('reasoning_effort', 'medium'),
            use_quantization=False  # Disable for evaluation
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def run_full_evaluation(self) -> Dict[str, BenchmarkResult]:
        """Run complete evaluation suite"""
        self.logger.info("Starting full evaluation suite...")
        
        results = {}
        
        # 1. Perplexity evaluation
        self.logger.info("=" * 50)
        self.logger.info("PERPLEXITY EVALUATION")
        self.logger.info("=" * 50)
        
        for dataset in ["wikitext-2"]:  # Add more as needed
            start_time = time.time()
            memory_start = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
            
            ppl_results = self.perplexity_evaluator.evaluate_dataset(dataset)
            
            execution_time = time.time() - start_time
            memory_end = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
            
            results[f"perplexity_{dataset}"] = BenchmarkResult(
                name=f"Perplexity ({dataset})",
                score=ppl_results["perplexity"],
                samples=ppl_results["tokens"],
                metrics=ppl_results,
                execution_time=execution_time,
                memory_usage=memory_end - memory_start
            )
        
        # 2. HellaSwag evaluation
        self.logger.info("=" * 50)
        self.logger.info("HELLASWAG EVALUATION")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        memory_start = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
        
        hellaswag_results = self.hellaswag_evaluator.evaluate(num_samples=500)  # Reduced for Mac Mini
        
        execution_time = time.time() - start_time
        memory_end = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
        
        results["hellaswag"] = BenchmarkResult(
            name="HellaSwag",
            score=hellaswag_results["accuracy"],
            samples=hellaswag_results["total"],
            metrics=hellaswag_results,
            execution_time=execution_time,
            memory_usage=memory_end - memory_start
        )
        
        # 3. LAMBADA evaluation
        self.logger.info("=" * 50)
        self.logger.info("LAMBADA EVALUATION")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        memory_start = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
        
        lambada_results = self.lambada_evaluator.evaluate(num_samples=500)
        
        execution_time = time.time() - start_time
        memory_end = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
        
        results["lambada"] = BenchmarkResult(
            name="LAMBADA",
            score=lambada_results["accuracy"],
            samples=lambada_results["total"],
            metrics=lambada_results,
            execution_time=execution_time,
            memory_usage=memory_end - memory_start
        )
        
        # 4. Winogrande evaluation
        self.logger.info("=" * 50)
        self.logger.info("WINOGRANDE EVALUATION")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        memory_start = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
        
        winogrande_results = self.winogrande_evaluator.evaluate(num_samples=300)
        
        execution_time = time.time() - start_time
        memory_end = torch.mps.current_allocated_memory() / 1024**2 if self.device.type == "mps" else 0
        
        results["winogrande"] = BenchmarkResult(
            name="Winogrande",
            score=winogrande_results["accuracy"],
            samples=winogrande_results["total"],
            metrics=winogrande_results,
            execution_time=execution_time,
            memory_usage=memory_end - memory_start
        )
        
        # 5. Performance benchmark
        self.logger.info("=" * 50)
        self.logger.info("PERFORMANCE BENCHMARK")
        self.logger.info("=" * 50)
        
        perf_results = self.performance_benchmark.benchmark_inference()
        
        results["performance"] = BenchmarkResult(
            name="Performance",
            score=perf_results.get("batch_1_seq_512", {}).get("tokens_per_second", 0),
            samples=len(perf_results),
            metrics=perf_results,
            execution_time=0,
            memory_usage=0
        )
        
        return results
    
    def generate_report(self, results: Dict[str, BenchmarkResult], output_path: str = "evaluation_report.json"):
        """Generate evaluation report"""
        self.logger.info("Generating evaluation report...")
        
        # Create summary
        summary = {
            "model_info": self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(self.device),
            "results": {}
        }
        
        # Add results
        for key, result in results.items():
            summary["results"][key] = {
                "name": result.name,
                "score": result.score,
                "samples": result.samples,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage,
                "metrics": result.metrics
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to: {output_path}")
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: Dict[str, BenchmarkResult]):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        for key, result in results.items():
            print(f"\n{result.name}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Samples: {result.samples}")
            print(f"  Time: {result.execution_time:.2f}s")
            if result.memory_usage > 0:
                print(f"  Memory: {result.memory_usage:.1f} MB")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS MoE model")
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
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output path for evaluation report"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "perplexity", "hellaswag", "lambada", "winogrande", "performance"],
        default="all",
        help="Which benchmark to run"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.config, args.checkpoint)
        
        # Run evaluation
        if args.benchmark == "all":
            results = evaluator.run_full_evaluation()
        else:
            # Run specific benchmark
            results = {}
            if args.benchmark == "perplexity":
                ppl_results = evaluator.perplexity_evaluator.evaluate_dataset("wikitext-2")
                results["perplexity_wikitext-2"] = BenchmarkResult(
                    name="Perplexity (wikitext-2)",
                    score=ppl_results["perplexity"],
                    samples=ppl_results["tokens"],
                    metrics=ppl_results,
                    execution_time=0,
                    memory_usage=0
                )
            # Add other specific benchmarks as needed
        
        # Generate report
        evaluator.generate_report(results, args.output)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()