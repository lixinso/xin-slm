# xinSLM v06: GPT-OSS MoE Model for Mac Mini

A GPT-OSS inspired Mixture of Experts (MoE) language model optimized for Mac Mini with 16GB RAM. This implementation replicates the GPT-OSS architecture with MXFP4-style quantization and Apple Silicon optimizations.

## Features

🚀 **GPT-OSS Architecture**
- Mixture of Experts (MoE) with configurable expert counts
- Grouped-Query Attention (GQA) for memory efficiency  
- Token-choice routing with top-k expert selection
- Configurable reasoning effort levels (low/medium/high)

🍎 **Mac Mini Optimized**
- Metal Performance Shaders (MPS) support
- 4-bit MXFP4-style quantization for memory efficiency
- Gradient checkpointing and memory-efficient attention
- CPU fallback for unsupported operations

⚡ **Performance Features**
- ~600M active parameters (from ~3B total parameters)
- Fits in 16GB RAM with quantization
- Streaming generation support
- Interactive chat interface

## Quick Start

### Requirements

```bash
pip install torch torchvision torchaudio transformers datasets accelerate
pip install pyyaml tqdm wandb numpy scipy
```

### Model Creation and Testing

```python
from models.gpt_oss_moe import create_gpt_oss_moe

# Create model optimized for Mac Mini
model = create_gpt_oss_moe(
    vocab_size=50257,
    hidden_size=768,
    num_layers=20,
    num_heads=12,
    num_kv_heads=4,
    max_seq_len=2048,
    num_experts=32,
    num_experts_per_tok=2,
    reasoning_effort="medium",
    use_quantization=True
)

# Get model info
info = model.get_model_info()
print(f"Active parameters: {info['active_parameters']:,}")
print(f"Total parameters: {info['total_parameters']:,}")
print(f"Memory usage: {info['active_size_mb']:.1f} MB")
```

### Training

```bash
# Train with default configuration
python scripts/train_gpt_oss_moe.py --config configs/training_config.yaml

# Train with custom settings
python scripts/train_gpt_oss_moe.py \
    --config configs/training_config.yaml \
    --model-variant light \
    --reasoning-effort low
```

### Inference

```bash
# Interactive chat mode
python scripts/inference.py --config configs/model_config.yaml --interactive

# Single prompt
python scripts/inference.py \
    --config configs/model_config.yaml \
    --prompt "Explain quantum computing" \
    --stream

# Load trained model
python scripts/inference.py \
    --config configs/model_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

### Evaluation

```bash
# Run full benchmark suite
python evaluation/benchmark.py \
    --config configs/model_config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output evaluation_report.json

# Run specific benchmarks
python evaluation/benchmark.py \
    --config configs/model_config.yaml \
    --benchmark hellaswag
```

## Architecture Details

### Model Variants

| Variant | Hidden Size | Layers | Experts | Active Params | Memory |
|---------|-------------|--------|---------|---------------|--------|
| ultra_light | 512 | 12 | 16 | ~200M | ~2GB |
| light | 640 | 16 | 24 | ~400M | ~3GB |
| standard | 768 | 20 | 32 | ~600M | ~4GB |
| performance | 896 | 22 | 48 | ~800M | ~6GB |

### MoE Configuration

- **Experts per Layer**: 16-48 experts depending on variant
- **Active Experts**: Top-2 routing (configurable)
- **Router**: Learned linear routing with softmax
- **Load Balancing**: Auxiliary loss for expert utilization

### Quantization

- **Method**: MXFP4-style 4-bit quantization
- **Target**: MoE expert weights (90% of parameters)
- **Precision**: FP16 for attention, embeddings, and normalization
- **Compression**: 3-4x memory reduction

## Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  vocab_size: 50257
  hidden_size: 768
  num_hidden_layers: 20
  num_attention_heads: 12
  num_key_value_heads: 4
  num_experts: 32
  num_experts_per_tok: 2
  reasoning_effort: "medium"

quantization:
  enable_quantization: true
  bits: 4
  quantize_moe_weights: true
  quantize_attention: false

mac_optimizations:
  use_metal_performance_shaders: true
  gradient_checkpointing: true
  memory_efficient_attention: true
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  learning_rate: 3e-4
  num_train_epochs: 3
  max_seq_length: 1024

mac_optimizations:
  device: "mps"
  num_workers: 2
  empty_cache_interval: 100
```

## Memory Optimization Strategies

### 1. Gradient Accumulation
- Micro-batch size of 1 to minimize memory peaks
- Accumulate gradients over 32 steps for effective batch size of 32

### 2. Quantization
- 4-bit quantization of MoE expert weights
- Keep critical components (attention, embeddings) in FP16
- Group-wise quantization with 32-element groups

### 3. Attention Optimization
- Grouped-Query Attention reduces KV cache size
- Memory-efficient attention implementation
- Flash attention fallback when available

### 4. Model Architecture
- Deep-narrow design (more layers, smaller width)
- Sparse MoE reduces active parameter count
- Tied embeddings for input/output projections

## Performance Benchmarks

### Hardware: Mac Mini M2, 16GB RAM

| Metric | Value |
|--------|-------|
| Inference Speed | ~15-25 tokens/sec |
| Memory Usage | ~8GB peak |
| Training Speed | ~1.5 hours/epoch (WikiText-2) |
| Perplexity (WikiText-2) | ~25-35 (target) |

### Evaluation Results

| Benchmark | Score | Samples |
|-----------|-------|---------|
| HellaSwag | ~0.35 | 500 |
| LAMBADA | ~0.25 | 500 |
| Winogrande | ~0.55 | 300 |
| WikiText-2 PPL | ~30 | Full |

## Project Structure

```
xinSLM_v06_gpt_oss/
├── models/
│   ├── gpt_oss_moe.py          # Main model implementation
│   └── quantization.py         # MXFP4-style quantization
├── configs/
│   ├── model_config.yaml       # Model architecture config
│   ├── training_config.yaml    # Training configuration
│   └── quantization_config.yaml # Quantization settings
├── scripts/
│   ├── train_gpt_oss_moe.py   # Training script
│   ├── inference.py            # Inference and chat
│   └── test_model.py           # Test suite
├── evaluation/
│   └── benchmark.py            # Evaluation benchmarks
├── deployment/
├── checkpoints/                # Model checkpoints
├── logs/                      # Training logs
└── README.md
```

## Advanced Usage

### Custom Model Configuration

```python
from models.gpt_oss_moe import GPTOSSMoEConfig, GPTOSSForCausalLM

config = GPTOSSMoEConfig(
    vocab_size=50257,
    hidden_size=512,
    num_hidden_layers=16,
    num_experts=24,
    num_experts_per_tok=3,
    reasoning_effort="high",
    use_quantization=True
)

model = GPTOSSForCausalLM(config)
```

### Quantization Control

```python
from models.quantization import ModelQuantizer, QuantizationConfig

quant_config = QuantizationConfig(
    bits=4,
    group_size=64,
    quantize_moe_experts=True,
    quantize_attention=False,
    use_mps_kernels=True
)

quantizer = ModelQuantizer(quant_config)
quantized_model = quantizer.quantize_model(model)
```

### Reasoning Effort Control

```python
# Adjust reasoning effort during inference
inference_engine = GPTOSSInference(config_path, checkpoint_path)

# Low effort: 1 expert per token, faster inference
response_fast = inference_engine.generate(
    "Quick question:", 
    reasoning_effort="low"
)

# High effort: 4 experts per token, better quality
response_detailed = inference_engine.generate(
    "Complex reasoning task:", 
    reasoning_effort="high"
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `per_device_train_batch_size` to 1
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing
   - Use a smaller model variant

2. **MPS Not Available**
   - Ensure macOS 12.3+ with Apple Silicon
   - Install PyTorch with MPS support
   - Falls back to CPU automatically

3. **Slow Training**
   - Enable gradient checkpointing
   - Use mixed precision (FP16)
   - Reduce sequence length
   - Use fewer experts

### Memory Monitoring

```python
# Monitor MPS memory usage
if torch.backends.mps.is_available():
    allocated = torch.mps.current_allocated_memory() / 1024**2
    print(f"MPS Memory: {allocated:.1f} MB")
    
    # Clear cache if needed
    torch.mps.empty_cache()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python scripts/test_model.py`
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- OpenAI for the GPT-OSS architecture inspiration
- Hugging Face for the transformers library
- Apple for Metal Performance Shaders optimization
- The open-source community for MoE research

## Citation

```bibtex
@misc{xinslm-v06-gpt-oss,
  title={xinSLM v06: GPT-OSS MoE Model for Mac Mini},
  author={xinSLM Team},
  year={2025},
  url={https://github.com/xinsongli/xin-slm}
}
```