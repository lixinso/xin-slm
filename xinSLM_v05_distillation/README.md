# xinSLM v04: Knowledge Distillation Model

## Overview

This directory implements a knowledge distillation framework for training efficient Small Language Models (SLMs) optimized for local deployment on resource-constrained devices like the Mac Mini (16GB RAM).

## Key Features

- **~1B Parameter Architecture**: LLaMA-style decoder with deep-narrow design
- **Knowledge Distillation**: Teacher-student training framework
- **Grouped-Query Attention (GQA)**: Efficient attention mechanism
- **4-bit Quantization**: Memory-optimized deployment
- **Mac Mini Optimized**: CPU/Apple Silicon inference support

## Architecture Design

### Model Configuration
- **Parameters**: ~1B (target)
- **Layers**: 24-30 (deep-narrow design)
- **Hidden Size**: 1024-1536
- **Attention Heads**: 16-32 with GQA
- **Context Length**: 2048 tokens
- **Vocabulary**: 32k tokens (LLaMA-compatible)

### Key Architectural Features
- Pre-normalization (LayerNorm before attention/MLP)
- SiLU/SwiGLU activation functions
- Grouped-Query Attention for efficiency
- Embedding weight sharing
- RoPE positional embeddings

## Directory Structure

```
xinSLM_v04_distillation/
├── models/                 # Model architecture implementations
├── data/                  # Training and evaluation datasets
├── scripts/               # Training and inference scripts
├── configs/               # Model and training configurations
├── checkpoints/           # Saved model checkpoints
├── evaluation/            # Evaluation and benchmarking tools
├── deployment/            # Quantization and deployment scripts
└── logs/                  # Training logs and metrics
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Model**:
   Edit `configs/model_config.yaml` for architecture settings

3. **Prepare Data**:
   Run `scripts/prepare_distillation_data.py` to setup training data

4. **Train Model**:
   ```bash
   python scripts/train_distillation.py --config configs/distillation_config.yaml
   ```

5. **Evaluate**:
   ```bash
   python scripts/evaluate_model.py --checkpoint checkpoints/best_model
   ```

6. **Deploy**:
   ```bash
   python scripts/quantize_model.py --model checkpoints/best_model --bits 4
   ```

## Knowledge Distillation Framework

### Teacher Models
- LLaMA-2 7B/13B (instruction-tuned)
- DeepSeek models
- Other open-source LLMs

### Training Strategy
- **Loss Function**: α * CE_loss + β * KL_divergence
- **Temperature**: Softened teacher logits (T=2.0)
- **Data**: Instruction datasets + general text corpus
- **Optimization**: Mixed precision training with gradient checkpointing

### Advanced Techniques
- Sequence-level distillation
- On-policy distillation with teacher-generated data
- LoRA fine-tuning for teacher adaptation

## Quantization & Deployment

### Quantization Methods
- **GPTQ**: 4-bit post-training quantization
- **Group-wise Quantization**: 32-weight groups with individual scales
- **Mixed Precision**: 4-bit weights, 8-bit activations

### Deployment Targets
- **llama.cpp**: GGUF format for CPU inference
- **CoreML**: Apple Neural Engine optimization
- **Hugging Face**: 4-bit transformers integration

### Memory Requirements
- **FP16**: ~2GB
- **4-bit**: ~0.5-1GB
- **Target**: <1GB total memory footprint

## Evaluation Framework

### Benchmarks
- HellaSwag (commonsense reasoning)
- TruthfulQA (truthfulness)
- MMLU (knowledge)
- HumanEval (coding)
- Custom instruction-following tasks

### Metrics
- Perplexity on validation sets
- Task-specific accuracy scores
- Inference speed (tokens/second)
- Memory usage profiling

## Implementation Status

- [ ] Model architecture (models/)
- [ ] Distillation training framework (scripts/)
- [ ] Configuration system (configs/)
- [ ] Evaluation pipeline (evaluation/)
- [ ] Quantization tools (deployment/)
- [ ] Mac Mini deployment (deployment/)

## Research References

This implementation is based on recent advances in:
- Knowledge Distillation (Hinton et al., 2015)
- MobileLLM (deep-narrow architectures)
- LLaMA architecture design
- DeepSeek-R1 distillation methods
- SmolLM efficiency techniques

## License

Open-source implementation following Apache 2.0 license for maximum compatibility.