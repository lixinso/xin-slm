# BookCorpus Training Guide for xinSLM GPT-OSS MoE

This guide explains how to train your GPT-OSS MoE model using the BookCorpus dataset for improved long-form text generation and narrative coherence.

## Overview

BookCorpus provides high-quality long-form text that helps language models learn:
- **Long-range dependencies** through book-length narratives
- **Narrative coherence** and story progression  
- **Literary vocabulary** and diverse writing styles
- **Context awareness** across extended sequences

## Prerequisites

### 1. System Requirements
- **Memory**: 16GB RAM (Mac Mini compatible)
- **Storage**: 2-3GB free space for dataset caching
- **Network**: Stable internet for dataset download

### 2. Required Files
Ensure you have these new files in your xinSLM_v06_gpt_oss directory:
```
xinSLM_v06_gpt_oss/
├── scripts/
│   ├── multi_dataset_loader.py          # Multi-dataset support
│   └── train_gpt_oss_moe.py             # Updated training script
├── configs/
│   └── bookcorpus_training_config.yaml  # BookCorpus configuration
└── BOOKCORPUS_TRAINING_GUIDE.md         # This guide
```

## Quick Start

### 1. Basic BookCorpus Training
```bash
cd xinSLM_v06_gpt_oss

# Train with BookCorpus + WikiText combination
python scripts/train_gpt_oss_moe.py \
  --config configs/bookcorpus_training_config.yaml
```

### 2. BookCorpus-Only Training
```bash
# Create a BookCorpus-only config
python scripts/train_gpt_oss_moe.py \
  --config configs/bookcorpus_training_config.yaml \
  --override training.train_datasets='[{"name": "bookcorpus", "weight": 1.0}]'
```

## Configuration Options

### Dataset Combinations

#### Option 1: Balanced Mix (Recommended)
```yaml
train_datasets:
  - name: "bookcorpus"
    max_samples: null
    weight: 0.6          # 60% BookCorpus
  - name: "wikitext-103"
    max_samples: 50000
    weight: 0.3          # 30% WikiText-103
  - name: "wikitext-2"
    max_samples: null
    weight: 0.1          # 10% WikiText-2
```

#### Option 2: BookCorpus Heavy
```yaml
train_datasets:
  - name: "bookcorpus"
    max_samples: null
    weight: 0.8          # 80% BookCorpus
  - name: "wikitext-103"
    max_samples: 30000
    weight: 0.2          # 20% WikiText-103
```

#### Option 3: BookCorpus Only
```yaml
train_datasets:
  - name: "bookcorpus"
    max_samples: null
    weight: 1.0          # 100% BookCorpus
```

### Memory-Safe Settings

For Mac Mini 16GB, use these memory-optimized settings:

```yaml
# Model variant
model_variant: "light"    # Use light variant (150M active params)

# Batch settings
per_device_train_batch_size: 1
gradient_accumulation_steps: 64
max_seq_length: 512       # Shorter sequences for memory

# Memory optimization
mac_optimizations:
  max_memory_allocation: "10GB"  # Conservative limit
  empty_cache_interval: 50       # Frequent cache clearing
```

## Training Commands

### Standard Training
```bash
# Full BookCorpus training with monitoring
python scripts/safe_train.py \
  --config configs/bookcorpus_training_config.yaml \
  --output_dir checkpoints_bookcorpus \
  --logging_dir logs_bookcorpus
```

### Memory-Safe Training
```bash
# Ultra-safe training for maximum stability
python scripts/safe_train.py \
  --config configs/bookcorpus_training_config.yaml \
  --override training.model_variant=micro \
  --override training.max_seq_length=256 \
  --override training.per_device_train_batch_size=1
```

### Resume Training
```bash
# Resume from checkpoint
python scripts/train_gpt_oss_moe.py \
  --config configs/bookcorpus_training_config.yaml \
  --resume_from_checkpoint checkpoints_bookcorpus/checkpoint-1000
```

## Advanced Configuration

### Custom Dataset Mixing
Create your own dataset combination:

```python
# In your config file
train_datasets:
  - name: "bookcorpus"
    max_samples: 100000    # Limit BookCorpus for faster training
    weight: 0.5
  - name: "openwebtext" 
    max_samples: 50000     # Add web text for diversity
    weight: 0.3
  - name: "wikitext-103"
    max_samples: 20000
    weight: 0.2
```

### BookCorpus-Specific Preprocessing

```yaml
data:
  bookcorpus_preprocessing:
    remove_short_texts: true
    min_words: 50           # Minimum 50 words per text chunk
    clean_encoding: true    # Fix encoding issues
    remove_headers: true    # Remove book metadata
    deduplicate: true       # Remove duplicate content
```

### Generation Evaluation

Enable book-style generation testing:

```yaml
evaluation:
  eval_generation: true
  generation_max_length: 256
  generation_prompts:
    - "Once upon a time in a distant land,"
    - "The old man walked slowly down the street,"
    - "She opened the mysterious letter and discovered"
    - "In the quiet village where nothing ever happened,"
```

## Monitoring and Troubleshooting

### Training Progress

Monitor training with these key metrics:
```bash
# Check training logs
tail -f logs_bookcorpus/training.log

# Monitor resource usage
tail -f logs_bookcorpus/resource_usage.json
```

### Expected Behavior

#### First 100 Steps
- **Loss**: Should start around 10-11, decrease to 9-10
- **Memory**: Stable 70-85% usage
- **Speed**: 1-2 steps per second

#### After 1000 Steps  
- **Loss**: Should reach 7-8 range
- **Generation**: Basic sentence structure
- **Perplexity**: Decreasing steadily

#### After 5000 Steps
- **Loss**: 5-7 range
- **Generation**: Coherent short paragraphs
- **Book-style text**: Improved narrative flow

### Common Issues

#### Out of Memory
```yaml
# Reduce these settings
training:
  max_seq_length: 256          # Reduce from 512
  gradient_accumulation_steps: 32  # Reduce from 64
  model_variant: "micro"       # Use smaller model
```

#### Slow Training
```yaml
# Optimize for speed
training:
  max_samples: 50000          # Limit dataset size
mac_optimizations:
  num_workers: 1              # Reduce workers
  prefetch_factor: 1          # Reduce prefetching
```

#### Dataset Loading Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/
rm -rf ./cache_bookcorpus/
```

## Model Testing

### Quick Generation Test
```python
# Test BookCorpus-trained model
python test_model.py \
  --checkpoint checkpoints_bookcorpus/best_model.pt \
  --prompt "Once upon a time in a magical forest," \
  --max_length 100 \
  --temperature 0.7
```

### Narrative Quality Assessment
```python
# Test story continuation ability
python test_model.py \
  --checkpoint checkpoints_bookcorpus/best_model.pt \
  --prompt "The detective examined the evidence carefully. The case had been puzzling him for weeks, but now" \
  --max_length 200 \
  --temperature 0.8
```

## Performance Expectations

### Training Time
- **Micro Model**: 2-3 hours for 1000 steps
- **Light Model**: 4-6 hours for 1000 steps  
- **Full Training**: 12-24 hours for good quality

### Memory Usage
- **Micro**: 4-6GB peak
- **Light**: 8-12GB peak
- **Standard**: 12-16GB peak (may exceed Mac Mini limit)

### Text Quality Milestones

#### 1000 Steps
- Basic sentence structure
- Some word coherence
- Random topic jumping

#### 5000 Steps
- Short paragraph coherence
- Better grammar
- Some narrative flow

#### 10000 Steps
- Multi-paragraph stories
- Character consistency
- Plot development

## Best Practices

### 1. Start Small
- Begin with `micro` model variant
- Use limited dataset (`max_samples: 10000`)
- Verify everything works before scaling up

### 2. Monitor Memory
- Watch memory usage closely in first 100 steps
- Adjust settings if approaching 90% usage
- Use `safe_train.py` for automatic monitoring

### 3. Save Frequently
- Set `save_steps: 250` for valuable checkpoints
- Keep multiple checkpoints (`save_total_limit: 5`)
- Test model quality at each checkpoint

### 4. Evaluate Regularly
- Run generation tests every 1000 steps
- Compare outputs to track improvement
- Save good examples for reference

## Next Steps

After successful BookCorpus training:

1. **Scale Up**: Try larger model variants (light → standard)
2. **Fine-tune**: Target specific genres or writing styles
3. **Combine Datasets**: Add OpenWebText, C4, or other text sources
4. **Publish**: Share your improved model on Hugging Face

## Support

For issues with BookCorpus training:
1. Check the training logs for specific error messages
2. Verify your system meets memory requirements  
3. Test with smaller configurations first
4. Review the multi-dataset loader code for debugging

**Happy training with BookCorpus!** 📚🚀