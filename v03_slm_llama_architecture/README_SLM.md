# Small Language Model (SLM) Implementation

A PyTorch implementation of a Small Language Model based on Llama 3.2 1B architecture, featuring Grouped-Query Attention, RoPE positional encoding, and knowledge distillation support.

## Architecture Overview

This implementation follows the Llama 3.2 1B specifications:

- **Model Size**: 1.23 billion parameters
- **Layers**: 16 decoder layers
- **Hidden Size**: 2048
- **Attention Heads**: 32 (with 8 key-value heads for GQA)
- **Feed-Forward**: 8192 intermediate size
- **Vocabulary**: ~128K tokens (expandable)
- **Context Length**: Up to 128K tokens (with RoPE scaling)
- **Architecture**: Transformer decoder with RMSNorm and SiLU activation

## Key Features

### 1. Grouped-Query Attention (GQA)
- Reduces memory usage by sharing key-value projections across query groups
- 32 attention heads grouped into 8 key-value heads (4:1 ratio)
- Improves inference speed while maintaining model quality

### 2. Rotary Positional Embeddings (RoPE)
- Supports context lengths up to 128K tokens
- Implements RoPE scaling for long contexts (RoPE-2θ)
- Dynamic scaling options for extrapolation beyond training length

### 3. Knowledge Distillation
- Support for training with teacher model guidance
- Combines standard cross-entropy loss with KL divergence from teacher
- Configurable temperature and alpha parameters

### 4. Modern Training Features
- AdamW optimizer with cosine learning rate scheduling
- Gradient clipping and mixed precision training
- Comprehensive logging and checkpointing
- Evaluation metrics including perplexity

## File Structure

```
xin-slm/
├── slm_config.py      # Model configuration class
├── model.py           # Main model implementation
├── rope.py            # Rotary position embedding implementation
├── tokenizer.py       # Tokenizer with large vocabulary support
├── train.py           # Training script with distillation
├── requirements.txt   # Package dependencies
└── README_SLM.md     # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For development:
```bash
pip install -e .
```

## Quick Start

### Basic Training

```python
from slm_config import SLMConfig
from model import SLMForCausalLM
from tokenizer import SLMTokenizer
from train import SLMTrainer, TextDataset

# Create configuration
config = SLMConfig(
    vocab_size=128256,
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    learning_rate=1e-3,
    batch_size=4,
    sequence_length=2048
)

# Initialize model and tokenizer
tokenizer = SLMTokenizer(vocab_size=config.vocab_size)
model = SLMForCausalLM(config)

# Create dataset
train_texts = ["Your training texts here..."]
train_dataset = TextDataset(train_texts, tokenizer, max_length=config.sequence_length)

# Train
trainer = SLMTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    output_dir="./checkpoints"
)

trainer.train(num_epochs=10)
```

### Training with Knowledge Distillation

```python
# Load or create teacher model
teacher_model = SLMForCausalLM.from_pretrained("path/to/teacher")

# Configure distillation
config.use_distillation = True
config.distillation_alpha = 0.5
config.distillation_temperature = 3.0

# Create trainer with teacher
trainer = SLMTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    teacher_model=teacher_model,
    output_dir="./checkpoints"
)
```

### Inference

```python
import torch

# Load trained model
model = SLMForCausalLM.from_pretrained("./checkpoints/best_model")
tokenizer = SLMTokenizer.from_pretrained("./checkpoints/best_model")

# Generate text
prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([input_ids])

with torch.no_grad():
    output = model.generate(
        input_tensor,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(output[0])
print(generated_text)
```

## Configuration Options

### Model Architecture
- `hidden_size`: Hidden dimension (default: 2048)
- `num_hidden_layers`: Number of layers (default: 16)
- `num_attention_heads`: Number of attention heads (default: 32)
- `num_key_value_heads`: Number of key-value heads for GQA (default: 8)
- `intermediate_size`: FFN intermediate size (default: 8192)
- `vocab_size`: Vocabulary size (default: 128256)

### Training Parameters
- `learning_rate`: Initial learning rate (default: 1e-3)
- `batch_size`: Training batch size (default: 64)
- `sequence_length`: Maximum sequence length (default: 8192)
- `gradient_clip_norm`: Gradient clipping norm (default: 1.0)
- `warmup_steps`: Learning rate warmup steps (default: 2000)

### Distillation Settings
- `use_distillation`: Enable knowledge distillation (default: False)
- `distillation_alpha`: Balance between student and distillation loss (default: 0.5)
- `distillation_temperature`: Temperature for soft targets (default: 3.0)

## Training Data Format

The trainer expects text data as a list of strings:

```python
train_texts = [
    "First document text...",
    "Second document text...",
    "Third document text..."
]
```

For long documents, the `TextDataset` class automatically creates sliding windows with configurable stride.

## Memory Requirements

### Training
- **Full Precision (FP32)**: ~8-10 GB GPU memory for batch size 1
- **Mixed Precision (FP16)**: ~4-6 GB GPU memory for batch size 1
- **Gradient Checkpointing**: Additional 20-30% memory savings

### Inference
- **Full Precision**: ~2.3 GB model weights + context memory
- **Quantized (4-bit)**: ~1.1 GB model weights + context memory

### Scaling Context Length
Memory usage scales linearly with context length due to attention computation:
- 8K context: Base memory usage
- 32K context: ~4× memory increase
- 128K context: ~16× memory increase

## Performance Optimizations

### Grouped-Query Attention
- Reduces KV cache memory by 4× (32 heads → 8 KV heads)
- Maintains model quality with minimal performance impact
- Essential for long context inference

### RoPE Scaling
- Enables training on short contexts (8K) then extending to long contexts (128K)
- Uses frequency scaling to handle position extrapolation
- Supports both linear and dynamic scaling strategies

### Knowledge Distillation
- Transfers knowledge from larger teacher models
- Achieves better performance than training from scratch
- Configurable loss weighting and temperature

## Monitoring and Logging

The trainer provides comprehensive logging:

- **Training Metrics**: Loss, learning rate, gradient norms
- **Evaluation Metrics**: Validation loss, perplexity
- **Checkpointing**: Regular saves with best model selection
- **Integration**: Supports WandB and TensorBoard logging

## Extending the Implementation

### Custom Tokenizers
Implement the tokenizer interface:
```python
class CustomTokenizer:
    def encode(self, text: str) -> List[int]: ...
    def decode(self, token_ids: List[int]) -> str: ...
    def __len__(self) -> int: ...
```

### Custom Loss Functions
Extend the distillation loss:
```python
class CustomDistillationLoss(DistillationLoss):
    def forward(self, student_logits, teacher_logits, labels):
        # Your custom loss implementation
        pass
```

### Long Context Training
For contexts beyond 128K:
1. Adjust `max_position_embeddings` in config
2. Use appropriate RoPE scaling factors
3. Consider sequence parallelism for very long contexts

## Benchmarking

The model can be evaluated on standard benchmarks:
- **Language Modeling**: WikiText-2, The Pile
- **Code Generation**: HumanEval, MBPP
- **Reasoning**: GSM8K, MATH
- **Multilingual**: FLORES, WMT datasets

See `BENCHMARK.md` for comprehensive evaluation metrics and benchmark descriptions.

## References

- [Llama 3.2 Technical Report](https://ai.meta.com/research/publications/)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)

## Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing style and structure
2. New features include appropriate tests
3. Documentation is updated for API changes
4. Performance implications are considered

## License

This implementation is for research and educational purposes. Please respect the licensing terms of any pre-trained models or datasets used.