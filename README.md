# 🧠 XinSLM: Small Language Models That Pack a Punch

> *Building efficient, locally-trainable language models from LSTM to state-of-the-art Transformer architectures*

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Ready-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/documentation/apple-silicon)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**XinSLM** is a research project exploring the evolution of Small Language Models - from classical LSTM architectures to modern Transformer designs inspired by Llama 3.2. All models are designed to train and run efficiently on local hardware, including Apple Silicon (Mac Mini M4).

## 🚀 What Makes This Project Special?

This isn't just another language model implementation. It's a **learning journey** that demonstrates:

- **🔄 Evolution of architectures**: From RNNs to cutting-edge Transformers
- **⚡ Local-first approach**: Train powerful models on your own hardware
- **🎯 Production-ready code**: Clean, well-documented implementations
- **📊 Comprehensive benchmarking**: Detailed metrics and evaluation
- **🔬 Research-oriented**: Knowledge distillation, attention mechanisms, and more

---

## 🏗️ Architecture Journey

### v01: LSTM Foundation 🔄
*Classical sequence modeling with modern training techniques*

```
📁 xinSLM_v01_train_lstm_language_model/
```

- **2-layer LSTM** with 100-dimensional embeddings
- **Weight tying** between embedding and output layers
- **WikiText-2 training** with truncated backpropagation
- **Gradient clipping** for stable training
- **Temperature-controlled generation**

**Key Innovation**: Demonstrates how classical RNN architectures can still be effective for language modeling with proper regularization and training techniques.

### v02: GPT-Style Transformer 🎯
*Minimal but powerful decoder-only architecture*

```
📁 xinSLM_v02_train_gpt_language_model/
```

- **nanoGPT-inspired** 4-layer transformer
- **Custom BPE tokenizer** with 2K merges
- **Multi-head self-attention** (4 heads, 128d embeddings)
- **Training metrics visualization** with loss curves
- **Both demo and production versions**

**Key Innovation**: Shows how a minimal transformer can achieve strong results with careful implementation and training.

### v03: Llama-Inspired SLM 🚀
*State-of-the-art architecture with modern optimizations*

```
📁 xinSLM_v03_slm_llama_architecture/
```

- **1.23B parameter** Llama 3.2-based architecture
- **Grouped-Query Attention (GQA)** for efficient inference
- **RoPE positional embeddings** supporting 128K context
- **Knowledge distillation** support
- **RMSNorm + SiLU activation**
- **Professional training pipeline** with checkpointing

**Key Innovation**: Production-ready implementation with all the bells and whistles of modern LLMs, optimized for local training.

---

## 📊 Performance & Benchmarks

### Training Performance
| Model | Parameters | Context Length | Training Time* | Memory Usage |
|-------|------------|----------------|----------------|--------------|
| LSTM v01 | ~8M | 30 tokens | 10 min | 2GB |
| GPT v02 | ~1.2M | 64 tokens | 15 min | 4GB |
| SLM v03 | 1.23B | 128K tokens | 8 hours | 16GB |

*On Mac Mini M4 with 32GB RAM

### Supported Benchmarks
Our models can be evaluated against standard metrics:
- **MMLU Pro** - Multitask language understanding
- **HellaSwag** - Commonsense reasoning
- **MATH-500** - Mathematical problem solving
- **LiveCodeBench** - Coding capabilities
- **Multilingual MMLU** - Cross-lingual performance

## 🛠️ Quick Start

### Prerequisites
```bash
# Install PyTorch with MPS support (Mac) or CUDA (PC)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

### Train Your First Model

**🚀 Start Simple (LSTM)**
```bash
cd xinSLM_v01_train_lstm_language_model
python train_lstm_language_model.py
```

**⚡ Go Modern (GPT-style)**
```bash
cd xinSLM_v02_train_gpt_language_model
python train_gpt_language_model.py
```

**🔥 Full Power (Llama-inspired)**
```bash
cd xinSLM_v03_slm_llama_architecture
python train.py
```

### Generate Text
```python
# LSTM model
from train_lstm_language_model import generate_text
print(generate_text("Once upon a time", max_len=50, temperature=0.8))

# SLM model
from xinSLM_v03_slm_llama_architecture.example_usage import generate
model = load_model("checkpoints/best_model")
print(generate("The future of AI is", max_length=100))
```

## 🔬 Research Features

### Knowledge Distillation
Train smaller models to mimic larger ones:
```python
# Train student model with teacher guidance
trainer = SLMTrainer(
    model=student_model,
    teacher_model=teacher_model,
    distillation_alpha=0.7,
    temperature=3.0
)
```

### Advanced Attention Mechanisms
- **Grouped-Query Attention**: 4x reduction in KV cache memory
- **RoPE Scaling**: Support for contexts beyond training length
- **Flash Attention**: Memory-efficient attention computation

### Training Optimizations
- **Mixed precision training** for 2x speedup
- **Gradient accumulation** for large effective batch sizes
- **Cosine learning rate scheduling**
- **Automatic checkpoint management**

## 📈 Training Reports & Visualizations

Each model version includes comprehensive training analysis:
- **Loss curves** and convergence metrics
- **Model comparison** charts
- **Attention pattern** visualizations
- **Generated text samples** at different temperatures

Check out the interactive training reports in each version's folder!

## 🎯 Use Cases

### Educational
- **Learn modern NLP**: From basics to state-of-the-art
- **Research platform**: Experiment with new architectures
- **Benchmark testing**: Evaluate on standard datasets

### Production
- **Local deployment**: No API dependencies
- **Custom fine-tuning**: Adapt to your specific domain
- **Resource-constrained environments**: Run on edge devices

### Research
- **Architecture experiments**: Easy to modify and extend
- **Distillation studies**: Teacher-student learning dynamics
- **Efficiency research**: Memory and compute optimizations

## 🤝 Contributing

We welcome contributions! Areas of interest:
- **New architectures**: Implement latest research papers
- **Optimization techniques**: Better training strategies
- **Evaluation metrics**: More comprehensive benchmarks
- **Documentation**: Tutorials and examples

## 📚 References & Inspiration

- **Vaswani et al.** - "Attention Is All You Need" (Transformer)
- **Touvron et al.** - "Llama 2: Open Foundation and Fine-Tuned Chat Models"
- **Karpathy** - nanoGPT (Minimal GPT implementation)
- **Meta** - Llama 3.2 architecture specifications

## 📄 License

MIT License - feel free to use for research and commercial projects!

---

**Built with ❤️ for the ML community | Optimized for Apple Silicon & CUDA**

