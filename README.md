# LSTM Language Model

This repository provides a minimal PyTorch implementation of a two-layer LSTM language model with 100-dimensional embeddings and hidden states, trained on the WikiText-2 dataset using truncated backpropagation through time. The model applies dropout regularization between layers and ties the input embedding and output projection weights for parameter efficiency. Training is performed with an Adam optimizer and gradient clipping to ensure stable convergence. An autoregressive sampling routine generates text by iteratively feeding back predicted tokens, with optional temperature scaling for diversity.

## Architecture

- **Embedding Layer:** Maps token indices to 100-dimensional vectors, initialized uniformly to break symmetry.
- **LSTM Layers:** Two stacked `nn.LSTM` layers capture sequence dependencies with a dropout of 0.5 between them.
- **Dropout:** Applied after embeddings and LSTM outputs to mitigate overfitting.
- **Decoder (Linear Layer):** Projects the final hidden state at each time step back to the vocabulary size to produce logits for next-token prediction.
- **Weight Tying:** Shares the embedding and decoder weight matrices to reduce total parameters and improve generalization.
- **Hidden State Initialization:** Zero-initialization at the start of training or inference, with detachment between BPTT segments to prevent gradient explosion.

## Training Loop

1. **Data Batching:** The concatenated token stream is divided into batches of size 20 and split into sequence chunks of length 30 for truncated BPTT.
2. **Forward Pass:** Each chunk is fed through the embedding, LSTM, and decoder to compute logits.
3. **Loss Computation:** Uses `nn.CrossEntropyLoss` between logits and next-token targets.
4. **Backward Pass & Clipping:** Gradients are backpropagated, then clipped to a norm of 0.25 to stabilize training.
5. **Optimization:** Parameters updated via the Adam optimizer (learning rate 1e-3) for fast convergence.

## Inference

The `generate_text` function takes a prompt string, encodes it, and iteratively samples the next token by applying softmax with an adjustable temperature. The newly sampled token is appended to the input sequence, and the process repeats until an end-of-sequence token or maximum length is reached.

## Usage

### 1. Installation

```bash
pip install torch torchtext
```

Follow general PyTorch project structuring best practices in your environment setup.

### 2. Training

```bash
python train_lstm_language_model.py
```

### 3. Text Generation

After training, call:

```python
from train_lstm_language_model import generate_text
print(generate_text("Once upon a time", max_len=30, temperature=0.8))
```

### 4. Project Structure

- `train_lstm_language_model.py`: Main script
- `README.md`: Project overview and instructions

---

Inspired by and adapted from multiple PyTorch language model tutorials and repositories, including simple word-level and character-level LSTM implementations, and the Deep-Learning Project Template for minimal boilerplate.

## Run with Docker

```bash
docker build -t lstm-lm .
docker run --rm lstm-lm
```