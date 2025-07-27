#!/usr/bin/env python3
"""
Demo version of GPT training with synthetic data for quick testing.
This creates metrics without needing to download large datasets.
"""

import math
import time
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

# ====== Configuration ======
BATCH_SIZE = 8
BLOCK_SIZE = 32   # context length
N_LAYER = 2       # number of transformer blocks
N_HEAD  = 2
N_EMBD  = 64
DROP    = 0.1
LR      = 3e-4
EPOCHS  = 3
DEVICE = 'cpu'  # Force CPU for demo
VOCAB_SIZE = 100  # Small vocabulary

# ====== Synthetic Dataset ======
# Create synthetic training data
def create_synthetic_data(vocab_size, seq_length, num_sequences):
    """Create synthetic sequential data for language modeling."""
    np.random.seed(42)  # For reproducibility
    data = []
    for _ in range(num_sequences):
        # Create sequences with some patterns (e.g., arithmetic sequences)
        start = np.random.randint(0, vocab_size // 2)
        seq = [(start + i) % vocab_size for i in range(seq_length)]
        data.extend(seq)
    return torch.tensor(data, dtype=torch.long)

# Generate synthetic training and validation data
train_data = create_synthetic_data(VOCAB_SIZE, BLOCK_SIZE * 2, 500)
val_data = create_synthetic_data(VOCAB_SIZE, BLOCK_SIZE * 2, 100)

def batchify(data):
    n_batch = len(data) // (BATCH_SIZE * BLOCK_SIZE)
    data = data[:n_batch * BATCH_SIZE * BLOCK_SIZE]
    return data.view(BATCH_SIZE, -1)

train_data = batchify(train_data)
val_data = batchify(val_data)

print(f"Train data shape: {train_data.shape}")
print(f"Val data shape: {val_data.shape}")

# ====== Model (simplified version) ======
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROP)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROP),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Parameter(torch.zeros(1, BLOCK_SIZE, N_EMBD))
        self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        pos = self.pos_emb[:, :T, :]
        x = self.tok_emb(idx) + pos
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

model = GPT(VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ====== Training Loop with Metrics ======
metrics = {
    "train_losses": [],
    "val_losses": [],
    "epochs": [],
    "steps": [],
    "timestamps": []
}

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_train_losses = []
    
    for i in range(0, train_data.size(1) - BLOCK_SIZE, BLOCK_SIZE):
        xb = train_data[:, i:i+BLOCK_SIZE].to(DEVICE)
        yb = train_data[:, i+1:i+1+BLOCK_SIZE].to(DEVICE)
        logits = model(xb)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        epoch_train_losses.append(current_loss)
        
        # Log every 5 steps for demo
        if i % (BLOCK_SIZE * 5) == 0:
            print(f"Epoch {epoch} | Step {i//BLOCK_SIZE} | Loss {current_loss:.4f}")
            metrics["train_losses"].append(current_loss)
            metrics["steps"].append(i//BLOCK_SIZE + epoch * (train_data.size(1) // BLOCK_SIZE))
            metrics["timestamps"].append(time.time() - start_time)
    
    # Validation step
    model.eval()
    with torch.no_grad():
        xb = val_data[:, :BLOCK_SIZE].to(DEVICE)
        yb = val_data[:, 1:BLOCK_SIZE+1].to(DEVICE)
        logits = model(xb)
        val_loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
    
    val_loss_value = val_loss.item()
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    
    print(f"Epoch {epoch} | Avg Train Loss {avg_train_loss:.4f} | Val Loss {val_loss_value:.4f}")
    
    metrics["val_losses"].append(val_loss_value)
    metrics["epochs"].append(epoch)

# Save metrics to JSON file
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
print(f"Metrics saved to training_metrics.json")

# ====== Simple Text Generation Demo ======
def generate_demo():
    model.eval()
    # Start with a random token
    idx = torch.tensor([[np.random.randint(0, VOCAB_SIZE)]], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        for _ in range(10):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = model(idx_cond)
            # Use sampling instead of argmax for more interesting generation
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
    
    generated_sequence = idx[0].tolist()
    print(f"\nGenerated sequence: {generated_sequence}")

generate_demo()