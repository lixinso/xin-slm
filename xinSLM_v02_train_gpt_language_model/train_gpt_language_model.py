# Minimal GPT-style language model inspired by nanoGPT
# References:
# - Vaswani et al., "Attention Is All You Need" (2017) [oai_citation:1‡arxiv.org](https://arxiv.org/abs/1706.03762)
# - OpenAI GPT-2 (Radford et al., 2019) [oai_citation:2‡openai.com](https://openai.com/research/language-unsupervised)
# - nanoGPT by Andrej Karpathy [oai_citation:3‡github.com](https://github.com/karpathy/nanoGPT)

import math
import time
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset

# ====== Configuration ======
BATCH_SIZE = 16
BLOCK_SIZE = 64   # context length
N_LAYER = 4       # number of transformer blocks
N_HEAD  = 4
N_EMBD  = 128
DROP    = 0.1
LR      = 3e-4
EPOCHS  = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Simple BPE Tokenizer ======
# This is a toy Byte-Pair Encoding implementation used for demonstration only.
@dataclass
class BPETokenizer:
    vocab: dict
    merges: dict

    @classmethod
    def train(cls, texts, vocab_size=1000, merges=1000):
        # Initialize vocabulary with characters
        vocab = {ch: idx for idx, ch in enumerate(sorted(set(''.join(texts))))}
        idx = len(vocab)
        merges_dict = {}
        for _ in range(merges):
            pairs = {}
            for text in texts:
                tokens = text.split()
                for token in tokens:
                    for i in range(len(token)-1):
                        pair = token[i:i+2]
                        pairs[pair] = pairs.get(pair, 0) + 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges_dict[best] = idx
            vocab[best] = idx
            idx += 1
        return cls(vocab, merges_dict)

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if i+1 < len(text) and text[i:i+2] in self.merges:
                tokens.append(self.merges[text[i:i+2]])
                i += 2
            else:
                tokens.append(self.vocab[text[i]])
                i += 1
        return tokens

# ====== Dataset ======
raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_texts = [item['text'] for item in raw_datasets['train'] if item['text']]
val_texts   = [item['text'] for item in raw_datasets['validation'] if item['text']]

bpe_tokenizer = BPETokenizer.train(train_texts, merges=2000)

# Numericalize
train_data = torch.tensor([tok for line in train_texts for tok in bpe_tokenizer.encode(line + '\n')], dtype=torch.long)
val_data   = torch.tensor([tok for line in val_texts for tok in bpe_tokenizer.encode(line + '\n')], dtype=torch.long)

def batchify(data):
    n_batch = len(data) // (BATCH_SIZE * BLOCK_SIZE)
    data = data[:n_batch * BATCH_SIZE * BLOCK_SIZE]
    return data.view(BATCH_SIZE, -1)

train_data = batchify(train_data)
val_data   = batchify(val_data)

# ====== Model ======
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

model = GPT(len(bpe_tokenizer.vocab)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ====== Training Loop ======
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
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        epoch_train_losses.append(current_loss)
        
        if i % (BLOCK_SIZE * 100) == 0:
            print(f"Epoch {epoch} | Step {i//BLOCK_SIZE} | Loss {current_loss:.4f}")
            metrics["train_losses"].append(current_loss)
            metrics["steps"].append(i//BLOCK_SIZE + epoch * (train_data.size(1) // BLOCK_SIZE))
            metrics["timestamps"].append(time.time() - start_time)
    
    # Validation step (quick)
    model.eval()
    with torch.no_grad():
        xb = val_data[:, :BLOCK_SIZE].to(DEVICE)
        yb = val_data[:, 1:BLOCK_SIZE+1].to(DEVICE)
        logits = model(xb)
        val_loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
    
    val_loss_value = val_loss.item()
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    
    print(f"Epoch {epoch} | Avg Train Loss {avg_train_loss:.4f} | Val Loss {val_loss_value:.4f}")
    
    metrics["val_losses"].append(val_loss_value)
    metrics["epochs"].append(epoch)

# Save metrics to JSON file
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to training_metrics.json")

# ====== Text Generation ======
def generate(prompt: str, max_new_tokens: int = 20):
    model.eval()
    tokens = bpe_tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, :]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = model(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, next_token[:, None]], dim=1)
    inv_vocab = {v: k for k, v in bpe_tokenizer.vocab.items()}
    return ''.join(inv_vocab[i] for i in idx[0].tolist())

print(generate("The meaning of life is"))
