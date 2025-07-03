import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from datasets import load_dataset

# ====== Configuration ======
BATCH_SIZE = 20      # number of sequences per batch  [oai_citation:7‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
SEQ_LEN    = 30      # truncated BPTT length  [oai_citation:8‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
EMBED_SIZE = 100     # embedding & hidden dimension  [oai_citation:9‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
HIDDEN_SIZE= 100     # LSTM hidden size  [oai_citation:10‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
NUM_LAYERS = 2       # number of LSTM layers  [oai_citation:11‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
DROPOUT    = 0.5     # for regularization  [oai_citation:12‡github.com](https://github.com/Ebimsv/Torch-Linguist?utm_source=chatgpt.com)
CLIP_NORM  = 0.25    # gradient clipping threshold  [oai_citation:13‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com) [oai_citation:14‡github.com](https://github.com/Ebimsv/Torch-Linguist?utm_source=chatgpt.com)
EPOCHS     = 10
LR         = 1e-3    # learning rate

DEVICE = torch.device('mps' if torch.backends.mps.is_available() 
                      else 'cuda' if torch.cuda.is_available() 
                      else 'cpu')
print(f"Using device: {DEVICE}")  # supports Apple MPS  [oai_citation:15‡docs.pytorch.org](https://docs.pytorch.org/tutorials/?utm_source=chatgpt.com)

# ====== Data Preparation ======
tokenizer = get_tokenizer('basic_english')
raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

def yield_tokens(data_iter):
    for item in data_iter:
        if item['text']:
            yield tokenizer(item['text'])

# Build vocabulary from training set
vocab = build_vocab_from_iterator(yield_tokens(raw_datasets['train']), specials=['<unk>', '<eos>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter):
    """Tokenize, numericalize, and append <eos> token."""
    data = [torch.tensor(vocab(tokenizer(item['text'])) + [vocab['<eos>']], dtype=torch.long)
            for item in raw_text_iter if item['text']]
    return torch.cat(tuple(data))

train_data = data_process(raw_datasets['train'])
val_data   = data_process(raw_datasets['validation'])

def batchify(data, batch_size):
    # Drop extra tokens so that data can be evenly divided
    n_batch = data.size(0) // batch_size
    data = data[:n_batch * batch_size]
    return data.view(batch_size, -1).t().contiguous()

train_data = batchify(train_data, BATCH_SIZE)
val_data   = batchify(val_data, BATCH_SIZE)

def get_batch(source, i):
    seq_len = min(SEQ_LEN, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data.to(DEVICE), target.to(DEVICE)

# ====== Model Definition ======
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Weight tying (encoder & decoder share weights)  [oai_citation:16‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
        self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE),
                weight.new_zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE))

model = LSTMModel(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)

# ====== Training Setup ======
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ====== Training Loop ======
def train():
    model.train()
    total_loss = 0.
    hidden = model.init_hidden(BATCH_SIZE)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQ_LEN)):
        data, targets = get_batch(train_data, i)
        # Detach hidden state to prevent backprop through entire history
        hidden = tuple(h.detach() for h in hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        # Gradient clipping  [oai_citation:17‡github.com](https://github.com/liux2/RNN-on-wikitext2?utm_source=chatgpt.com)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        if batch % 100 == 0 and batch > 0:
            cur_loss = total_loss / 100
            print(f'| Batch {batch:3d} | Loss {cur_loss:5.2f} | Perplexity {torch.exp(torch.tensor(cur_loss)):.2f}')
            total_loss = 0

# ====== Evaluation Function ======
def evaluate(data_source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            total_loss += SEQ_LEN * criterion(output, targets).item()
            hidden = tuple(h.detach() for h in hidden)
    return total_loss / (len(data_source) - 1)

# ====== Main Training Loop ======
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(val_data)
    print('-' * 89)
    print(f'| End of Epoch {epoch:3d} | Time: {(time.time() - epoch_start_time):5.2f}s '
          f'| Valid Loss {val_loss:5.2f} | Valid PPL {torch.exp(torch.tensor(val_loss)):.2f}')
    print('-' * 89)

# ====== Autoregressive Text Generation ======
def generate_text(prompt: str, max_len: int = 50, temperature: float = 1.0):
    model.eval()
    tokens = [vocab[token] for token in tokenizer(prompt)]
    hidden = model.init_hidden(1)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input_tensor, hidden)
            logits = output[-1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == vocab['<eos>']:
                break
            tokens.append(next_token)
            input_tensor = torch.tensor([next_token], dtype=torch.long).unsqueeze(1).to(DEVICE)

    inv_vocab = {v: k for k, v in vocab.get_stoi().items()}
    return ' '.join(inv_vocab[t] for t in tokens)

# Example usage:
print(generate_text("Once upon a time", max_len=30, temperature=0.8))