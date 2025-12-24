# xinSLM v08 — nanoChat-D4-inspired Transformer Architecture (Design Spec)

## 1) Goal
Build a minimal, readable decoder-only Transformer for chat/SFT that mirrors the key nanochat D4 ideas:

- RMSNorm throughout (no learned affine parameters)
- RoPE (no learned positional embeddings)
- Causal self-attention with GQA support
- QK normalization inside attention
- MLP with ReLU² activation
- No bias in linear layers
- Untied embedding and unembedding weights
- Logit softcap (tanh @ 15) before softmax/loss

This is an *architecture note*, not a full training recipe.

---

## 2) Baseline Reference (nanochat D4)
Reference config (as provided):

- n_layer = 4
- n_head = 2
- n_kv_head = 2
- n_embd = 256
- vocab_size = 65536
- MLP: 256 → 1024 → 256 (ReLU²)
- Output: lm_head 256 → 65536 + logit softcap tanh@15
- Params: ~36.7M

---

## 3) xinSLM v08 Proposed Config (editable)
### 3.1 Core sizes
Choose these first:

- `vocab_size`: ____  (depends on tokenizer; 32k/50k/65k are common)
- `context_len` (max seq): ____ (e.g., 512/1024/2048)
- `n_layer`: ____ (start with 4 to match D4)
- `n_embd`: ____ (start with 256 to match D4)
- `n_head`: ____ (must divide n_embd)
- `n_kv_head`: ____ (GQA: <= n_head; if equal, it’s standard MHA)
- `mlp_mult`: ____ (D4 uses 4×: 256→1024)

### 3.2 “D4 feature flags”
- `use_rmsnorm = true`
- `use_rope = true`
- `use_qk_norm = true`
- `mlp_activation = relu2`
- `use_bias = false`
- `tie_embeddings = false` (D4 says untied)
- `logit_softcap = 15.0`

---

## 4) Module-level Architecture

### 4.1 Top-level model
Input: token ids `idx` with shape `[B, T]`

1. Token embedding: `x = wte(idx)` → `[B, T, n_embd]`
2. (Optional) initial RMSNorm (D4 shows RMSNorm before blocks)
3. For each block `i in [0..n_layer-1]`:
   - `x = x + Attention(RMSNorm(x))`
   - `x = x + MLP(RMSNorm(x))`
4. Final RMSNorm: `x = RMSNorm(x)`
5. Logits: `logits = lm_head(x)` → `[B, T, vocab_size]`
6. Logit softcap: `logits = softcap(logits, cap=15.0)`
7. Loss: standard next-token cross entropy (masking handled by dataloader for SFT/chat)

**Softcap definition**
- `softcap(z, cap) = cap * tanh(z / cap)`

This bounds logits magnitude and can stabilize training.

---

## 5) Transformer Block Details (nanochat-style)

### 5.1 RMSNorm (no learned affine)
Given `x` shape `[..., d]`:
- `rms = sqrt(mean(x^2, dim=-1, keepdim=true) + eps)`
- `y = x / rms`

No gamma/beta parameters.

### 5.2 Causal Self-Attention with GQA + RoPE + QK Norm

#### Projections (no bias)
- `q = Wq x`
- `k = Wk x`
- `v = Wv x`
- `o = Wo attn_out`

Shapes:
- `q`: `[B, T, n_head, head_dim]`
- `k,v`: `[B, T, n_kv_head, head_dim]`
- `head_dim = n_embd / n_head`

#### RoPE
Apply RoPE to `q` and `k` (per head_dim) before attention.

#### QK normalization
Normalize q and k (typical variants: per-head RMSNorm or L2 norm). Pick one and document it.

Recommended minimal choice:
- `q = q / (sqrt(mean(q^2)+eps))`
- `k = k / (sqrt(mean(k^2)+eps))`

(where mean is over head_dim)

#### Attention
Compute causal attention with KV-head sharing for GQA (map each query head to a KV head group).

- scores: `a = (q @ k^T) * scale`
- apply causal mask
- weights: `p = softmax(a)`
- output: `y = p @ v`
- project: `out = Wo y`

---

## 6) MLP with ReLU²
D4 MLP: `d → 4d → d`, no bias.

- `h = W1 x` where `W1: d → 4d`
- `h = relu(h) ^ 2`  (elementwise square after ReLU)
- `out = W2 h` where `W2: 4d → d`

---

## 7) Embeddings: Untied vs tied
D4 says **untied**:
- `wte`: token embedding matrix `[vocab_size, n_embd]`
- `lm_head`: output projection `[n_embd, vocab_size]` (separate weights)

If you need fewer parameters, you can tie them, but that deviates from D4.

---

## 8) Training/Inference Implications (what this architecture expects)
- Use causal LM objective.
- For chat SFT: mask loss so it only applies to assistant tokens.
- GQA reduces KV cache size at inference when `n_kv_head < n_head`.
- Logit softcap helps prevent extreme logits (often stabilizes early training).

---

## 9) Implementation Checklist (minimal order)
1. Implement `RMSNorm` (no affine)
2. Implement `RoPE(q,k)` helper
3. Implement `Attention` with:
   - q/k/v projection (no bias)
   - RoPE
   - QK norm
   - causal masking
   - GQA head mapping
4. Implement `MLP` with ReLU²
5. Implement `Block` with residuals + pre-norm
6. Implement `Model` + `lm_head` + softcap
7. Add a tiny forward test:
   - random input → logits shape correct
   - loss runs without NaNs
8. Add a tiny generation test (greedy decode) to ensure cache/mask correctness (optional if you don’t build KV cache yet)

---

## 10) Open Decisions (answer these and lock them)
1. Tokenizer + `vocab_size`: do you want 32k, 50k, or 65,536?
2. `context_len`: 512 / 1024 / 2048?
3. Exact QK norm formula: RMS-based or L2-based?
4. GQA: keep `n_kv_head == n_head` initially (simpler), or implement true GQA from day 1?
5. Untied embeddings (match D4) vs tied (smaller)?

---

## 11) References and License Attribution
- nanochat repository: https://github.com/karpathy/nanochat
- nanochat license: MIT
  - License file in upstream repo: https://github.com/karpathy/nanochat/blob/master/LICENSE

This v08 architecture note is *inspired by* nanochat’s published design; if you copy or adapt code from nanochat, keep the upstream MIT license notice and attribution consistent with the MIT terms.
