# Decoder-Only Architecture — Causal Masking and Autoregressive Generation

> **Decoder-only models (GPT, LLaMA, Claude) power all modern LLMs.** Unlike BERT's bidirectional encoding, decoders generate text one token at a time, predicting the future by learning only from the past.

---

## Table of Contents

1. [Encoder vs Decoder vs Encoder-Decoder](#1-encoder-vs-decoder-vs-encoder-decoder)
2. [Decoder-Only Architecture](#2-decoder-only-architecture)
3. [Causal Masking](#3-causal-masking)
4. [Autoregressive Generation](#4-autoregressive-generation)
5. [Training Objective](#5-training-objective)
6. [Sampling Strategies](#6-sampling-strategies)
7. [GELU vs ReLU](#7-gelu-vs-relu)
8. [From GPT to Modern LLMs](#8-from-gpt-to-modern-llms)
9. [How This Connects to Fine-tuning](#9-how-this-connects-to-fine-tuning)

---

## 1. Encoder vs Decoder vs Encoder-Decoder

### Three Transformer Architectures

```
ENCODER (BERT):
  Input:  "The cat sat on the mat"
  Sees:   All tokens (left and right)
  Output: Contextual representations
  Use:    Understanding (classification, Q&A, NER)

DECODER (GPT, LLaMA, Claude):
  Input:  "The cat sat on the"
  Sees:   Only previous tokens (left only)
  Output: Probability of next token
  Use:    Generation (text, code, dialogue)

ENCODER-DECODER (T5, BART):
  Encoder: Understands input (bidirectional)
  Decoder: Generates output (causal)
  Use:     Translation, summarisation
```

### When to Use Which

| Task | Architecture | Why |
|------|-------------|-----|
| Sentiment classification | Encoder | Needs full context to understand |
| Text generation | Decoder | Generates one token at a time |
| Machine translation | Encoder-Decoder | Understands source, generates target |
| Summarisation | Encoder-Decoder | Understands input, generates summary |
| Chat / Q&A | Decoder | Generate responses token by token |

---

## 2. Decoder-Only Architecture

### Structure

A decoder-only model is a stack of **Transformer blocks with causal masking:**

```
Input token IDs
  ↓
Token embeddings (learned)
+ Positional embeddings (learned or sinusoidal)
  ↓
Decoder block 1:
  Causal self-attention (can only see past)
  Residual + LayerNorm
  Feed-forward (GELU activation)
  Residual + LayerNorm
  ↓
Decoder block 2
  ↓
...
  ↓
Decoder block N
  ↓
Final LayerNorm
  ↓
Linear layer (d_model → vocab_size)
  ↓
Softmax → probability distribution over tokens
  ↓
Sample or argmax → next token
```

### Key Difference from BERT

Identical structure to BERT's encoder blocks, with **one critical difference:**

```
BERT encoder block:  Multi-head attention (no mask)
GPT decoder block:   Multi-head attention (causal mask)

That's it. One triangular matrix changes everything.
```

### Parameter Counts

```
GPT-2 small:    12 layers, 12 heads, 768 dim  = 124M parameters
GPT-2 large:    36 layers, 20 heads, 1280 dim = 774M parameters
GPT-3:          96 layers, 96 heads, 12288 dim = 175B parameters
LLaMA 3.2 1B:   16 layers, 32 heads, 2048 dim = 1B parameters
LLaMA 3.2 3B:   28 layers, 24 heads, 3072 dim = 3B parameters
```

---

## 3. Causal Masking

### The Problem Without Masking

```
Without causal masking (WRONG for generation):
  Predicting "sat" in "The cat sat on the mat":
  Model can see: "The cat" AND "on the mat" (future!)
  This is cheating — it knows the answer already

With causal masking (CORRECT):
  Predicting "sat" in "The cat sat on the mat":
  Model can see: "The cat" only
  Cannot see: "on the mat"
  Realistic — only knows what came before
```

### The Causal Mask

A **lower triangular matrix** that blocks future positions:

```
Tokens:  The  cat  sat  on  the  mat
The  [  1    0    0    0    0    0  ]  ← can only see itself
cat  [  1    1    0    0    0    0  ]  ← sees "The", "cat"
sat  [  1    1    1    0    0    0  ]  ← sees "The", "cat", "sat"
on   [  1    1    1    1    0    0  ]
the  [  1    1    1    1    1    0  ]
mat  [  1    1    1    1    1    1  ]  ← sees all

1 = allowed to attend
0 = blocked (future position)
```

### Implementation

```python
def create_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))

# Applied in attention:
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
scores = scores.masked_fill(mask == 0, float('-inf'))
# -inf → softmax → 0 (future positions effectively blocked)
weights = torch.softmax(scores, dim=-1)
```

---

## 4. Autoregressive Generation

### The Process

Decoders generate text **one token at a time**, feeding each output back as input:

```
Step 1: Input ["The"]
        → Model predicts: P("cat")=0.6, P("dog")=0.3, ...
        → Sample: "cat"

Step 2: Input ["The", "cat"]
        → Model predicts: P("sat")=0.7, P("ran")=0.2, ...
        → Sample: "sat"

Step 3: Input ["The", "cat", "sat"]
        → Model predicts: P("on")=0.8, P("in")=0.1, ...
        → Sample: "on"

...continue until <END> token or max length
```

### Why "Autoregressive"?

**Auto** = self, **Regressive** = depends on previous outputs

```
Each new token depends on all previously generated tokens.
Output at step t feeds into input at step t+1.
This creates a dependency chain through the generated sequence.
```

### Context Window

Each decoder has a maximum sequence length it can handle:

```
GPT-2:    1,024 tokens
GPT-3:    4,096 tokens
GPT-4:    128,000 tokens
Claude:   200,000 tokens (Claude 3)
LLaMA:    128,000 tokens (LLaMA 3)
```

Beyond the context window, the model can't attend to earlier tokens.

---

## 5. Training Objective

### Next Token Prediction

The training signal is simple: **predict the next token at every position.**

```
Sentence: "The cat sat on the mat"

Input:  ["The",  "cat", "sat", "on",  "the"]
Target: ["cat",  "sat", "on",  "the", "mat"]

For each position:
  "The"  → predict "cat"   → compute loss
  "cat"  → predict "sat"   → compute loss
  "sat"  → predict "on"    → compute loss
  "on"   → predict "the"   → compute loss
  "the"  → predict "mat"   → compute loss

One sentence gives N-1 training signals.
Scale to trillions of tokens → trillions of training signals.
```

### Loss Function

Cross-entropy at every position:

```python
# Flatten predictions and targets
logits = logits.view(-1, vocab_size)   # (batch * seq_len, vocab_size)
targets = targets.view(-1)             # (batch * seq_len,)

loss = F.cross_entropy(logits, targets)
```

### Why This Works

Training to predict next tokens forces the model to learn:

```
"The cat ___":
  Must know grammar → verb likely
  Must know semantics → "sat", "slept", "ran" all reasonable

"The capital of France is ___":
  Must know world facts → "Paris"

"def fibonacci(n):":
  Must know code patterns → function body follows

Predicting well = understanding language = learning everything
```

---

## 6. Sampling Strategies

Once you have probabilities over the vocabulary, how do you pick the next token?

### Greedy Decoding

```python
next_token = logits.argmax(dim=-1)
```

- Always picks highest probability token
- Deterministic (same output every time)
- Often repetitive and boring

### Temperature Sampling

```python
logits = logits / temperature
probs = softmax(logits)
next_token = torch.multinomial(probs, 1)
```

```
Temperature = 0.1:  Very focused (near greedy)
Temperature = 1.0:  Original distribution
Temperature = 2.0:  Very random and diverse
```

### Top-k Sampling

```python
# Zero out all but top k logits
values, _ = torch.topk(logits, k)
logits[logits < values[-1]] = -inf
probs = softmax(logits)
next_token = torch.multinomial(probs, 1)
```

Only samples from the k most likely tokens. Prevents very unlikely tokens from being chosen.

### Top-p (Nucleus) Sampling

```python
# Sort probabilities descending
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
# Remove tokens beyond cumulative probability p
sorted_probs[cumulative_probs > p] = 0
next_token = torch.multinomial(sorted_probs, 1)
```

Samples from the smallest set of tokens whose cumulative probability exceeds p.

Most modern LLMs use **top-p with temperature** as default.

### Comparison

| Strategy | Diversity | Quality | Use case |
|----------|-----------|---------|----------|
| Greedy | None | High | Factual Q&A |
| Temperature | High | Variable | Creative writing |
| Top-k | Medium | Medium | General use |
| Top-p | Adaptive | High | Most LLMs |

---

## 7. GELU vs ReLU

GPT uses **GELU** (Gaussian Error Linear Unit) instead of ReLU in feed-forward blocks:

```
ReLU(x) = max(0, x)
  - Hard cutoff at 0
  - Gradient = 0 for negative inputs (dead neurons)

GELU(x) = x * Φ(x)  where Φ is the Gaussian CDF
  - Smooth approximation
  - Gradient flows even for small negative values
  - Empirically works better for language models
```

Visually:

```
x:     -3   -2   -1    0    1    2    3
ReLU:   0    0    0    0    1    2    3
GELU:  -0.004 -0.045 -0.159  0  0.841  1.955  2.996
```

The smooth curve helps gradient flow during training, especially in deep networks.

---

## 8. From GPT to Modern LLMs

### GPT-2 (2019) — The Foundation

```
Architecture: 12-48 layers, decoder-only
Training:     WebText dataset (40GB)
Key insight:  Language models are multitask learners
              Train on text → gets translation, summarisation, Q&A for free
```

### GPT-3 (2020) — Scale Changes Everything

```
Architecture: 96 layers, 175B parameters
Training:     300B tokens from Common Crawl, books, Wikipedia
Key insight:  In-context learning (few-shot prompting works!)
              No fine-tuning needed for many tasks
```

### LLaMA (2023) — Open Source Revolution

```
Architecture: Similar to GPT but with improvements
Improvements:
  - RoPE (Rotary Position Embedding) instead of learned positions
  - RMSNorm instead of LayerNorm (faster)
  - SwiGLU instead of GELU (better)
  - Grouped Query Attention (faster inference)
Key insight:  Smaller, open models can compete with larger closed ones
```

### Modern Improvements Summary

| Component | Old (GPT-2) | New (LLaMA) | Benefit |
|-----------|------------|-------------|---------|
| Position encoding | Learned | RoPE | Better long sequences |
| Normalisation | LayerNorm | RMSNorm | Faster, more stable |
| Activation | GELU | SwiGLU | Better performance |
| Attention | Full MHA | Grouped Query | Faster inference |

### Claude

```
Architecture: Decoder-only (like GPT)
Training:     Pre-training + RLHF + Constitutional AI
Key insight:  Alignment matters as much as capability
              Training to be helpful, harmless, honest
```

---

## 9. How This Connects to Fine-tuning

Understanding decoder architecture is essential for fine-tuning:

### Why Architecture Matters for Fine-tuning

```
Fine-tuning = adapting pre-trained weights to new task

To do this efficiently, you need to know:
  - Which layers to freeze (earlier layers = general features)
  - Which layers to update (later layers = task-specific)
  - How many parameters to update (LoRA: update a tiny fraction)
  - What the training objective is (still next-token prediction)
```

### Fine-tuning a Decoder

```
Step 1: Load pre-trained decoder (GPT-2, LLaMA)
        All weights from pre-training intact

Step 2: Prepare task-specific data
        Format as text (decoders only understand text)
        "Question: What is the capital of France?\nAnswer: Paris"

Step 3: Continue training with next-token prediction
        Same objective as pre-training, but on your data
        Model learns your task patterns

Step 4: Evaluate on held-out set
        Measure task-specific metrics
```

### LoRA Preview

Full fine-tuning updates all 7B parameters (expensive). LoRA updates only 0.1% (cheap):

```
Full fine-tuning:  Update all weights W
                   Memory: huge, slow

LoRA:              Freeze W, add small matrices A, B
                   Only train A and B (rank r << d)
                   Memory: tiny, fast
                   Performance: nearly identical
```

This is exactly what Phase 4 Part 2 covers.

---

## Summary

**Decoder-only architecture is the foundation of all modern LLMs:**

1. **Causal masking** — can only see past tokens (enforces left-to-right generation)
2. **Autoregressive** — generate one token at a time, feed back as input
3. **Training** — predict next token at every position (N signals per sentence)
4. **Sampling** — temperature, top-k, top-p control diversity vs quality
5. **Scale** — more layers, more heads, more data = better models

The difference between BERT and GPT is literally one triangular matrix. That one constraint — "only see the past" — is what makes generation possible and powers every LLM you've ever used.

---

*Phase 4 — concept 01 → you are here*  
*Previous → Phase 3: Transformer Deep Dive (complete)*  
*Next → Read: Karpathy's nanoGPT + "Unreasonable Effectiveness of RNNs"*  
*Then → LoRA and PEFT — efficient fine-tuning*
