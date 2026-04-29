# Attention Mechanisms — The Core of Transformers

> **The single most important mechanism in modern NLP.** Attention answers one question: "Which previous tokens matter for predicting this one?" Everything in Transformers flows from this.

---

## Table of Contents

1. [The Problem with RNNs](#1-the-problem-with-rnns)
2. [What is Attention?](#2-what-is-attention)
3. [Scaled Dot-Product Attention](#3-scaled-dot-product-attention)
4. [Multi-Head Attention](#4-multi-head-attention)
5. [Why Attention Works](#5-why-attention-works)
6. [Positional Encoding](#6-positional-encoding)
7. [Feed-Forward Networks](#7-feed-forward-networks)
8. [Full Transformer Block](#8-full-transformer-block)
9. [How This Connects to BERT & GPT](#9-how-this-connects-to-bert--gpt)

---

## 1. The Problem with RNNs

**RNNs process sequences one token at a time:**

```
Input:  "The cat sat on the mat"
        ↓ (token 1: "The")
Hidden state h₁
        ↓ (token 2: "cat")
Hidden state h₂
        ↓ (token 3: "sat")
Hidden state h₃
        ...
```

Each hidden state is a bottleneck — it must carry all information from previous tokens into the future. The further you go, the more information gets squeezed through this narrow channel.

### Why this is a problem

```
Predicting token at position 50:

With RNN:
  "The" (position 1) → h₁ → h₂ → ... → h₅₀
  Signal has to pass through 49 intermediate states
  By the time it reaches h₅₀, "The" is diluted/forgotten

With Attention:
  "The" (position 1) directly attends to position 50
  No bottleneck, full signal preserved
```

**Vanishing gradients:** Backpropagating through 50 steps often causes gradients to vanish. RNNs struggle with long-range dependencies.

### The solution: Attention

**Every token can directly attend to every other token, in parallel.**

```
Input: "The cat sat on the mat"

When predicting next token after "sat":
  - Attend to "The": 0.1 (low relevance)
  - Attend to "cat": 0.6 (high relevance — subject)
  - Attend to "sat": 0.2 (verb itself)
  - Attend to "on": 0.05 (preposition)
  - Attend to "the": 0.03
  - Attend to "mat": 0.05 (object)

Sum of attention weights = 1.0 (probability distribution)
```

No sequential processing, no bottleneck, no vanishing gradients. **All tokens process in parallel.**

---

## 2. What is Attention?

### The Core Question

**"Given a query, which keys are relevant, and what values do they contribute?"**

```
Query (q):     "What's happening at position i?"
Keys (k):      "What is each token?" (all positions)
Values (v):    "What information does each token have?" (all positions)

Attention weights = how much each key matches the query
Output = weighted sum of values
```

### Example: Predicting after "sat"

```
Query: "sat" embedding
  "What comes after 'sat'?"

Keys: All token embeddings
  "The": keyword embedding
  "cat": noun embedding
  "sat": verb embedding
  "on": preposition embedding
  "the": article embedding
  "mat": noun embedding

Attention mechanism:
  How much does "sat" match each token?
    "The" ← low match
    "cat" ← high match (subject of the verb)
    "sat" ← medium match (the verb itself)
    "on" ← low-medium match (preposition that follows)
    ...

Output: Weighted sum of all embeddings, weighted by relevance
```

### The "Attention" Intuition

Imagine you're reading: "The cat sat on the **mat**"

To predict what comes after "mat", you attend to:
- "The" — tells you about articles
- "cat" — the subject
- "sat" — the verb
- "on" — the preposition
- "the" — another article
- "mat" — the object

You give more attention (weight) to relevant tokens ("mat", "cat", "sat") and less to irrelevant ones.

**That's literally what the attention mechanism does.** Compute relevance scores and use them to weight the information.

---

## 3. Scaled Dot-Product Attention

### The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
  Q = Query matrix (batch, seq_len, d_k)
  K = Key matrix   (batch, seq_len, d_k)
  V = Value matrix (batch, seq_len, d_v)
  d_k = dimension of keys (scaling factor)
```

### Step-by-Step

#### Step 1: Compute similarities (QK^T)

```
Q shape: (seq_len, d_k)    e.g., (6, 64)
K shape: (seq_len, d_k)    e.g., (6, 64)
QK^T shape: (seq_len, seq_len)  e.g., (6, 6)

Each element (i, j) of QK^T = dot product of query i with key j
Result: similarity scores for all pairs

Example for query at position 3 ("sat"):
  score(sat, The)   = dot(embed_sat, embed_The)   = 2.1
  score(sat, cat)   = dot(embed_sat, embed_cat)   = 8.3  (high!)
  score(sat, sat)   = dot(embed_sat, embed_sat)   = 5.7
  score(sat, on)    = dot(embed_sat, embed_on)    = 3.2
  score(sat, the)   = dot(embed_sat, embed_the)   = 1.5
  score(sat, mat)   = dot(embed_sat, embed_mat)   = 3.8
```

#### Step 2: Scale by √d_k

```
Why divide by √d_k?

Without scaling:
  - Dot products get very large (magnitudes grow with dimension)
  - softmax pushes all probability into one token (one very high score)
  - Gradients become small (saturated softmax)

With scaling:
  - Keeps dot products in reasonable range
  - softmax probabilities stay distributed
  - Better gradients during training

Example: d_k = 64, so √d_k ≈ 8
  Scaled scores:
    2.1 / 8 = 0.26
    8.3 / 8 = 1.04
    5.7 / 8 = 0.71
    3.2 / 8 = 0.40
    1.5 / 8 = 0.19
    3.8 / 8 = 0.47
```

#### Step 3: Apply softmax

```
softmax([0.26, 1.04, 0.71, 0.40, 0.19, 0.47])
= [0.11, 0.27, 0.20, 0.15, 0.08, 0.19]

Now it's a probability distribution:
  - "cat" (the, at position 1) gets 27% of attention ← highest
  - "sat" (position 2) gets 20%
  - "mat" (position 5) gets 19%
  - Others get lower weights
  - Sums to 1.0
```

#### Step 4: Apply to values (weighted sum)

```
Output = 0.11 * value_The + 0.27 * value_cat + 0.20 * value_sat 
         + 0.15 * value_on + 0.08 * value_the + 0.19 * value_mat

Result: weighted sum of all token embeddings
  High weight on "cat" and "sat" (most relevant)
  Lower weight on "the", "on" (less relevant)
```

### Complete Example

```
Input:     "The cat sat on the mat"
Query at position 3 (after "sat"):

1. Embed all tokens
2. Compute all dot products: QK^T (6×6 matrix)
3. Scale by √64
4. Apply softmax → attention weights
5. Apply to values → output

Output[3] = weighted combination of all embeddings
  Heavily influenced by "cat" and "sat" (high attention)
  Slightly influenced by "on" and "mat" (low attention)
```

---

## 4. Multi-Head Attention

**One attention head sees one pattern. Multiple heads see multiple patterns.**

### Why Multiple Heads?

With a single attention head:
- When predicting "sat", head attends to "cat" (subject)
- But it might miss other patterns (verb agreement, tense, etc.)

With multiple heads:
- Head 1 attends to subjects (learns "cat" is important)
- Head 2 attends to verbs (learns "past tense verbs")
- Head 3 attends to objects (learns "mat" is the target)
- Head 4 attends to function words

Each head specialises in a different pattern.

### The Formula

```
MultiHeadAttention(Q, K, V) = Concat(head₁, head₂, ..., headₕ) W^O

Where:
  headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
  
  Wᵢ^Q, Wᵢ^K, Wᵢ^V = learnable projections for head i
  W^O = output projection (combines all heads)
  h = number of heads (e.g., 8 or 12)
```

### Example with 2 heads

```
Input embedding: (6, 512)  [6 tokens, 512-dim each]

Head 1:
  Project to d_model/2 = 256-dim
  Compute attention (256-dim dot products)
  Output: (6, 256)

Head 2:
  Project to 256-dim
  Compute attention
  Output: (6, 256)

Concatenate: (6, 256) + (6, 256) = (6, 512)
Apply W^O projection: (6, 512) → (6, 512)
Output: (6, 512)  [same shape as input]
```

### What Each Head Learns

Empirically, different heads learn different patterns:

| Head | Pattern | Example |
|---|---|---|
| Head 1 | Subject-verb agreement | "cat" attends to verb |
| Head 2 | Object references | "sat" attends to "on the mat" |
| Head 3 | Articles and determiners | Articles attend to nouns |
| Head 4 | Long-range dependencies | First token attends across sequence |

**The model learns what each head should do.**

---

## 5. Why Attention Works

### Problem 1: RNN Bottleneck ✓

**RNN:**
```
Position 50 depends on hidden state h₄₉
h₄₉ depends on h₄₈
...
h₁ depends on input

Signal flows sequentially: input → h₁ → h₂ → ... → h₅₀
Gradient flows backward: h₅₀ → h₄₉ → ... → h₁
Long path = vanishing gradients
```

**Attention:**
```
Position 50 directly attends to position 1
Gradient path: h₅₀ → h₁ (direct, one step)
No vanishing gradients
```

### Problem 2: Parallelisation ✓

**RNN:** Must process tokens sequentially (can't start h₂ until h₁ is done)

**Attention:** All tokens compute in parallel (all attention heads run at once)

```
RNN:    Token 1 → Token 2 → Token 3 → ...  (sequential)
        Time: 3 steps

Attention: Token 1 ↔ Token 2 ↔ Token 3 ...  (parallel)
           Time: 1 step (everything happens at once)
```

This is why Transformers train **10-100x faster** than RNNs on GPUs.

### Problem 3: Interpretability ✓

You can visualise attention weights and see what the model is paying attention to.

```
"The cat sat on the mat"

Attention weights for predicting next token after "sat":
  The:  ████░░░░░░░░░  (8%)
  cat:  ███████████░░░  (65%)  ← high attention
  sat:  ███████░░░░░░░  (25%)
  on:   ███░░░░░░░░░░░  (12%)
  the:  ██░░░░░░░░░░░░  (5%)
  mat:  ████░░░░░░░░░░  (10%)
```

With RNNs, you can't see what it was paying attention to.

---

## 6. Positional Encoding

### The Problem

Attention is **permutation-invariant** — the order of tokens doesn't matter:

```
"The cat sat" and "sat cat The" would produce the same attention scores
(Both have the same set of words, just different order)

But word order is crucial for language!
"cat bites dog" ≠ "dog bites cat"
```

### The Solution: Positional Encoding

Add position information to every embedding:

```
embedding_with_position = embedding + positional_encoding(position)
```

### Sinusoidal Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos = position in sequence (0, 1, 2, ...)
  i = dimension index (0, 1, 2, ..., d_model/2)
  d_model = embedding dimension (e.g., 512)
```

### Why Sinusoids?

Different frequencies encode different distances:
- **Low frequencies** — capture long-range positions (position 1 vs 100)
- **High frequencies** — capture local positions (position 50 vs 51)

```
Combined, they encode position unambiguously.

Also: sin and cos functions mean the PE is bounded (always between -1 and 1)
Good for numerical stability.
```

### Example

```
Position 0:  PE = [sin(0), cos(0), sin(0), cos(0), ...]  = [0, 1, 0, 1, ...]
Position 1:  PE = [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
Position 2:  PE = [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
...
Position 100: PE = [sin(100/1), cos(100/1), sin(100/100), cos(100/100), ...]

Each position gets a unique encoding.
The model can learn that position matters.
```

---

## 7. Feed-Forward Networks

After attention, each position is passed through an **identical MLP** (position-wise feed-forward).

```
FFN(x) = ReLU(xW₁ + b₁) W₂ + b₂

Where:
  W₁: (d_model, d_ff)  — project up (usually d_ff = 4 * d_model)
  W₂: (d_ff, d_model)  — project back down
```

### Why two layers?

```
Layer 1: d_model → d_ff
  Project to higher dimension
  Add non-linearity (ReLU)
  Increase model capacity

Layer 2: d_ff → d_model
  Project back to original dimension
  Mix information across dimensions
```

### Applied to every position

```
Input: (batch, seq_len, d_model)  e.g., (32, 50, 512)
Apply FFN: (batch, seq_len, d_model)  e.g., (32, 50, 512)
Output: (batch, seq_len, d_model)  e.g., (32, 50, 512)

Each of the 50 tokens gets its own FFN applied (independently).
```

### Not much computation

FFN is usually 10-20% of Transformer compute. Attention is the main cost.

---

## 8. Full Transformer Block

**One Transformer layer combines: attention + FFN**

```
Input: x (batch, seq_len, d_model)

1. Layer Norm
   x_norm = LayerNorm(x)

2. Multi-Head Attention
   attn_out = MultiHeadAttention(x_norm, x_norm, x_norm)

3. Residual connection
   x = x + attn_out

4. Layer Norm
   x_norm = LayerNorm(x)

5. Feed-Forward
   ffn_out = FFN(x_norm)

6. Residual connection
   x = x + ffn_out

Output: x (same shape as input)
```

### Why Residual Connections?

```
Without residuals:
  x → Attention → LayerNorm → FFN → LayerNorm → y
  
  If one layer learns nothing useful, gradient signal is lost

With residuals:
  x → Attention → + x → LayerNorm → FFN → + x → y
       ↘_______↗                    ↘_↗
  
  Information from x flows directly through
  Even if attention or FFN learn nothing, x still flows
  Easier to train deep networks
```

### Stacking Layers

**BERT has 12 layers, GPT-3 has 96 layers**

```
Input: (batch, seq_len, 768)

Block 1 → Block 2 → Block 3 → ... → Block 12

Each block processes in parallel (once input is ready)
Each block refines the representations
```

---

## 9. How This Connects to BERT & GPT

### BERT (Bidirectional Encoder Representations)

```
Input: "The cat sat on the mat"

BERT:
  Token embeddings
  + Positional encoding
  + Segment embeddings (A or B)
  → 12 encoder blocks
  → [CLS] token becomes sentence representation
  → Fine-tune on downstream task

Key difference from GPT: Can attend bidirectionally (to past AND future tokens)
Trained with masked language modelling (random tokens masked, predict them)
```

### GPT (Generative Pre-trained Transformer)

```
Input: "The cat sat on the"

GPT:
  Token embeddings
  + Positional encoding
  → 12-96 decoder blocks (with causal masking)
  → Output logits over vocabulary
  → argmax or sample next token

Key difference from BERT: Can only attend to PAST tokens (causal masking)
Trained with next-token prediction
Generates text autoregressive (one token at a time)
```

### The Core Difference

```
BERT:  Can see past AND future → good for understanding/classification
GPT:   Can see PAST only → good for generation

But the architecture is the same:
  Position embeddings
  Multi-head attention
  Feed-forward
  Residual connections
  Layer normalization
  Stacking
```

### What You'll Implement

In Phase 3 demos, you'll build:
1. Scaled dot-product attention
2. Multi-head attention
3. Positional encoding
4. Feed-forward block
5. Full encoder block
6. Stack multiple blocks

Then fine-tune pre-trained BERT (HuggingFace) on sentiment data and see it crush baseline and LSTM.

---

## Summary

**Attention is the answer to the RNN bottleneck.**

Instead of processing sequentially with hidden states:
```
h₁ → h₂ → h₃ → ... → hₙ (bottleneck)
```

Process in parallel with attention:
```
All tokens attend to all tokens simultaneously
No bottleneck, no vanishing gradients, fully parallelisable
```

**Multi-head attention** lets the model learn multiple patterns simultaneously.

**Positional encoding** tells the model about word order (which attention doesn't naturally capture).

**Feed-forward networks** add non-linearity and capacity.

**Stacking these blocks** creates deep, expressive models.

This is the architecture behind BERT, GPT, T5, and every modern LLM.

Implement it from scratch and you'll understand language models at a fundamental level.

---

*Phase 3 — concept 01 → you are here*  
*Next → Implement scaled dot-product attention in PyTorch*
