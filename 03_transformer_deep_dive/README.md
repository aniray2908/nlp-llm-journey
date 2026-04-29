# Phase 3 — Transformer Deep Dive

> **Understanding and implementing the architecture behind BERT, GPT, and modern LLMs.** Build attention from scratch, then fine-tune pre-trained models.

---

## What You'll Learn

| Concept | What | Why |
|---------|------|-----|
| **Attention** | "Which tokens matter for predicting this one?" | Core mechanism of Transformers |
| **Scaled dot-product attention** | Efficient attention computation | How it's actually implemented |
| **Multi-head attention** | Multiple attention patterns in parallel | Different relationships per head |
| **Positional encoding** | Adding position information | Tells the model word order |
| **Feed-forward blocks** | MLP applied per position | Adds non-linearity |
| **Full Transformer encoder** | Stacking everything together | How BERT works |
| **BERT fine-tuning** | Training on custom task | State-of-the-art for classification |

---

## Part 1 — Implement Attention From Scratch

### Files

- **`01_attention_from_scratch.py`** — Complete implementation with tests
  - Scaled dot-product attention
  - Multi-head attention
  - Positional encoding
  - Feed-forward networks
  - Full Transformer encoder blocks
  - Preview of pre-trained BERT

### How to Run

**In Colab or local Jupyter:**

```python
# Run the script
exec(open('01_attention_from_scratch.py').read())
```

Or in a terminal:
```bash
python 01_attention_from_scratch.py
```

### What You'll See

**Attention visualisations:**
- Heatmap of which tokens attend to which (self-attention on "The cat sat on the mat")
- Attention distribution for individual query positions
- Clear pattern: tokens mostly attend to themselves and nearby tokens

**Positional encoding heatmap:**
- Low frequencies (top rows) change slowly across positions
- High frequencies (bottom rows) oscillate rapidly
- Each position gets a unique encoding

**Layer stacking:**
- Single block: attention + residual + FFN + residual
- Multiple blocks: stacked encoder
- Parameter counts for each component

### Key Insights

1. **Attention solves RNN bottleneck**
   - RNNs: sequential (slow, vanishing gradients)
   - Attention: parallel (fast, direct long-range connections)

2. **Multi-head attention = multiple patterns**
   - Head 1 might learn "subject-verb agreement"
   - Head 2 might learn "object references"
   - Head 3 might learn "articles and determiners"

3. **Positional encoding enables order awareness**
   - Attention alone is permutation-invariant (order doesn't matter)
   - Adding positional info tells the model position

4. **Residual connections make deep networks trainable**
   - Without them: deep networks have vanishing gradients
   - With them: gradient flows directly through

---

## Part 2 — Read "Attention Is All You Need" Paper

**Paper:** https://arxiv.org/abs/1706.03762 (Vaswani et al., 2017)

**Reading guide:**

| Section | Focus | Time |
|---------|-------|------|
| Abstract & Intro | Understand the problem (RNN limitations) | 10 min |
| Section 3.1 | Encoder-Decoder Architecture | 15 min |
| Section 3.2 | Multi-Head Attention | 20 min |
| Section 3.3 | Applications of Attention | 10 min |
| Section 4 | Training details (optional but good) | 15 min |

**Total reading time:** ~1 hour

**While reading:**
- Visualise the architecture diagrams as you read
- Match equations to the code you wrote
- Notice how every detail has a reason

**Key equations to understand:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## Part 3 — Fine-tune BERT on Sentiment Analysis

### Coming Next

- Load pre-trained BERT from Hugging Face
- Fine-tune on IMDB reviews (your Phase 2 data)
- Compare against Phase 2 baseline (TF-IDF + LogReg) and LSTM

**Expected results:** BERT will significantly outperform both, demonstrating why Transformers revolutionised NLP.

---

## Key Files in This Phase

```
03_transformer_deep_dive/
├── README.md (this file)
├── concepts/
│   └── 01_attention_mechanisms.md
└── demos/
    └── 01_attention_from_scratch.py
```

---

## Results & Learnings

### Scaled Dot-Product Attention

```
Input: Q, K, V (batch, seq_len, d_k/d_v)
Output: (batch, seq_len, d_v)

Each token attends to all tokens weighted by relevance.
Fully parallelisable (no sequential processing).
```

### Multi-Head Attention

```
Input: (batch, seq_len, d_model)
8 heads → each operates on d_model/8 dimensions
Output: (batch, seq_len, d_model)

Multiple patterns learned simultaneously.
Each head specialises in different relationships.
```

### Positional Encoding

```
Sinusoidal functions at different frequencies.
Low frequencies: capture long-range positions.
High frequencies: capture local positions.
Result: unique, continuous representation per position.
```

### Full Transformer Block

```
Input → LayerNorm → Attention → Residual
     ↘_______________________________↗

Output → LayerNorm → FFN → Residual
     ↘_____________↗

No recurrence, fully parallelisable.
Gradient flows directly (no vanishing gradient problem).
```

---

## Why This Matters

Everything in modern NLP uses this architecture:

| Model | Architecture |
|-------|--------------|
| **BERT** | Encoder-only (bidirectional) |
| **GPT** | Decoder-only (causal, left-to-right) |
| **T5** | Encoder-decoder |
| **LLaMA, Claude, etc.** | Decoder-only at scale |

Understanding Transformers means understanding the foundation of modern language models.

---

## Next Steps

1. ✅ **Run `01_attention_from_scratch.py`** — see it work
2. 📖 **Read the paper** — understand the theory
3. 🤖 **Fine-tune BERT** (Part 3) — apply to sentiment analysis
4. 📊 **Compare results** — BERT vs baseline vs LSTM
5. 📝 **Document findings** — update README with results

---

## Resources

- **Paper:** "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
- **Code:** Hugging Face Transformers (https://github.com/huggingface/transformers)
- **Visualisation:** The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)
- **Explanation:** 3Blue1Brown's attention series (YouTube)

---

*Phase 3 — Transformers Deep Dive → you are here*  
*Previous → Phase 2: Sentiment Analyser (complete)*  
*Next → Part 3: BERT Fine-tuning*
