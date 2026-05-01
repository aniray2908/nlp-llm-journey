# Phase 3 — Transformer Deep Dive

> **Understanding and implementing the architecture behind BERT, GPT, and modern LLMs.** Build attention from scratch, understand pre-training and fine-tuning, and learn why transfer learning dominates modern NLP.

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
| **Pre-training** | Training on massive unlabeled text | Learns general language understanding |
| **Fine-tuning** | Training on task-specific data | Adapts to your specific task |
| **Transfer learning** | Knowledge from one task helps another | Why pre-trained models dominate |

---

## Part 1 — Implement Attention From Scratch ✅

### Files

- **`concepts/01_attention_mechanisms.md`** — Deep explanation of attention
- **`demos/01_attention_from_scratch.py`** — Complete implementation with tests

### What You Built

- Scaled dot-product attention (QK^T / √d_k → softmax → apply to values)
- Multi-head attention (multiple heads, parallel processing)
- Positional encoding (sinusoidal functions for position awareness)
- Feed-forward networks (expand → ReLU → contract)
- Full Transformer encoder blocks (attention + residual + FFN + residual)
- Stacked encoder (multiple blocks)

### Key Insights

1. **Attention solves RNN bottleneck** — parallel processing, no vanishing gradients
2. **Multi-head attention** — different heads learn different patterns
3. **Positional encoding** — enables order awareness in Transformers
4. **Residual connections** — make deep networks trainable
5. **No recurrence** — fully parallelisable, trains 10-100x faster than RNNs

---

## Part 2 — Read "Attention Is All You Need" Paper ✅

**Paper:** https://arxiv.org/abs/1706.03762 (Vaswani et al., 2017)

### Sections Read

- Abstract & Introduction — RNN limitations
- Section 3.1 — Encoder-Decoder Architecture
- Section 3.2 — Multi-Head Attention
- Section 3.3 — Applications of Attention
- Training details and results

### Why It Matters

This paper introduced Transformers in 2017. Everything modern (BERT, GPT, Claude) is built on this foundation. Reading it after implementing attention makes every equation click.

---

## Part 3 — Pre-training, Fine-tuning, and Transfer Learning ✅

### Files

- **`concepts/02_bert_finetuning.md`** — BERT, pre-training, fine-tuning, transfer learning
- **`demos/02_bert_finetuning.py`** — Load pre-trained BERT, evaluate on sentiment

### What You Learned

#### BERT (Bidirectional Encoder Representations from Transformers)

```
BERT:
  - Encoder-only Transformer (12 layers, 110M parameters)
  - Pre-trained on 3.3B words with masked language modelling
  - Bidirectional context (sees left AND right)
  - Can be fine-tuned for any downstream task
```

#### Pre-training vs Fine-tuning

```
Pre-training (one-time, expensive):
  - Objective: Masked language modelling (predict [MASK] tokens)
  - Data: Wikipedia, Books, Web (3.3B words)
  - Cost: 4 days on 64 TPU chips (~$7,000)
  - Result: General language understanding

Fine-tuning (quick, task-specific):
  - Objective: Your task (sentiment, NER, Q&A, etc.)
  - Data: Your labeled examples (hundreds to thousands)
  - Cost: Hours on single GPU
  - Result: Task-specific model
```

#### Transfer Learning

The paradigm that dominates modern NLP:

```
Step 1: Pre-train on massive corpus
  One-time investment, shared by everyone

Step 2: Download pre-trained model
  Free, takes seconds

Step 3: Fine-tune on your data
  Hours to days, minimal compute

Step 4: Deploy
  State-of-the-art performance with minimal training
```

Why it works: Pre-training teaches grammar, semantics, reasoning, world knowledge. All of these transfer to new tasks.

### Results from Sentiment Analysis

**Pre-trained BERT (without fine-tuning):**
- Accuracy: 49.6% (random guessing)
- Shows that pre-training alone isn't enough

**Why?** BERT learns masked language modelling, not sentiment classification. It needs task-specific fine-tuning to excel.

**Key Learning:** Pre-trained models are powerful, but only after fine-tuning on your specific task.

---

## Key Concepts Covered

### Transformers

- ✅ Self-attention mechanism
- ✅ Multi-head attention
- ✅ Positional encoding
- ✅ Encoder-decoder architecture
- ✅ Why no recurrence → parallelisation
- ✅ Bidirectional vs unidirectional

### BERT and Fine-tuning

- ✅ What BERT is and how it's structured
- ✅ Masked language modelling pre-training
- ✅ How fine-tuning works
- ✅ [CLS] token for classification
- ✅ Why pre-training + fine-tuning dominates
- ✅ Transfer learning paradigm
- ✅ BERT variants (DistilBERT, RoBERTa, domain-specific)

### Modern LLMs

- ✅ How BERT evolved into GPT
- ✅ Decoder-only vs Encoder-only
- ✅ Scaling laws (more parameters = better performance)
- ✅ In-context learning (few-shot without fine-tuning)
- ✅ From BERT (110M) → GPT-3 (175B) → Claude (unknown scale)

---

## Results and Takeaways

### What Pre-training Teaches

Through masked language modelling, BERT learns:

| Capability | Example |
|---|---|
| **Grammar** | Subject-verb agreement, tense, structure |
| **Semantics** | Word relationships, synonymy, antonymy |
| **World knowledge** | Facts, entities, relationships |
| **Reasoning** | Logic, causality, inference |
| **Style** | Formality, tone, register |

All of this transfers to downstream tasks.

### Why Transfer Learning Revolutionised NLP

**Before (2017):**
```
Need: 100,000+ labeled examples
Time: Weeks to train from scratch
Cost: Massive GPU clusters
Result: Mediocre performance
```

**After (2018+):**
```
Need: 100-1000 labeled examples
Time: Hours to fine-tune
Cost: Single GPU
Result: State-of-the-art performance
```

This shift enabled NLP to scale to new domains and languages rapidly.

### Evolution of Models

```
BERT (2018):      110M params, encoder-only, fine-tuning required
GPT-2 (2019):     1.5B params, decoder-only, few-shot learning
GPT-3 (2020):     175B params, in-context learning (0-shot)
GPT-4 (2023):     ~1.7T params, reasoning, multimodal
Claude (2023):    Decoder-only, RLHF + Constitutional AI
```

All share: pre-training → transfer learning → domain adaptation

---

## How This Connects to Phase 4 & 5

### Phase 4 — LLM Internals + Fine-tuning

Build on Phase 3 knowledge:
- Decoder-only models (GPT-style generation)
- Causal masking (can only see previous tokens)
- LoRA and PEFT (efficient fine-tuning)
- RAG (Retrieval Augmented Generation)
- Fine-tune a small LLM on custom data

### Phase 5 — Mini-GPT Capstone

Apply everything:
- Build GPT from scratch (nanoGPT)
- Train on Shakespeare or code
- Understand every line
- Deploy and generate text

---

## Files in This Phase

```
03_phase_3/
├── README.md (this file)
├── concepts/
│   ├── 01_attention_mechanisms.md
│   └── 02_bert_finetuning.md
└── demos/
    ├── 01_attention_from_scratch.py
    └── 02_bert_finetuning.py
```

---

## Key Takeaways

1. **Attention is revolutionary** — solves RNN bottleneck, enables parallelisation
2. **Bidirectional context matters** — BERT sees left and right, learns better
3. **Pre-training works** — learning on massive text teaches general understanding
4. **Fine-tuning is efficient** — adapting to your task takes hours, not weeks
5. **Transfer learning dominates** — sharing pre-trained weights scales NLP
6. **Scale matters** — bigger models learn better (up to a point)
7. **Pre-training alone isn't enough** — need task-specific fine-tuning

---

## Progress

```
Phase 0: PyTorch Fundamentals        ✅ COMPLETE (6/6 concepts)
Phase 1: NLP Fundamentals            ✅ COMPLETE (6/6 concepts + reading)
Phase 2: Sentiment Analyser          ✅ COMPLETE (baseline + LSTM + Gradio)
Phase 3: Transformers Deep Dive      ✅ COMPLETE
  Part 1: Attention                  ✅ (implementation + visualization)
  Part 2: Read paper                 ✅ ("Attention Is All You Need")
  Part 3: Pre-training & fine-tuning ✅ (BERT evaluation + concepts)
Phase 4: LLM Internals               ⏳ NEXT
Phase 5: Mini-GPT Capstone           ⏳ FUTURE
```

**You're at ~40% of the full roadmap. 🚀**

---

## Resources

- **Paper:** "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
- **Paper:** "BERT: Pre-training of Deep Bidirectional Transformers" (https://arxiv.org/abs/1810.04805)
- **Library:** Hugging Face Transformers (https://github.com/huggingface/transformers)
- **Visualization:** The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)
- **Guide:** The Illustrated BERT (https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

---

*Phase 3 — Transformers Deep Dive → Complete ✅*  
*Previous → Phase 2: Sentiment Analyser*  
*Next → Phase 4: LLM Internals + Fine-tuning*
