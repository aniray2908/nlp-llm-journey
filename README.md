# NLP → LLM Learning Journey

> A structured, hands-on path from classical NLP to building language models. 6 phases, 3 portfolio projects, everything from tokenisation to training a mini-GPT from scratch.

---

## Status

```
Phase 0: PyTorch fundamentals              ✅ COMPLETE
Phase 1: NLP fundamentals                  ✅ COMPLETE  
Phase 2: Demo project — sentiment analyser ✅ COMPLETE
Phase 3: Transformer deep dive             ✅ COMPLETE
Phase 4: LLM internals + fine-tuning       🌱 IN PROGRESS
Phase 5: Capstone — mini GPT               ⏳ PENDING
```

**Progress: 15/39 tasks complete (~40%)**

---

## What This Is

A **learning roadmap and project repository** for understanding language models from first principles. Not a course or tutorial — a self-directed journey with:

- **6 phases** covering tokenisation → Transformers → LLMs
- **Concept notes** (markdown) explaining every idea
- **Demo notebooks** with hands-on code
- **3 portfolio projects** you can ship
- **No fixed timeline** — go at your own pace

---

## Who This Is For

- **Strong math/ML foundation** but new to NLP
- **Intermediate Python** and comfortable with PyTorch
- **Variable study time** (30min–2hrs/day, flexible)
- **Goal:** Get hired in AI/ML, build AI products, do research

---

## The Journey

### Phase 0 — PyTorch Warm-up ✅

**6 concepts, 6 demos**

Master PyTorch fundamentals needed for everything else:
- Tensors, slicing, broadcasting, GPU
- Autograd and backpropagation
- `nn.Module` and custom layers
- Training loops, DataLoaders, optimisers

**Projects:** Linear regression, MLPs, MNIST classifier (98% accuracy)

**Folder:** `00_pytorch_warmup/`

---

### Phase 1 — NLP Fundamentals ✅

**6 concepts, 6 demos, reading**

Classical NLP is the foundation. Understand it deeply:
- **Tokenisation** — word, subword (BPE), character-level
- **Text normalisation** — stemming, lemmatisation, stopwords
- **Bag of Words & TF-IDF** — sparse vector representations
- **Word embeddings** — Word2Vec, GloVe, semantic space
- **Language modelling** — n-grams, perplexity, autoregressive generation
- **Text classification** — complete pipeline, evaluation metrics

**Reading:** "Speech and Language Processing" (Jurafsky & Martin) Ch. 1–6

**Projects:** Tokeniser demo, embedding visualisations, n-gram models, text classification pipeline

**Folder:** `01_nlp_fundamentals/`

---

### Phase 2 — Sentiment Analyser Capstone ✅

**Your first real portfolio piece**

Build an end-to-end sentiment classifier on IMDB reviews:

**Baseline:** TF-IDF + Logistic Regression (82% accuracy)  
**Upgrade:** GloVe embeddings + PyTorch LSTM (67% accuracy)  
**Result:** Baseline wins! Shows that complex ≠ better. Start simple, measure, upgrade only if needed.

**Deliverables:**
- Training notebook with full pipeline
- Comprehensive README documenting findings
- Inference script for predictions
- Gradio web demo (interactive UI)

**Key learning:** Honest analysis of why baseline outperforms LSTM teaches more than fake success.

**Folder:** `02_demo_sentiment_analyser/`

---

### Phase 3 — Transformer Deep Dive ✅

**Understanding and implementing Transformers from scratch**

#### Part 1: Implement Attention From Scratch ✅

- Scaled dot-product attention (QK^T / √d_k → softmax → values)
- Multi-head attention (multiple heads, parallel processing)
- Positional encoding (sinusoidal, enables position awareness)
- Feed-forward networks (expand → ReLU → contract)
- Full Transformer encoder blocks
- Stacked encoder

**File:** `03_phase_3/demos/01_attention_from_scratch.py`

#### Part 2: Read "Attention Is All You Need" ✅

**Paper:** https://arxiv.org/abs/1706.03762 (Vaswani et al., 2017)

This paper introduced Transformers. Reading it after coding attention makes every equation click.

#### Part 3: Pre-training, Fine-tuning, Transfer Learning ✅

- What BERT is and how it's structured (encoder-only, 110M params)
- Masked language modelling (pre-training objective)
- Fine-tuning for sentiment classification
- Why pre-training alone isn't enough (50% accuracy without fine-tuning)
- Transfer learning paradigm (pre-train once, fine-tune many times)
- BERT variants and modern evolution

**File:** `03_phase_3/demos/02_bert_finetuning.py`

**Key insight:** Pre-trained BERT needs task-specific fine-tuning to excel. Pre-training teaches grammar and semantics, fine-tuning teaches task patterns.

**Folder:** `03_phase_3/`

---

### Phase 4 — LLM Internals + Fine-tuning 🌱

**From Transformers to GPT-style models**

- Decoder-only architecture and causal masking
- Autoregressive generation and sampling
- LoRA and PEFT for efficient fine-tuning
- RAG (Retrieval Augmented Generation)
- Fine-tune a small LLM (GPT-2 or LLaMA 3.2 1B) on custom data

**Folder:** `04_llm_internals_finetuning/` (coming next)

---

### Phase 5 — Capstone: Mini-GPT ⏳

**The project that sets you apart**

Build a character or word-level GPT from scratch (following Andrej Karpathy's nanoGPT):
- Implement tokeniser, embeddings, positional encoding
- Build full decoder-only Transformer stack
- Write training loop with gradient clipping and checkpointing
- Train on Shakespeare, code, or your own dataset
- Track loss curves, generate samples, visualise attention
- Write detailed README explaining every decision

This project demonstrates you understand language models at a deep level.

**Folder:** `05_capstone_mini_gpt/` (coming after Phase 4)

---

## How to Use This Repository

### 1. Start with Phase 0 if you're new to PyTorch

Read `00_pytorch_warmup/README.md`, work through concepts and demos.

### 2. Phase 1 is foundational — don't skip

Classical NLP teaches you principles that apply everywhere. Read the concept notes carefully, run the notebooks, do the reading from Jurafsky & Martin.

### 3. Phase 2 is your first real project

Build the sentiment analyser end-to-end. It's okay if models don't beat each other — the learning is in the process. This is your first portfolio piece.

### 4. Phases 3-5 build progressively

Each phase depends on earlier ones. By Phase 5, you'll have built a mini-GPT and understand language models deeply.

---

## Repository Structure

```
nlp-llm-journey/
├── README.md                        (this file)
├── 00_pytorch_warmup/
│   ├── README.md
│   ├── concepts/
│   └── demos/
├── 01_nlp_fundamentals/
│   ├── README.md
│   ├── concepts/
│   └── demos/
├── 02_demo_sentiment_analyser/
│   ├── README.md
│   ├── demos/
│   ├── inference.py
│   ├── gradio_demo.py
│   └── requirements.txt
├── 03_phase_3/
│   ├── README.md
│   ├── concepts/
│   │   ├── 01_attention_mechanisms.md
│   │   └── 02_bert_finetuning.md
│   └── demos/
│       ├── 01_attention_from_scratch.py
│       └── 02_bert_finetuning.py
├── 04_llm_internals_finetuning/     (coming next)
└── 05_capstone_mini_gpt/            (coming later)
```

---

## Key Principles

### 1. Understand deeply, not just build

Every phase has concept notes. Read them. They're written to be references you return to.

### 2. Always start simple

Baseline (TF-IDF + LogReg) beats complex models (LSTM) on small datasets. Measure before optimising.

### 3. Honest analysis matters

Document what worked and what didn't. Why baseline beat LSTM teaches more than fake success.

### 4. Data leakage is the biggest risk

Always split data before any processing. Always fit vectorisers on train only. This one mistake ruins projects.

### 5. Evolution, not revolution

BoW → embeddings → contextual embeddings → LLMs. Each step solves problems of the previous. Understand this arc.

---

## Learning Philosophy

**Classical NLP isn't outdated — it's foundational.**

Modern deep learning didn't replace it; it improved on it while keeping the same principles:

- Tokenisation is still essential (just more sophisticated)
- Feature engineering still matters (embeddings are learned features)
- Language modelling is still the training signal (just with Transformers)
- The pipeline is still the same (data → features → model → evaluate)

Engineers who understand this history navigate modern ML better.

---

## Time Estimates

| Phase | Topics | Time | Status |
|-------|--------|------|--------|
| 0 | PyTorch | 4–5 days | ✅ |
| 1 | NLP fundamentals | 5–7 days | ✅ |
| 2 | Sentiment analyser | 3–4 days | ✅ |
| 3 | Transformers | 5–7 days | ✅ |
| 4 | LLM internals | 4–5 days | 🌱 |
| 5 | Mini-GPT capstone | 5–7 days | ⏳ |
| **Total** | | **26–35 days** | |

Assuming 1.5–2 hours/day. Adjust based on your pace.

---

## Resources

### Textbooks

- **"Speech and Language Processing"** (Jurafsky & Martin, 3rd ed.) — NLP bible, free online
- **"Attention Is All You Need"** (Vaswani et al., 2017) — Transformer paper, essential reading
- **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018) — BERT paper

### Video Resources

- **Andrej Karpathy's "Neural Networks: Zero to Hero"** — Excellent PyTorch + nanoGPT walkthrough
- **"Attention Is All You Need" explained** — Multiple video breakdowns on YouTube

### Code References

- **nanoGPT** (Karpathy) — Minimal GPT implementation
- **Hugging Face Transformers** — Production-grade library
- **PyTorch Tutorials** — Official docs

---

## Goals

By the end, you'll have:

✓ **Deep understanding** of how language models work (not just API usage)  
✓ **Hands-on skills** — implemented attention, fine-tuned BERT, trained a GPT  
✓ **Portfolio projects** — 3 real projects you can show employers  
✓ **Intuition for the pipeline** — can build anything in NLP  
✓ **Knowledge to read papers** — understand modern research  

---

## Next Steps

1. If you're on **Phase 0** — Work through PyTorch fundamentals
2. If you're on **Phase 1** — Complete NLP concepts, do the reading
3. If you're on **Phase 2** — Build sentiment analyser (it's the hardest step, hardest means most learning)
4. If you're on **Phase 3+** — Keep building!
5. If you're done Phase 3 — Start Phase 4 (LLM internals)

---

## Contributing

This is a personal learning journey, but if you find errors or have suggestions:
- Open an issue
- Submit a PR
- Fork and build your own version

---

## Status Tracking

- **Phase 0:** 6/6 concepts ✅, 6/6 demos ✅
- **Phase 1:** 6/6 concepts ✅, 6/6 demos ✅, reading ✅
- **Phase 2:** Notebook ✅, README ✅, inference ✅, Gradio demo ✅
- **Phase 3:** Attention implementation ✅, paper reading ✅, BERT + fine-tuning ✅
- **Phase 4:** Starting soon...

---

## Author

Built as a self-directed learning project.

**Timeline:** Started April 2026, ongoing.

---

*Last updated: May 1, 2026*

---

## Philosophy

> Understanding language models deeply requires understanding the full journey: from tokenisation to Transformers. Skip the foundation, and you'll hit ceilings you can't explain. Build it all from scratch, and you can build anything.

This repository is that journey.

Start with Phase 0. Go at your pace. Build things. Understand deeply.

🚀
