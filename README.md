# NLP → LLM Learning Journey

> A structured, hands-on path from classical NLP to building and deploying language models. Six phases covering tokenisation, Transformers, fine-tuning, RAG, and a full capstone project — built from first principles.

---

## Status

```
Phase 0: PyTorch fundamentals              ✅ COMPLETE
Phase 1: NLP fundamentals                  ✅ COMPLETE
Phase 2: Demo project — sentiment analyser ✅ COMPLETE
Phase 3: Transformer deep dive             ✅ COMPLETE
Phase 4: LLM internals + fine-tuning       ✅ COMPLETE
Phase 5: Capstone — Educational Rewriter   ✅ COMPLETE
```

**Progress: ~37/39 tasks complete (~95%)**

---

## What This Is

A self-directed learning roadmap and project repository for understanding language models from first principles. Each phase builds on the previous, culminating in a fine-tuned LLM published to HuggingFace Hub.

- **6 phases** covering tokenisation through LLM deployment
- **Concept notes** — detailed markdown explanations for every topic
- **Demo notebooks** — hands-on implementation for every concept
- **3 portfolio projects** — sentiment analyser, fine-tuned rewriter, capstone analyzer

---

## The Journey

### Phase 0 — PyTorch Fundamentals ✅

Tensors, autograd, `nn.Module`, custom layers, training loops, DataLoaders, optimisers.

**Projects:** Linear regression, MLP on MNIST (98% accuracy)

**Folder:** `00_pytorch_warmup/`

---

### Phase 1 — NLP Fundamentals ✅

Tokenisation (BPE), text normalisation, Bag of Words, TF-IDF, Word2Vec, GloVe, n-gram language models, perplexity, text classification pipeline.

**Reading:** Speech and Language Processing (Jurafsky & Martin) Ch. 1–6

**Folder:** `01_nlp_fundamentals/`

---

### Phase 2 — Sentiment Analyser Capstone ✅

End-to-end sentiment classifier on IMDB reviews.

| Model | Accuracy | Notes |
|-------|----------|-------|
| TF-IDF + Logistic Regression | 82% | Baseline — won |
| GloVe + LSTM | 67% | Complex ≠ better |

**Deliverables:** Training notebook, inference script, Gradio demo

**Key learning:** Start simple. Measure. Upgrade only when needed.

**Folder:** `02_demo_sentiment_analyser/`

---

### Phase 3 — Transformer Deep Dive ✅

Implemented the full Transformer architecture from scratch. Fine-tuned pre-trained BERT.

**Built from scratch:**
- Scaled dot-product attention
- Multi-head attention
- Positional encoding
- Feed-forward blocks
- Full Transformer encoder stack

**Reading:** Attention Is All You Need (Vaswani et al., 2017)

**Key finding:** Pre-trained BERT without fine-tuning scores 50% (random). Fine-tuning is essential.

**Folder:** `03_transformer_deep_dive/`

---

### Phase 4 — LLM Internals and Fine-tuning ✅

Decoder-only architecture, causal masking, autoregressive generation, LoRA, QLoRA, RAG.

**Built:**
- Causal self-attention with triangular mask
- Full GPT-style decoder blocks
- LoRA from scratch (0.24% trainable params, 7.9x memory reduction)
- Fine-tuned GPT-2 on WikiText-2 with LoRA
- FAISS vector store and RAG pipeline

**Reading:** Karpathy's nanoGPT walkthrough, The Unreasonable Effectiveness of RNNs

**Folder:** `04_llm_internals_finetuning/`

---

### Phase 5 — Educational Rewriter GPT ✅

Fine-tuned LLaMA 3.2 3B with QLoRA to rewrite confusing educational content in 6 targeted modes.

**Pipeline:**
1. Collected 141 source passages (Wikipedia + arXiv)
2. Generated 846 rewrite pairs via Claude API ($3.12 total)
3. Fine-tuned LLaMA 3.2 3B with QLoRA (150 minutes, T4 GPU)
4. Built domain-agnostic jargon detection system (Simple English Wikipedia)
5. Published LoRA adapters to HuggingFace Hub

**Results:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.737 | 1.183 |
| 2 | 1.149 | 1.086 |
| 3 | 1.044 | 1.065 |

Train/val gap at epoch 3: 0.021 (minimal overfitting)

**Mode evaluation:**

| Mode | Status |
|------|--------|
| Default | Good |
| Simpler | Partial |
| Add Example | Partial |
| Concise | Good |
| Step by Step | Good |
| Add Analogy | Good |

**Model:** [ray-2908/educational-rewriter-lora](https://huggingface.co/ray-2908/educational-rewriter-lora)

**Folder:** `05_rewriter_gpt/`

---

## Repository Structure

```
nlp-llm-journey/
├── README.md
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
├── 03_transformer_deep_dive/
│   ├── README.md
│   ├── concepts/
│   └── demos/
├── 04_llm_internals_finetuning/
│   ├── README.md
│   ├── concepts/
│   └── demos/
└── 05_rewriter_gpt/
    ├── README.md
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── jargon/
    ├── demos/
    └── results/
```

---

## Key Learnings

### The NLP Evolution

```
Bag of Words     → sparse, no semantics
Word embeddings  → dense, semantic similarity
BERT             → contextual, bidirectional
GPT              → causal, generative
LLMs at scale    → emergent reasoning, in-context learning
```

### Fine-tuning Hierarchy

```
Full fine-tuning   → update all parameters (expensive)
LoRA               → update 0.1-1% of parameters (efficient)
QLoRA              → 4-bit base + LoRA (fits on consumer GPU)
RAG                → no training, retrieve at inference time
Prompting          → no training, no retrieval (fastest)
```

### Consistent Principles

- **Start simple.** The TF-IDF + LogReg baseline beat the LSTM in Phase 2.
- **Measure before optimising.** Every upgrade should be justified by data.
- **Document failures honestly.** The partial modes in Phase 5 have clear root causes and proposed fixes — that is more valuable than hiding failures.
- **Data quality matters more than model size.** The 3B model with 846 examples showed limitations that better training data would partially fix before needing a 7B model.
- **Never hardcode secrets.** Learned from a GitHub secret scanning alert in Phase 5.

---

## Projects and Artefacts

| Project | Description | Link |
|---------|-------------|------|
| Sentiment Analyser | TF-IDF baseline + LSTM + Gradio demo | `02_demo_sentiment_analyser/` |
| Educational Rewriter | Fine-tuned LLaMA 3.2 3B LoRA | [HuggingFace](https://huggingface.co/ray-2908/educational-rewriter-lora) |
| Teaching Quality Analyzer | Full capstone (coming next) | *In development* |

---

## Upcoming

**Teaching Quality Analyzer** — A full educational content quality platform powered by the Phase 5 rewriter model.

- Quality scorer with SHAP explainability
- Audience suitability score (1-10 formula)
- Iterative simplification loop
- Two Streamlit apps deployed on HuggingFace Spaces

**Repository:** [teaching-quality-analyzer](https://github.com/ray-2908/teaching-quality-analyzer) *(coming soon)*

---

## Reading List

| Paper / Book | Phase |
|-------------|-------|
| Speech and Language Processing — Jurafsky & Martin (Ch. 1-6) | Phase 1 |
| Attention Is All You Need — Vaswani et al. (2017) | Phase 3 |
| BERT — Devlin et al. (2018) | Phase 3 |
| LoRA — Hu et al. (2021) | Phase 4 |
| QLoRA — Dettmers et al. (2023) | Phase 4 |
| RAG — Lewis et al. (2020) | Phase 4 |
| Alpaca — Taori et al. (2023) | Phase 5 |

---

## To Explore After Capstone

- **RLHF** (Reinforcement Learning from Human Feedback) — how ChatGPT and Claude are aligned with human preferences
- **Constitutional AI** — Anthropic's alignment approach, what makes Claude different
- **Mixture of Experts** — how GPT-4 and Mixtral scale efficiently
- **Vision-Language Models** — CLIP, GPT-4V, LLaVA

---

## Author

**Anisha Ray**  
[HuggingFace](https://huggingface.co/ray-2908) · [GitHub](https://github.com/ray-2908)

**Timeline:** April 2026 — ongoing

---

*Last updated: May 2026*
