# NLP → LLM Learning Journey

> A structured, hands-on path from classical NLP to building language models. 5 phases, 1 portfolio project, everything from tokenisation to training a mini-GPT from scratch.

---

## Status

```
Phase 0: PyTorch fundamentals              ✅ COMPLETE
Phase 1: NLP fundamentals                  ✅ COMPLETE
Phase 2: Demo project — sentiment analyser ✅ COMPLETE
Phase 3: Transformer deep dive             ✅ COMPLETE
Phase 4: LLM internals + fine-tuning       ✅ COMPLETE
Phase 5: Capstone — mini GPT               🌱 NEXT
```

**Progress: ~32/39 tasks complete (~75%)**

---

## What This Is

A **learning roadmap and project repository** for understanding language models from first principles. Not a course or tutorial — a self-directed journey with:

- **6 phases** covering tokenisation → Transformers → LLMs
- **Concept notes** (markdown) explaining every idea
- **Demo notebooks** with hands-on code
- **3 portfolio projects** you can ship
- **No fixed timeline** — go at your own pace

---

## The Journey

### Phase 0 — PyTorch Warm-up ✅

**6 concepts, 6 demos**

- Tensors, slicing, broadcasting, GPU
- Autograd and backpropagation
- `nn.Module` and custom layers
- Training loops, DataLoaders, optimisers

**Projects:** Linear regression, MLPs, MNIST classifier (98% accuracy)

**Folder:** `00_pytorch_warmup/`

---

### Phase 1 — NLP Fundamentals ✅

**6 concepts, 6 demos, reading**

- Tokenisation — word, subword (BPE), character-level
- Text normalisation — stemming, lemmatisation, stopwords
- Bag of Words & TF-IDF — sparse vector representations
- Word embeddings — Word2Vec, GloVe, semantic space
- Language modelling — n-grams, perplexity, autoregressive generation
- Text classification — complete pipeline, evaluation metrics

**Reading:** "Speech and Language Processing" (Jurafsky & Martin) Ch. 1–6

**Folder:** `01_nlp_fundamentals/`

---

### Phase 2 — Sentiment Analyser Capstone ✅

**First portfolio piece**

- **Baseline:** TF-IDF + Logistic Regression (82% accuracy)
- **Upgrade:** GloVe + LSTM (67% accuracy)
- **Result:** Baseline wins — complex ≠ better

**Deliverables:** Training notebook, README, inference script, Gradio demo

**Key learning:** Start simple. Measure. Upgrade only if needed.

**Folder:** `02_demo_sentiment_analyser/`

---

### Phase 3 — Transformer Deep Dive ✅

**3 parts, 2 concepts, 2 demos, reading**

- **Part 1:** Implemented attention from scratch (scaled dot-product, multi-head, positional encoding, full encoder)
- **Part 2:** Read "Attention Is All You Need" (Vaswani et al., 2017)
- **Part 3:** Pre-training vs fine-tuning, transfer learning, BERT evaluation

**Key insight:** BERT without fine-tuning = 50% accuracy (random). Pre-training + fine-tuning unlocks the power.

**Folder:** `03_transformer_deep_dive/`

---

### Phase 4 — LLM Internals + Fine-tuning ✅

**4 parts, 4 concepts, 4 demos, reading**

#### Part 1: Decoder Architecture

- Causal masking (triangular matrix, blocks future tokens)
- Autoregressive generation (one token at a time, feed back as input)
- Training objective (predict next token, N signals per sentence)
- Sampling strategies (greedy, temperature, top-k, top-p)

#### Part 2: LoRA and PEFT

- Low-rank adaptation — freeze weights, train tiny A×B matrices
- 0.39% of parameters at rank 8 (256x reduction)
- QLoRA — 4-bit quantisation + LoRA, fine-tune 7B on a laptop

#### Part 3: Fine-tune GPT-2 with LoRA

- Applied LoRA to GPT-2 attention layers (rank=8)
- 0.24% trainable parameters, 7.9x memory reduction
- Fine-tuned on WikiText-2, model adopted encyclopedic style
- Generated coherent Wikipedia-style text

#### Part 4: RAG Pipeline

- Built knowledge base (18 NLP/ML documents)
- Embedded with sentence-transformers (384-dim vectors)
- FAISS vector store for similarity search
- Retrieval correctly ranked documents by semantic similarity
- RAG vs No RAG comparison (retrieval worked, GPT-2 too small for generation)

**Reading:** Karpathy's nanoGPT walkthrough + Unreasonable Effectiveness of RNNs

**Folder:** `04_llm_internals_finetuning/`

---

### Phase 5 — Capstone: Mini-GPT 🌱

**The project that sets you apart**

Build a character or word-level GPT from scratch:
- Implement tokeniser, embeddings, positional encoding
- Build full decoder-only Transformer stack
- Write training loop with gradient clipping and checkpointing
- Train on Shakespeare, code, or custom dataset
- Track loss curves, generate samples, visualise attention
- Write detailed README explaining every decision

This project demonstrates deep understanding of language models.

**Folder:** `05_capstone_mini_gpt/` (coming next)

---

## Key Learnings Across Phases

### The Evolution of NLP

```
BoW (Phase 1):         Sparse counts, no semantics
Embeddings (Phase 1):  Dense vectors, semantic similarity
BERT (Phase 3):        Contextual embeddings, bidirectional
GPT (Phase 4):         Causal generation, autoregressive
LLMs (Phase 4+):       Scale + alignment = Claude, GPT-4
```

### The Fine-tuning Hierarchy

```
Full fine-tuning:  Update all parameters (expensive)
LoRA:              Update 0.1-1% of parameters (efficient)
QLoRA:             4-bit base + LoRA (fits on laptop)
RAG:               No training, retrieve at inference time
Prompting:         No training, no retrieval (fastest)
```

### Start Simple, Always

```
Phase 2: TF-IDF + LogReg beats LSTM (82% vs 67%)
Phase 3: BERT needs fine-tuning (50% without it)
Phase 4: GPT-2 too small for instruction following
Lesson:  Measure first, upgrade only when needed
```

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
└── 05_capstone_mini_gpt/          (coming next)
```

---

## Time Estimates

| Phase | Topics | Time | Status |
|-------|--------|------|--------|
| 0 | PyTorch | 4–5 days | ✅ |
| 1 | NLP fundamentals | 5–7 days | ✅ |
| 2 | Sentiment analyser | 3–4 days | ✅ |
| 3 | Transformers | 5–7 days | ✅ |
| 4 | LLM internals | 4–5 days | ✅ |
| 5 | Mini-GPT capstone | 5–7 days | 🌱 |
| **Total** | | **26–35 days** | |

---

## Things to Explore After the Capstone

```
RLHF (Reinforcement Learning from Human Feedback)
  How ChatGPT and Claude are aligned with human preferences
  Reward models, PPO, human feedback loops

Constitutional AI (Anthropic)
  How Claude is trained differently from other LLMs
  Self-critique and revision using a set of principles
  What makes Claude helpful, harmless, honest

Mixture of Experts (MoE)
  How GPT-4 and Mixtral scale efficiently
  Sparse activation — not all parameters used per token

Vision-Language Models
  CLIP, GPT-4V, LLaVA
  How models process images + text together
```

---

## Resources

### Papers

- **"Attention Is All You Need"** — https://arxiv.org/abs/1706.03762
- **"BERT"** — https://arxiv.org/abs/1810.04805
- **"LoRA"** — https://arxiv.org/abs/2106.09685
- **"QLoRA"** — https://arxiv.org/abs/2305.14314
- **"RAG"** — https://arxiv.org/abs/2005.11401

### Videos

- **Karpathy's "Neural Networks: Zero to Hero"** — YouTube
- **Karpathy's nanoGPT walkthrough** — YouTube

### Books

- **"Speech and Language Processing"** (Jurafsky & Martin) — free online

### Libraries

- **Hugging Face Transformers** — https://huggingface.co/
- **PEFT** — https://github.com/huggingface/peft
- **FAISS** — https://github.com/facebookresearch/faiss
- **LangChain** — https://www.langchain.com/

---

## Goals

By the end, you'll have:

✓ Deep understanding of how LLMs work (not just API usage)  
✓ Implemented attention, fine-tuned BERT, trained a GPT  
✓ Built LoRA fine-tuning and RAG pipelines from scratch  
✓ Three portfolio projects you can show employers  
✓ Skills to read and understand research papers  
✓ Foundation to explore RLHF, Constitutional AI, MoE  

---

## Author

Built as a self-directed learning project.

**Timeline:** Started April 2026, ongoing.

---

*Last updated: May 2026*

---

> Understanding language models deeply requires the full journey: from tokenisation to Transformers. Skip the foundation, and you'll hit ceilings you can't explain. Build it all from scratch, and you can build anything.

🚀
