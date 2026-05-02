# Phase 4 — LLM Internals + Fine-tuning

> **From using Transformers to understanding them under the hood.** This phase covers decoder-only models, efficient fine-tuning with LoRA, and RAG — the three most employable skills in modern AI engineering.

---

## Why This Phase Matters

Phase 3 taught you how Transformers work (attention, encoding, BERT).

Phase 4 teaches you how **modern LLMs work** — GPT, LLaMA, Claude — and how to adapt them to your needs efficiently.

```
Phase 3: How do Transformers understand language? (BERT, encoders)
Phase 4: How do LLMs generate language? (GPT, decoders, LoRA, RAG)
```

LoRA and RAG are the two most in-demand skills in AI engineering right now. Companies need engineers who can fine-tune and deploy LLMs — not just call APIs.

---

## What's in This Phase

| # | Topic | Type | Status |
|---|-------|------|--------|
| 01 | Decoder-only architecture | Learn + Build | ✅ |
| 02 | LoRA and PEFT | Learn + Build | ✅ |
| 03 | Fine-tune GPT-2 with LoRA | Build | ✅ |
| 04 | RAG pipeline | Learn + Build | ✅ |

---

## Part 1 — Decoder Architecture ✅

### What You Learned

The decoder-only architecture powers every modern LLM:

```
BERT (encoder):   Bidirectional → understands text
GPT (decoder):    Causal → generates text

One difference: a triangular mask in attention
That one mask changes everything.
```

**Causal masking:**
- Lower triangular matrix blocks future tokens
- "The cat ___" can see "The cat" but NOT "on the mat"
- Enforces left-to-right generation

**Autoregressive generation:**
- Generate one token at a time
- Feed output back as input
- Stop at end token or max length
- This is how every LLM generates text

**Training objective:**
- Predict next token at every position
- One sentence = N training signals
- Scale to trillions of tokens = massive learning signal

**Sampling strategies:**
- Greedy: always pick highest probability
- Temperature: scale diversity
- Top-k: sample from top k tokens
- Top-p (nucleus): most modern LLMs use this

### Files

- `concepts/01_decoder_architecture.md`
- `demos/01_decoder_architecture.ipynb`

---

## Part 2 — LoRA and PEFT ✅

### What You Learned

Fine-tuning large models is expensive. LoRA makes it cheap.

**The problem:**
```
LLaMA 7B full fine-tune = 112 GB GPU memory
Most people don't have that.
```

**The solution (LoRA):**
```
Freeze original weights W (never updated)
Add two tiny matrices A and B (trainable)
W_new = W + A × B × scaling

Rank 8: 0.39% of parameters
256x fewer parameters than full fine-tuning
Nearly identical performance
```

**Results from demo (d=4096):**

```
Rank    LoRA Params    Reduction    % of Full
─────────────────────────────────────────────
1            8,192       2048x       0.049%
4           32,768        512x       0.195%
8           65,536        256x       0.391%  ← Sweet spot
16         131,072        128x       0.781%
64         524,288         32x       3.125%
```

**QLoRA — even better:**
- Quantise base model to 4-bit (28 GB → 3.5 GB for 7B)
- Add LoRA adapters in fp16
- Fine-tune 7B model on a laptop GPU

### Files

- `concepts/02_lora_peft.md`
- `demos/02_lora_peft.ipynb`

---

## Part 3 — Fine-tune GPT-2 with LoRA ✅

### What You Built

End-to-end fine-tuning pipeline:

1. Load pre-trained GPT-2 (124M parameters)
2. Apply LoRA to attention layers (rank=8)
3. Freeze 99.8% of parameters
4. Fine-tune on WikiText-2
5. Generate text with fine-tuned model

### Results

```
GPT-2 + LoRA (rank=8):
  Total parameters:     124,734,720
  Frozen parameters:    124,439,808 (99.8%)
  Trainable parameters:     294,912 (0.24%)
  Memory reduction:              7.9x

Training (3 epochs):
  Train loss: 1.20 | Perplexity: 3.3
  Val loss:   3.44 | Perplexity: 31.3
```

### Generated Text (After Fine-tuning on WikiText)

```
Prompt: "The history of artificial intelligence"
Output: "...ones that are considered superior to humans...
         theories based on natural selection and hypothesis testing..."

Prompt: "In the field of science"
Output: "...the term 'solar system' has applied broadly to any part
         of the solar system outside of its current form..."

Prompt: "The most important discovery"
Output: "...made during the month of Nahant, when the Egyptians
         closed their temples and began to rebuild..."
```

The model adopted WikiText's encyclopedic style — formal, factual, Wikipedia-like. This is fine-tuning working as intended.

### GPT-2 Quirks Discovered

```
1. Conv1D instead of Linear
   GPT-2 uses Conv1D for attention projections
   Weight shape is transposed vs nn.Linear
   Fix: read weight.shape[0] and weight.shape[1] directly

2. Combined QKV projection
   GPT-2 combines Q, K, V into one layer (c_attn)
   LoRA applied to c_attn covers all three projections

3. Freezing parameters
   Must explicitly freeze all non-LoRA params
   Or all 124M parameters remain trainable
```

### Files

- `concepts/03_finetune_gpt2_lora.md`
- `demos/03_finetune_gpt2_lora.ipynb`

---

## Part 4 — RAG Pipeline ✅

### What You Learned

RAG solves the two biggest problems with LLMs:

```
Problem 1: Knowledge cutoff
  LLMs don't know about events after training
  RAG: retrieve up-to-date docs at inference time

Problem 2: Private knowledge
  LLMs don't know your company's docs
  RAG: index your docs in a vector store
```

**Three steps:**

```
1. INDEX (one-time):
   Documents → embed → store in vector database

2. RETRIEVE (every query):
   Query → embed → find similar docs → top-k

3. GENERATE (every query):
   Query + retrieved docs → prompt → LLM → answer
```

**Why semantic search works:**

```
Traditional search: "return" finds "return" only
Semantic search:    "Can I return this?" finds "refund policy"
                    Because embeddings capture meaning, not keywords
```

### Results from Demo

**Similarity search** (query: "How does attention work in Transformers?"):

```
Rank 1 (sim=0.76): "Transformers are neural networks that use attention..."
Rank 2 (sim=0.53): "Attention mechanisms compute relevance scores..."
Rank 3 (sim=0.28): "Fine-tuning adapts a pre-trained model..."

Retrieval correctly ranked documents by semantic relevance.
```

**RAG vs No RAG** (query: "What is LoRA and how does it reduce memory?"):

```
Without RAG: "LoRA is a method of computing bytes in memory..."
             ❌ Hallucinated — GPT-2 doesn't know ML's LoRA

With RAG:    Retrieved correct LoRA document
             ⚠️ GPT-2 still struggled to use context
             → Retrieval worked, GPT-2 too small for generation
             → Production RAG needs GPT-4 / Claude / LLaMA 7B+
```

**Key lesson:** Retrieval quality depends on embedding model. Generation quality depends on LLM size.

### RAG vs Fine-tuning

| Aspect | Fine-tuning | RAG |
|--------|-------------|-----|
| **Knowledge update** | Retrain (expensive) | Update vector store (cheap) |
| **Latency** | Fast | Slightly slower |
| **Hallucination** | Can still occur | Less (grounded in docs) |
| **Transparency** | Black box | Can cite sources |
| **Best for** | Style, reasoning | Facts, private docs |

**Best practice: use both together.**

### Files

- `concepts/04_rag.md`
- `demos/04_rag_pipeline.ipynb`

---

## Key Learnings

### 1. Decoders are just encoders with a mask

```
BERT encoder block:  Multi-head attention (no mask)
GPT decoder block:   Multi-head attention (causal mask)

That triangular matrix is the entire difference.
```

### 2. LoRA is remarkable

```
Train 0.24% of parameters
Achieve comparable performance to full fine-tuning
7.9x memory reduction on GPT-2
Same code scales from GPT-2 to LLaMA 70B
```

### 3. Fine-tuning adapts style AND knowledge

```
WikiText fine-tuning → encyclopedic, formal style
Medical data → clinical language
Instruction data → chat assistant behaviour
```

### 4. RAG retrieval is semantic, not keyword-based

```
"Can I return this?" → finds "refund policy"
Embeddings capture meaning
Cosine similarity finds semantic neighbours
```

### 5. Generation quality depends on LLM size

```
GPT-2 (124M): Poor instruction following, ignores context
LLaMA 7B+:    Good instruction following, uses context well
GPT-4/Claude: Excellent, production-ready
```

### 6. RLHF and Constitutional AI — Note to Explore Later

```
RLHF: Reinforcement Learning from Human Feedback
  How LLMs like ChatGPT are aligned with human preferences

Constitutional AI: Anthropic's alignment approach
  Trains Claude to evaluate its own outputs against principles
  What makes Claude different from other LLMs

Both are post-Phase 5 rabbit holes worth exploring.
```

---

## Folder Structure

```
04_llm_internals_finetuning/
├── README.md                          (this file)
├── concepts/
│   ├── 01_decoder_architecture.md
│   ├── 02_lora_peft.md
│   ├── 03_finetune_gpt2_lora.md
│   └── 04_rag.md
└── demos/
    ├── 01_decoder_architecture.ipynb
    ├── 02_lora_peft.ipynb
    ├── 03_finetune_gpt2_lora.ipynb
    └── 04_rag_pipeline.ipynb
```

---

## Reading List

- **Karpathy's nanoGPT walkthrough** ✅ — https://www.youtube.com/watch?v=kCc8FmEb1nY
- **The Unreasonable Effectiveness of RNNs** ✅ — http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- **LoRA paper** — https://arxiv.org/abs/2106.09685 (Hu et al., 2021)
- **QLoRA paper** — https://arxiv.org/abs/2305.14314 (Dettmers et al., 2023)
- **RAG paper** — https://arxiv.org/abs/2005.11401 (Lewis et al., 2020)

---

## Resources

- **Hugging Face PEFT** — https://github.com/huggingface/peft
- **FAISS** — https://github.com/facebookresearch/faiss
- **ChromaDB** — https://www.trychroma.com/
- **LangChain** — https://www.langchain.com/
- **LlamaIndex** — https://www.llamaindex.ai/

---

## Progress

```
Phase 0: PyTorch Fundamentals        ✅ COMPLETE
Phase 1: NLP Fundamentals            ✅ COMPLETE
Phase 2: Sentiment Analyser          ✅ COMPLETE
Phase 3: Transformers Deep Dive      ✅ COMPLETE
Phase 4: LLM Internals + Fine-tuning ✅ COMPLETE
  Part 1: Decoder architecture       ✅
  Part 2: LoRA and PEFT              ✅
  Part 3: Fine-tune GPT-2 with LoRA  ✅
  Part 4: RAG pipeline               ✅
Phase 5: Mini-GPT Capstone           ⏳ NEXT
```

**You're at ~75% of the full roadmap. 🚀**

---

*Phase 4 — LLM Internals + Fine-tuning → Complete ✅*  
*Previous → Phase 3: Transformer Deep Dive*  
*Next → Phase 5: Mini-GPT Capstone*
