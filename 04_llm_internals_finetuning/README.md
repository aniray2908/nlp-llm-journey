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
| 04 | RAG | Learn + Build | ⏳ |

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

### Key Visualisations

- Causal mask heatmap (lower triangular, future blocked)
- BERT vs GPT attention comparison (full matrix vs triangular)
- Autoregressive generation step by step

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

**Why low-rank works:**
- Fine-tuning updates are naturally low-rank
- Model already knows language
- Adaptation is a small shift, not a full rewrite
- Rank 8-64 captures most useful adaptation

**Results from demo:**

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

**The model adopted WikiText's encyclopedic style** — formal, factual, Wikipedia-like. This is fine-tuning working as intended.

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
   Or all 124M parameters remain trainable (defeats the purpose)
```

### Files

- `concepts/03_finetune_gpt2_lora.md`
- `demos/03_finetune_gpt2_lora.ipynb`

---

## Part 4 — RAG ⏳

### Coming Next

**RAG** = Retrieval Augmented Generation

An alternative to fine-tuning:

```
Fine-tuning:
  Bake knowledge into model weights
  Fast inference, but needs retraining when knowledge changes

RAG:
  Retrieve relevant documents at inference time
  No retraining needed, always up-to-date
  More flexible for dynamic knowledge bases
```

**What you'll build:**
- Vector store (FAISS or ChromaDB)
- Embedding-based retrieval
- Simple RAG pipeline end-to-end

---

## Key Learnings

### 1. Decoders are just encoders with a mask

```
BERT encoder block:  Multi-head attention (no mask)
GPT decoder block:   Multi-head attention (causal mask)

That triangular matrix is the entire difference.
```

### 2. Autoregressive generation is simple

```
Predict next token → append → repeat
Same mechanism from GPT-2 to GPT-4 to Claude
Scale and data changed, not the mechanism
```

### 3. LoRA is remarkable

```
Train 0.24% of parameters
Achieve comparable performance to full fine-tuning
7.9x memory reduction
Same code scales from GPT-2 to LLaMA 70B
```

### 4. Fine-tuning adapts style AND knowledge

```
WikiText fine-tuning → encyclopedic style
Medical data fine-tuning → clinical language
Code fine-tuning → your codebase patterns
Instruction data fine-tuning → chat assistant
```

### 5. Production is the same code, bigger model

```
# This demo:
model_name = "gpt2"   # 124M params

# Production:
model_name = "meta-llama/Llama-3.2-1B"  # 1B params
model_name = "mistralai/Mistral-7B"      # 7B params
+ add QLoRA for memory efficiency
```

---

## Folder Structure

```
04_llm_internals_finetuning/
├── README.md                          (this file)
├── concepts/
│   ├── 01_decoder_architecture.md
│   ├── 02_lora_peft.md
│   └── 03_finetune_gpt2_lora.md
└── demos/
    ├── 01_decoder_architecture.ipynb
    ├── 02_lora_peft.ipynb
    └── 03_finetune_gpt2_lora.ipynb
```

---

## Reading List

- **Karpathy's nanoGPT walkthrough** ✅ — https://www.youtube.com/watch?v=kCc8FmEb1nY
- **The Unreasonable Effectiveness of RNNs** ✅ — http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- **LoRA paper** — https://arxiv.org/abs/2106.09685 (Hu et al., 2021)
- **QLoRA paper** — https://arxiv.org/abs/2305.14314 (Dettmers et al., 2023)

---

## Resources

- **Hugging Face PEFT** — https://github.com/huggingface/peft
- **Hugging Face Transformers** — https://github.com/huggingface/transformers
- **nanoGPT** — https://github.com/karpathy/nanoGPT
- **LLaMA models** — https://huggingface.co/meta-llama

---

## Progress

```
Phase 0: PyTorch Fundamentals        ✅ COMPLETE
Phase 1: NLP Fundamentals            ✅ COMPLETE
Phase 2: Sentiment Analyser          ✅ COMPLETE
Phase 3: Transformers Deep Dive      ✅ COMPLETE
Phase 4: LLM Internals + Fine-tuning 🌱 IN PROGRESS
  Part 1: Decoder architecture       ✅ DONE
  Part 2: LoRA and PEFT              ✅ DONE
  Part 3: Fine-tune GPT-2 with LoRA  ✅ DONE
  Part 4: RAG                        ⏳ NEXT
Phase 5: Mini-GPT Capstone           ⏳ FUTURE
```

**You're at ~55% of the full roadmap.**

---

*Phase 4 — LLM Internals + Fine-tuning → in progress*  
*Previous → Phase 3: Transformer Deep Dive*  
*Next → Part 4: RAG — Retrieval Augmented Generation*
