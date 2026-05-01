# LoRA and PEFT — Efficient Fine-tuning

> **LoRA revolutionised fine-tuning by showing you don't need to update all parameters — just a tiny fraction.** The result: fine-tune billion-parameter models on consumer hardware in hours, not days.

---

## Table of Contents

1. [The Problem with Full Fine-tuning](#1-the-problem-with-full-fine-tuning)
2. [What is PEFT?](#2-what-is-peft)
3. [What is LoRA?](#3-what-is-lora)
4. [Why Low-Rank Works](#4-why-low-rank-works)
5. [LoRA in Practice](#5-lora-in-practice)
6. [QLoRA — Even More Efficient](#6-qlora----even-more-efficient)
7. [Choosing the Right Rank](#7-choosing-the-right-rank)
8. [Merging LoRA Weights](#8-merging-lora-weights)
9. [When to Use What](#9-when-to-use-what)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. The Problem with Full Fine-tuning

Fine-tuning all parameters of a large model is extremely expensive:

```
Model           Parameters    Memory (fp32)    Full Fine-tune Memory
─────────────────────────────────────────────────────────────────────
GPT-2 (small)   124M          0.5 GB           ~2 GB
GPT-2 (large)   774M          3 GB             ~12 GB
LLaMA 1B        1B            4 GB             ~16 GB
LLaMA 7B        7B            28 GB            ~112 GB
LLaMA 70B       70B           280 GB           ~1.1 TB
```

Full fine-tune memory = weights + gradients + optimizer states (Adam keeps 2 copies)

**Most people don't have 112GB of GPU memory for LLaMA 7B.**

This is the problem LoRA solves.

---

## 2. What is PEFT?

**PEFT** = **P**arameter-**E**fficient **F**ine-**T**uning

An umbrella term for methods that fine-tune only a small fraction of parameters:

| Method | Approach | Trainable Params | Notes |
|--------|----------|-----------------|-------|
| **LoRA** | Low-rank weight updates | 0.1–1% | Most popular |
| **QLoRA** | LoRA + 4-bit quantisation | 0.1% + less memory | Best for large models |
| **Adapter** | Small layers between blocks | 1–5% | Older approach |
| **Prefix tuning** | Learnable soft prompts | ~0.1% | Task-specific prompts |
| **Prompt tuning** | Tune input embeddings | ~0.01% | Very parameter-efficient |

**LoRA and QLoRA dominate** — best balance of efficiency and performance.

---

## 3. What is LoRA?

**LoRA** = **Lo**w-**R**ank **A**daptation (Hu et al., 2021)

### The Simple Explanation

Imagine a very smart friend who knows everything. Teaching them took 10 years and millions of dollars.

Now you want them to learn one specific new skill — like reviewing movies.

```
Option 1: Retrain from scratch (Full Fine-tuning)
  Retrain everything from scratch with movie reviews mixed in
  ❌ Takes 10 years again
  ❌ Costs millions again

Option 2: Give them a cheat sheet (LoRA)
  Give them a tiny notebook with just the movie-specific stuff
  ✅ Tiny notebook (not 10 years of knowledge)
  ✅ They're now great at movie reviews
  ✅ Everything else still works perfectly
```

**That tiny notebook = LoRA adapters**

### The Math

Full fine-tuning updates weight matrix W directly:

```
W_new = W + ΔW
ΔW shape: (d, k)   e.g. (4096, 4096) = 16M parameters
```

LoRA decomposes the update into two small matrices:

```
W_new = W + A × B

A shape: (d, r)    e.g. (4096, 8) = 32K parameters
B shape: (r, k)    e.g. (8, 4096) = 32K parameters
r = rank (usually 4–64)

Total LoRA params: 64K instead of 16M
That's 256x fewer parameters!
```

### Visually

```
Original weight matrix W (FROZEN — never updated):
┌─────────────────────┐
│                     │
│   W  (4096 × 4096)  │   16M parameters
│                     │
└─────────────────────┘

LoRA adds two tiny matrices (TRAINABLE):
┌───┐     ┌─────────────────────┐
│   │     │                     │
│ A │  ×  │          B          │
│   │     │                     │
└───┘     └─────────────────────┘
4096×8         8×4096
32K params     32K params = 64K total

Output = W·x + (A·B)·x × scaling
       = frozen contribution + LoRA contribution
```

---

## 4. Why Low-Rank Works

### The Intuition

Research showed that the **intrinsic dimensionality** of fine-tuning is low.

```
Pre-trained model: knows grammar, facts, reasoning, writing
Fine-tuning task: learn sentiment classification

What changes during fine-tuning?
  - A small shift in how the model applies its knowledge
  - Not a complete rewrite of its knowledge
  - This shift lives in a low-dimensional subspace

Low rank captures this small shift efficiently.
```

### Evidence

```
LLaMA 7B fine-tuned with LoRA (rank 8):
  Trainable parameters: 0.39% of total
  Performance: ~98% of full fine-tuning

Full fine-tuning: 100% performance, 100% compute cost
LoRA rank 8:       98% performance,   0.4% compute cost

That's 250x more efficient for 2% performance drop.
In practice: often unnoticeable difference.
```

### Parameter Reduction at Different Ranks (d=4096)

```
Rank    LoRA Params    Reduction    % of Full
─────────────────────────────────────────────
1            8,192       2048x       0.049%
2           16,384       1024x       0.098%
4           32,768        512x       0.195%
8           65,536        256x       0.391%  ← Sweet spot
16         131,072        128x       0.781%
32         262,144         64x       1.562%
64         524,288         32x       3.125%
128      1,048,576         16x       6.250%
```

Rank 8 is typically sufficient. Use rank 16–64 for more complex tasks.

---

## 5. LoRA in Practice

### Implementation

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        
        # Original linear layer — FROZEN
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        # LoRA matrices — TRAINABLE
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Scaling: alpha/rank controls magnitude of updates
        self.scaling = alpha / rank
        
        # Initialise: A = random, B = zeros
        # At init: LoRA contribution = 0 (identity)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        original = self.linear(x)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        return original + lora
```

### Why B Initialised to Zeros

```
At start of fine-tuning:
  A × B = A × 0 = 0

So: W_new = W + 0 = W

The model starts identical to pre-trained.
Training gradually adjusts A and B.
This ensures stable training from the start.
```

### Where to Apply LoRA

In Transformers, LoRA is typically applied to attention projections:

```
Applied to:
  - Q projection (query)
  - K projection (key)
  - V projection (value)
  - O projection (output)

Optionally:
  - FFN layers (less common)
  - Embedding layers (rare)

Not applied to:
  - LayerNorm (very few parameters)
  - Biases (very few parameters)
```

### Alpha and Scaling

```
scaling = alpha / rank

Alpha controls the magnitude of the LoRA update.
Common settings:
  alpha = rank      → scaling = 1.0
  alpha = 2 × rank  → scaling = 2.0  (more aggressive updates)
  alpha = 16, rank = 8 → scaling = 2.0

Higher alpha = LoRA has more influence.
Start with alpha = 2 × rank as a rule of thumb.
```

---

## 6. QLoRA — Even More Efficient

### What is Quantisation?

Reduce the precision (number of bits) used to store weights:

```
float32:  32 bits per parameter   (most precise)
float16:  16 bits per parameter   (half precision)
int8:      8 bits per parameter   (quantised)
int4:      4 bits per parameter   (heavily quantised)

LLaMA 7B in float32: 28 GB
LLaMA 7B in float16: 14 GB
LLaMA 7B in int8:     7 GB
LLaMA 7B in int4:   3.5 GB  ← Fits on a laptop GPU!
```

### QLoRA = 4-bit quantisation + LoRA

```
Step 1: Load base model in 4-bit (NF4 format)
  Model memory drops from 28 GB → 3.5 GB (LLaMA 7B)

Step 2: Add LoRA adapters in float16
  Adapters are tiny (0.1% of params)

Step 3: Train only the LoRA adapters
  Gradients flow through 4-bit base → fp16 adapters
  Base model remains frozen and quantised

Result: Fine-tune 7B model on a 6GB GPU!
```

### Performance

```
Full fine-tuning (fp32):  Best performance, most memory
LoRA (fp16):              ~98% performance, 10x less memory
QLoRA (4-bit + LoRA):     ~97% performance, 8x less memory than LoRA
```

QLoRA made fine-tuning accessible to everyone with a consumer GPU.

---

## 7. Choosing the Right Rank

### Rank Selection Guide

| Task complexity | Recommended rank | Parameters |
|----------------|-----------------|------------|
| Simple classification | 4–8 | ~0.1–0.4% |
| Domain adaptation | 8–16 | ~0.4–0.8% |
| Complex reasoning | 16–64 | ~0.8–3.1% |
| Creative writing | 8–32 | ~0.4–1.6% |
| Code generation | 16–64 | ~0.8–3.1% |

### Rule of Thumb

```
Start with rank=8, alpha=16
If performance is insufficient → increase rank
If overfitting → decrease rank or add dropout
If memory is an issue → decrease rank
```

### Rank vs Performance

```
Higher rank:
  ✅ More expressive (can learn more complex patterns)
  ✅ Better performance on complex tasks
  ❌ More parameters (but still tiny vs full FT)
  ❌ Slightly more memory

Lower rank:
  ✅ Fewer parameters
  ✅ Less memory
  ✅ Faster training
  ❌ Less expressive
  ❌ May miss complex patterns
```

---

## 8. Merging LoRA Weights

After fine-tuning, you can merge LoRA weights back into the base model:

```python
def merge_lora_weights(layer):
    with torch.no_grad():
        # Compute the LoRA update
        lora_update = (layer.lora_B.weight @ layer.lora_A.weight) * layer.scaling
        
        # Add to original weights
        layer.linear.weight.data += lora_update
```

### Why Merge?

```
During training:
  output = W·x + (A·B)·x × scaling
  Two separate computations (slightly slower inference)

After merging:
  W_merged = W + A×B × scaling
  output = W_merged·x
  Single computation (same speed as original model)
```

### Benefits of Merging

```
✅ Zero inference overhead
✅ Same model size as original
✅ No dependency on LoRA code at inference
✅ Can be deployed like any normal model
```

### Multiple Tasks

```
Base model (frozen) + Task A adapter
Base model (frozen) + Task B adapter
Base model (frozen) + Task C adapter

Switch adapters at inference time!
No need to keep separate full models.
Storage cost: base model + tiny adapters per task.
```

---

## 9. When to Use What

```
Situation                           Recommendation
─────────────────────────────────────────────────────────────────
GPU < 8GB, model > 7B               QLoRA (4-bit + LoRA)
GPU 8-16GB, model 1-7B              LoRA (fp16)
GPU > 40GB, model < 13B             Full fine-tuning
Multiple tasks, one model           LoRA (separate adapters)
Production deployment               Merged LoRA or full FT
Research / maximum performance      Full fine-tuning
Quick experiment / prototyping      LoRA (rank 8)
```

### Practical Defaults

```python
# Good starting config for most tasks
lora_config = {
    "rank": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj"],  # Apply to Q and V
    "lora_dropout": 0.05,
    "bias": "none",
}
```

---

## 10. How This Connects to LLMs

### The Modern Fine-tuning Pipeline

```
Step 1: Download pre-trained LLM
  LLaMA 3.2 1B, GPT-2, Mistral 7B, etc.
  Pre-trained on trillions of tokens

Step 2: Prepare your dataset
  Format as instruction-response pairs
  "Question: ...\nAnswer: ..."

Step 3: Apply LoRA
  Add A and B matrices to attention layers
  Freeze everything else

Step 4: Fine-tune
  Training objective: next token prediction (same as pre-training)
  On your task-specific data

Step 5: Evaluate
  Measure task-specific metrics
  Compare to baseline

Step 6: Merge and deploy
  Merge LoRA weights into base
  Deploy like any model
```

### Why This is Employable

LoRA and PEFT are the dominant approach in industry:

```
Use cases:
  - Fine-tune GPT-2 for your domain
  - Specialise LLaMA for customer support
  - Adapt Mistral for medical Q&A
  - Train coding assistant on your codebase

All with:
  - Consumer GPU (8-24GB)
  - Hours of training (not days/weeks)
  - Minimal cost (personal hardware or cheap cloud)
```

### The Ecosystem

```
Hugging Face PEFT library:
  from peft import LoraConfig, get_peft_model
  
  config = LoraConfig(r=8, lora_alpha=16, ...)
  model = get_peft_model(base_model, config)
  
  model.print_trainable_parameters()
  # trainable params: 0.4% || all params: 100%

Works with:
  - Any Hugging Face model
  - Any PEFT method (LoRA, QLoRA, adapters)
  - Any training framework (Trainer, custom loop)
```

---

## Summary

**LoRA changed fine-tuning forever:**

1. **Freeze base model** — preserves pre-trained knowledge
2. **Add A and B matrices** — tiny trainable update
3. **Train only A and B** — 0.1-1% of parameters
4. **Merge at inference** — zero overhead

The math is simple: `W_new = W + A × B`

The impact is massive: fine-tune billion-parameter models on consumer hardware.

**QLoRA goes further:** quantise the base model to 4-bit, keeping LoRA adapters in fp16. LLaMA 7B fits in 3.5GB — a gaming laptop can fine-tune it.

This is why LoRA and PEFT are among the most employable skills in modern AI engineering.

---

*Phase 4 — concept 02 → you are here*  
*Previous concept → [01 — Decoder Architecture](./01_decoder_architecture.md)*  
*Next → Fine-tune GPT-2 on custom data using LoRA*  
*Then → RAG — Retrieval Augmented Generation*
