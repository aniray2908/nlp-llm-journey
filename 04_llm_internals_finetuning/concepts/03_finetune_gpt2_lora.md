# Fine-tuning GPT-2 with LoRA — From Pre-trained to Domain-Adapted

> **Fine-tuning a language model on custom data is one of the most powerful and employable skills in modern AI.** With LoRA, you can do it on consumer hardware in minutes.

---

## Table of Contents

1. [Why Fine-tune?](#1-why-fine-tune)
2. [The Setup](#2-the-setup)
3. [GPT-2 Architecture Quirks](#3-gpt-2-architecture-quirks)
4. [Applying LoRA to GPT-2](#4-applying-lora-to-gpt2)
5. [The Training Objective](#5-the-training-objective)
6. [Results](#6-results)
7. [Understanding the Outputs](#7-understanding-the-outputs)
8. [Common Issues and Fixes](#8-common-issues-and-fixes)
9. [Production Workflow](#9-production-workflow)
10. [What's Next](#10-whats-next)

---

## 1. Why Fine-tune?

### Pre-trained GPT-2 Knows Everything General

```
GPT-2 trained on:
  - Web pages (Reddit links, news, blogs)
  - Books
  - General English text

GPT-2 knows:
  - Grammar
  - General facts
  - Writing styles
  - Common reasoning patterns
```

### But Sometimes You Need Domain-Specific

```
Use case: Customer support bot for your company
  GPT-2 knows English but not your product
  Fine-tune on your support tickets → learns your domain

Use case: Medical Q&A
  GPT-2 knows general facts but not clinical language
  Fine-tune on PubMed abstracts → learns medical style

Use case: Code generation
  GPT-2 knows some code but not your codebase
  Fine-tune on your repo → learns your patterns
```

Fine-tuning = **transfer GPT-2's language knowledge + teach it your domain.**

---

## 2. The Setup

### What We Used

```
Base model:     GPT-2 (124M parameters)
Dataset:        WikiText-2 (Wikipedia text)
Fine-tuning:    LoRA (rank=8, alpha=16)
Epochs:         3
Learning rate:  3e-4 (higher than BERT — LoRA has few params)
Batch size:     4
Max length:     128 tokens
```

### Why GPT-2 for This Demo

```
✅ Small enough to run on CPU/laptop (124M params)
✅ Well-understood architecture
✅ Good pre-trained knowledge
✅ Same decoder-only architecture as GPT-3, LLaMA, Claude
✅ Production workflow is identical (just bigger model)
```

Same code works for LLaMA 7B — just change the model name and add QLoRA.

---

## 3. GPT-2 Architecture Quirks

### Conv1D Instead of Linear

GPT-2 uses `Conv1D` layers for attention projections instead of `nn.Linear`:

```
Standard Transformer:  nn.Linear(in, out)
  weight shape: (out, in)

GPT-2:                 Conv1D(in, out)
  weight shape: (in, out)   ← transposed!

Why? Historical implementation detail.
Same math, different storage convention.
```

This matters when applying LoRA — you need to read the weight shape correctly:

```python
# nn.Linear
in_features = layer.in_features
out_features = layer.out_features

# Conv1D (GPT-2)
in_features = layer.weight.shape[0]
out_features = layer.weight.shape[1]
```

### Combined QKV Projection

GPT-2 combines Q, K, V into one layer (`c_attn`):

```
Standard attention:
  Q = W_q * x   (separate layer)
  K = W_k * x   (separate layer)
  V = W_v * x   (separate layer)

GPT-2:
  QKV = c_attn * x   (one layer, output split 3 ways)
  Q, K, V = QKV.split(d_model, dim=2)
```

We apply LoRA to `c_attn` which covers all three projections at once.

---

## 4. Applying LoRA to GPT-2

### The Key Step

```python
# For each transformer block in GPT-2:
for block in model.transformer.h:
    original_layer = block.attn.c_attn  # Combined QKV projection
    lora_layer = LoRAConv1D(original_layer, rank=8, alpha=16)
    block.attn.c_attn = lora_layer      # Replace with LoRA version
```

### Freezing Parameters

Critical step: freeze everything except LoRA matrices.

```python
def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
```

Without this, all parameters would train (defeating the purpose of LoRA).

### Results of Applying LoRA

```
GPT-2 (124M parameters) + LoRA (rank=8):
  Total parameters:     124,734,720
  Frozen parameters:    124,439,808 (99.8%)
  Trainable parameters:     294,912 (0.24%)

Memory reduction: ~7.9x vs full fine-tuning
```

**0.24% of parameters trained.** The rest are frozen.

---

## 5. The Training Objective

### Language Modelling (Same as Pre-training)

Fine-tuning uses the same objective as pre-training — **predict the next token:**

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=input_ids  # Labels = input (model predicts next token at each position)
)
loss = outputs.loss  # Cross-entropy averaged over all positions
```

GPT-2 internally shifts labels by 1:

```
Input:  ["The", "cat", "sat", "on",  "the"]
Target: ["cat", "sat", "on",  "the", "mat"]

Loss computed at every position:
  "The" → predict "cat"  → loss₁
  "cat" → predict "sat"  → loss₂
  ...

Total loss = average of all position losses
```

### Learning Rate for LoRA

```
Full fine-tuning:  lr = 2e-5 (small — updating many params)
LoRA fine-tuning:  lr = 3e-4 (larger — updating few params)

Why higher for LoRA?
  Fewer parameters = each update needs to do more work
  Higher LR helps LoRA converge faster
```

---

## 6. Results

### Training Curves

```
Epoch 1:  Train loss: 1.22, Perplexity: 3.4
Epoch 2:  Train loss: 1.22, Perplexity: 3.4
Epoch 3:  Train loss: 1.20, Perplexity: 3.3

Val loss:  3.44, Val perplexity: 31.3
```

### Interpreting the Results

```
Train perplexity: 3.3
  Model is very confident on training data
  "Only 3.3 equally likely words at each position"
  → Model learned the training distribution well

Val perplexity: 31.3
  Model less confident on unseen text
  "31.3 equally likely words at each position"
  → Some overfitting on the small dataset

Gap (train vs val):
  Train: 3.3, Val: 31.3 → gap of ~28
  Indicates overfitting due to small dataset (400 training examples)
  On larger datasets: gap narrows significantly
```

### Parameter Efficiency

```
Trainable: 0.24% of parameters
Memory:    7.9x reduction vs full fine-tuning
Training:  3 epochs on CPU in minutes
Result:    Model learns WikiText style
```

---

## 7. Understanding the Outputs

### Generated Text

After fine-tuning on WikiText-2, GPT-2 generates Wikipedia-style text:

```
Prompt: "The history of artificial intelligence"
Output: "...ones that are considered superior to humans...
         theories based on natural selection and hypothesis testing..."

Prompt: "In the field of science"
Output: "...the term 'solar system' has applied broadly...
         The term comes from the Latin sí, meaning 'within'..."

Prompt: "The most important discovery"
Output: "...made during the month of Nahant, when the Egyptians
         closed their temples and began to rebuild..."
```

### Why It Sounds Like Wikipedia

Fine-tuning on WikiText teaches:
- **Formal, encyclopedic tone** — no casual language
- **Latin etymology** — Wikipedia often traces word origins
- **Historical narrative** — structured chronological writing
- **Hedging language** — "is said to", "is considered"

**The model adopted the dataset's style.** This is exactly what fine-tuning does.

---

## 8. Common Issues and Fixes

### Issue 1: Conv1D vs Linear

```
Error: AttributeError: 'Conv1D' object has no attribute 'in_features'

Fix: Read weight shape directly
  in_features = layer.weight.shape[0]
  out_features = layer.weight.shape[1]
```

### Issue 2: LoRA Not Freezing Correctly

```
Symptom: Trainable params = 83% (should be 0.2%)

Fix: Explicitly freeze all non-LoRA params
  for name, param in model.named_parameters():
      if 'lora_A' not in name and 'lora_B' not in name:
          param.requires_grad = False
```

### Issue 3: Train Loss Much Lower Than Val Loss

```
Symptom: Train perplexity 3.3, Val perplexity 31.3

Cause: Small dataset (400 examples)
Fix:
  - Use more training data
  - Add dropout to LoRA
  - Reduce epochs (early stopping)
  - Use larger dataset
```

### Issue 4: Model Not Learning

```
Symptom: Loss not decreasing

Possible causes:
  - Learning rate too low (try 1e-3)
  - LoRA not applied correctly
  - Batch size too small
  - Gradient not flowing to LoRA params
```

---

## 9. Production Workflow

### Same Code, Bigger Model

Everything in this demo scales directly to production:

```python
# This demo:
model_name = "gpt2"          # 124M params

# Production:
model_name = "meta-llama/Llama-3.2-1B"   # 1B params
model_name = "mistralai/Mistral-7B-v0.1" # 7B params

# Add QLoRA for large models:
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
```

### Using Hugging Face PEFT (Production Standard)

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=16,                # Scaling
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 0.24% || all params: 100%
```

### Instruction Fine-tuning Format

For chat/instruction models, format data as:

```
### Instruction:
Summarize this article in 3 bullet points.

### Input:
[Article text here]

### Response:
- Point 1
- Point 2
- Point 3
```

This is how Alpaca, Vicuna, and most open-source chat models were trained.

---

## 10. What's Next

### Fine-tuning Evolution

```
This demo:
  GPT-2 (124M) + LoRA + WikiText
  → Model learns Wikipedia style

Next step:
  LLaMA 3.2 1B + QLoRA + your custom data
  → Model learns your domain

Production:
  LLaMA 7B + QLoRA + instruction data
  → Custom chat assistant for your use case
```

### RAG — The Alternative to Fine-tuning

Fine-tuning teaches the model new knowledge by updating weights.

RAG retrieves knowledge at inference time without updating weights:

```
Fine-tuning:
  Bake knowledge into model weights
  Fast inference, but needs retraining when knowledge changes

RAG:
  Retrieve relevant documents at inference time
  No retraining, always up-to-date
  Slightly slower inference
```

Both are essential skills. RAG is next in Phase 4.

---

## Summary

**Fine-tuning GPT-2 with LoRA showed:**

1. **LoRA works** — 0.24% parameters trained, model still learns
2. **GPT-2 has quirks** — Conv1D instead of Linear, combined QKV
3. **Training objective is the same** — next token prediction
4. **Generated text reflects the dataset** — WikiText → encyclopedic style
5. **Production is identical** — same code, bigger model, add QLoRA

The workflow scales: GPT-2 → LLaMA 1B → LLaMA 7B → any decoder-only model.

---

*Phase 4 — concept 03 → you are here*  
*Previous concept → [02 — LoRA and PEFT](./02_lora_peft.md)*  
*Next → RAG — Retrieval Augmented Generation*
