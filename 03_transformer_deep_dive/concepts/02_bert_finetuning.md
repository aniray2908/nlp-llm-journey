# BERT and Fine-tuning — From Pre-training to Task-Specific Models

> **BERT revolutionised NLP by showing that pre-training on massive text, then fine-tuning on specific tasks, achieves state-of-the-art results.** The model learns general language understanding that transfers to anything.

---

## Table of Contents

1. [What is BERT?](#1-what-is-bert)
2. [Pre-training vs Fine-tuning](#2-pre-training-vs-fine-tuning)
3. [How BERT is Pre-trained](#3-how-bert-is-pre-trained)
4. [Fine-tuning for Classification](#4-fine-tuning-for-classification)
5. [Transfer Learning](#5-transfer-learning)
6. [Why Pre-training Alone Isn't Enough](#6-why-pre-training-alone-isnt-enough)
7. [The Fine-tuning Process](#7-the-fine-tuning-process)
8. [Practical Considerations](#8-practical-considerations)
9. [BERT Variants](#9-bert-variants)
10. [How This Connects to Modern LLMs](#10-how-this-connects-to-modern-llms)

---

## 1. What is BERT?

**BERT** = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers

Released by Google in 2018, BERT is a **pre-trained Transformer encoder** that learns bidirectional context.

### Key properties

```
BERT:
  - Encoder-only (no decoder)
  - Bidirectional (sees left AND right context)
  - Pre-trained on massive text corpus (3.3B words)
  - 12 layers, 12 attention heads, 110M parameters (BERT-base)
  - Can be fine-tuned for any downstream task
```

### Bidirectional vs Unidirectional

```
GPT (unidirectional):
  "The cat sat on the ___"
  Can only look left (at previous words)
  → predicts next token

BERT (bidirectional):
  "The cat [MASK] on the mat"
  Can look left AND right
  → predicts masked token using full context
```

BERT's bidirectional nature is why it's excellent for understanding (classification, Q&A) but not generation.

---

## 2. Pre-training vs Fine-tuning

### Pre-training

**Objective:** Learn general language understanding from huge text corpus

```
Input:  Billions of words from Wikipedia, Books, etc.
Task:   Masked language modelling (predict [MASK] tokens)
Result: Model learns:
  - Grammar
  - Vocabulary
  - Semantics
  - World knowledge
  - Reasoning patterns
```

**Why it works:**
- Model must understand language to predict masked words
- Bidirectional context forces deep understanding
- Billions of words provide signal for all language phenomena

**Cost:** Weeks of training on TPU clusters

### Fine-tuning

**Objective:** Adapt pre-trained model to specific task

```
Input:  Your task-specific data (reviews, Q&A, etc.)
Task:   Sentiment classification, NER, Q&A, etc.
Result: Model learns:
  - How general knowledge applies to your task
  - Task-specific patterns
```

**Why it's fast:**
- Model already knows language
- Only needs to learn task patterns
- Few hours/days on single GPU

**Cost:** Hours to days on modest hardware

---

## 3. How BERT is Pre-trained

### Masked Language Modelling (MLM)

**The training objective:**

```
Original:  "The cat sat on the mat"
Masked:    "The [MASK] sat on the mat"
Task:      Predict "cat" from context

BERT sees both:
  "The" (left)
  "sat on the mat" (right)
And must predict the hidden word.
```

**Why masking works:**
- Forces bidirectional understanding
- 15% of tokens are masked per training step
- Model can't cheat by just copying previous token
- Must understand full sentence structure

### Next Sentence Prediction (NSP)

Secondary objective (less important now):

```
Sentence A: "The cat sat on the mat"
Sentence B: "It was comfortable"

Task: Predict if B follows A logically

This teaches the model about discourse structure.
```

Modern BERT variants often skip NSP since MLM is sufficient.

### Training scale

```
Corpus:    3.3 billion words
Batch:     32,000 examples
Epochs:    40
Duration:  4 days on 64 TPU v3 chips
Cost:      ~$7,000 USD
Result:    110M parameter model
```

This is why pre-trained models are valuable — you get the benefit without the cost.

---

## 4. Fine-tuning for Classification

### Task: Sentiment Analysis

**Goal:** Classify review as positive or negative

### Architecture for classification

```
Input: "This movie was amazing"
  ↓
Tokenizer: [CLS] This movie was amazing [SEP]
  ↓
BERT encoder (12 layers of Transformers)
  ↓
[CLS] token representation (learned summary of input)
  ↓
Linear layer: 768 → 2 (positive/negative)
  ↓
Softmax: [0.1, 0.9] (90% positive, 10% negative)
  ↓
Output: Positive
```

### The [CLS] token

```
[CLS] = Classification token
  - Special token added at start
  - After BERT processing, becomes input representation
  - Used for all classification tasks
  - Model learns to encode "entire document summary" here
```

### Fine-tuning process

**Step 1: Load pre-trained BERT**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
```

**Step 2: Add task-specific layer**
```
BERT outputs: (batch, seq_len, 768)
Take [CLS] token: (batch, 768)
Linear layer: (batch, 768) → (batch, 2)
Softmax: convert to probabilities
```

**Step 3: Train on your data**
```
For each review:
  Forward pass through BERT + classification head
  Compute loss (cross-entropy)
  Backprop
  Update parameters
```

**Key difference from pre-training:**
- Only your task data is used (not billions of words)
- Only final layers change significantly
- Earlier BERT layers mostly stay frozen

---

## 5. Transfer Learning

**The core idea:** Knowledge learned on one task transfers to another task.

### Why it works for NLP

Pre-training teaches:
- Tokenisation patterns
- Grammatical structure
- Word relationships
- Semantic understanding
- Common reasoning patterns

These apply to almost any language task.

### Transfer learning pipeline

```
Step 1: Pre-train on massive corpus
  BERT learns general language understanding
  (One-time cost, shared by everyone)

Step 2: Fine-tune on your task
  Download pre-trained BERT
  Train on your data (hundreds to thousands of examples)
  Model adapts to your task

Step 3: Deploy
  Model now works for your task
  Achieves SOTA with minimal training
```

### Why this is powerful

**Without transfer learning:**
```
Need: 100,000+ labeled examples
Time: Weeks to train from scratch
Cost: Massive compute
Result: Mediocre performance
```

**With transfer learning:**
```
Need: 100-1000 labeled examples
Time: Hours to fine-tune
Cost: Single GPU
Result: State-of-the-art performance
```

This is why pre-trained models revolutionised NLP.

---

## 6. Why Pre-training Alone Isn't Enough

### The core problem

Pre-trained BERT knows language, but not your task.

```
BERT pre-trained on:
  - Wikipedia articles
  - Books
  - Web text
  - General language patterns

Your task:
  - Sentiment in movie reviews
  - Medical coding
  - Legal document classification
  - Domain-specific patterns BERT hasn't seen
```

### Example: Sentiment analysis

**Pre-trained BERT:**
```
Input: "This movie was terrible"
Task:  Predict masked tokens, not sentiment
Result: Doesn't know "terrible" → negative sentiment (no training signal)

Accuracy: ~50% (random guessing)
```

**Fine-tuned BERT:**
```
Input: "This movie was terrible"
Task:  Classify as positive/negative
Training signal: "terrible" appears in negative reviews
Result: Learns "terrible" → negative

Accuracy: ~85-90%
```

### What fine-tuning does

```
Pre-trained weights: frozen (mostly)
  BERT's general knowledge preserved

Task-specific layer: trained
  Linear layer learns: input features → task output

Fine-tuning adjusts:
  - Last few BERT layers (slight adaptation)
  - Classification head (learned from scratch)
  - Attention patterns (minor updates)
```

The model learns to apply its general knowledge to your specific task.

---

## 7. The Fine-tuning Process

### Step-by-step

```
1. Data preparation
   - Tokenize reviews
   - Convert to token IDs
   - Pad to same length
   - Create train/val/test splits

2. Model setup
   - Load pre-trained BERT
   - Add classification head (2 classes: positive/negative)
   - Move to GPU
   - Set up optimizer (AdamW, learning rate 2e-5)

3. Training loop (for each epoch)
   For each batch of reviews:
     - Forward pass: reviews → logits
     - Compute loss: cross-entropy between prediction and true label
     - Backward pass: compute gradients
     - Update parameters: optimizer step
     - Track loss

4. Validation
   - Evaluate on validation set
   - Check for overfitting (val loss >> train loss)
   - Early stopping if needed

5. Evaluation
   - Test on held-out test set
   - Compute metrics: accuracy, precision, recall, F1
   - Analyze errors
```

### Key hyperparameters

| Param | Value | Why |
|-------|-------|-----|
| **Learning rate** | 2e-5 to 5e-5 | Low (pre-trained weights shouldn't change much) |
| **Batch size** | 16-32 | Limited by GPU memory |
| **Epochs** | 2-4 | Few (overfits easily on small datasets) |
| **Warmup** | 0-500 steps | Gradual LR increase (optional but helps) |
| **Weight decay** | 0.01 | Regularization to prevent overfitting |

### Training dynamics

```
Epoch 1:
  Train loss: 0.7 → 0.3
  Val loss: 0.5 → 0.4
  (Model learning task patterns)

Epoch 2:
  Train loss: 0.2 → 0.1
  Val loss: 0.4 → 0.35
  (Fine-tuning continues)

Epoch 3:
  Train loss: 0.05
  Val loss: 0.38 ← Starting to overfit
  (Train loss << val loss, consider stopping)

Epoch 4:
  Train loss: 0.01
  Val loss: 0.45 ← Overfitting clear, stop here
```

Early stopping at epoch 2-3 usually works best.

---

## 8. Practical Considerations

### Memory and compute

```
BERT-base: 110M parameters
  GPU memory needed: ~2-4 GB per batch

If you have:
  - 8 GB GPU: batch size 8-16
  - 16 GB GPU: batch size 16-32
  - 24 GB GPU: batch size 32-64
  - Out of memory: reduce batch size or use gradient accumulation
```

### Overfitting on small datasets

```
Dataset: 500 examples
BERT: 110M parameters

Risk: BERT memorises training data
Solution:
  - Early stopping (stop when val loss increases)
  - Dropout (0.1-0.3)
  - Lower learning rate
  - More epochs but shorter training
```

### Data augmentation

With limited data, augment to prevent overfitting:

```
Original: "This movie was amazing"
Augment 1: "This film was amazing" (synonym)
Augment 2: "amazing movie" (reorder)
Augment 3: "awesome movie" (synonym)

Gives model more examples without manual labeling.
```

### Class imbalance

If you have 900 positive, 100 negative reviews:

```
Solution 1: Weighted loss
  Penalise errors on minority class more

Solution 2: Oversampling
  Duplicate minority examples

Solution 3: Undersampling
  Remove majority examples (loses data)

Solution 4: SMOTE
  Synthetic minority oversampling
```

---

## 9. BERT Variants

### Official variants

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| **BERT-tiny** | 4M | 50x faster | Lower |
| **BERT-small** | 29M | 5x faster | Medium |
| **BERT-base** | 110M | Baseline | High |
| **BERT-large** | 340M | 8x slower | Highest |

### Popular alternatives

```
DistilBERT:  40% smaller, 60% faster, 97% performance
RoBERTa:     Improved pre-training, better than BERT
ALBERT:      Parameter-sharing, smaller
ELECTRA:     Different pre-training, competitive
```

### Domain-specific variants

```
SciBERT:     Scientific papers (PubMed, ArXiv)
FinBERT:     Financial texts
BiomedBERT:  Biomedical literature
LegalBERT:   Legal documents
```

These are pre-trained on domain data, fine-tune better on domain tasks.

### Multilingual variants

```
mBERT:       BERT trained on 104 languages
XLM-RoBERTa: 100+ languages
Fine-tune on English, use for other languages
```

---

## 10. How This Connects to Modern LLMs

### Evolution

```
BERT (2018):
  - Encoder-only
  - 110M parameters
  - Pre-trained for 4 days
  - Fine-tuned for specific tasks

GPT-2 (2019):
  - Decoder-only
  - 1.5B parameters
  - Pre-trained for weeks
  - Few-shot learning (no fine-tuning needed)

GPT-3 (2020):
  - Decoder-only
  - 175B parameters
  - Pre-trained on 300B tokens
  - In-context learning (0-shot, few-shot, chain-of-thought)

GPT-4 (2023):
  - Decoder-only
  - 1.7T parameters (estimated)
  - Trained for months
  - Reasoning, multi-modal, alignment

Claude (2023):
  - Decoder-only
  - Trained with RLHF + Constitutional AI
  - Fine-tuned for helpfulness, harmlessness, honesty
```

### Key differences

```
BERT:           GPT/Claude:
Encoder         Decoder
Bidirectional   Unidirectional (left-to-right)
Classification  Generation
Fine-tune       Few-shot + in-context learning
Small scale     Massive scale
```

### Modern fine-tuning

```
BERT way (2018):
  1. Get pre-trained BERT
  2. Add task-specific head
  3. Fine-tune on your data

Modern way (2024):
  1. Get pre-trained LLM (GPT, Claude, LLaMA)
  2. Few-shot prompt engineering (no training)
  3. Optional: Fine-tune with LoRA if needed

Why different:
  - LLMs are so large they learn from examples in context
  - BERT needed explicit fine-tuning
  - Scale changed the paradigm
```

### But BERT principles remain

Everything in modern LLMs traces back to BERT:

```
Pre-training:       Still the foundation
Transfer learning:  Still dominates
Fine-tuning:        Still works (now with LoRA)
Tokenisation:       BERT introduced special tokens ([CLS], [SEP])
Attention:          Still the core mechanism
```

---

## Summary

**BERT showed that:**

1. **Pre-training on massive text is worth it** — model learns general language understanding
2. **Fine-tuning is fast** — task adaptation takes hours, not weeks
3. **Transfer learning works** — knowledge from one task helps another
4. **Bidirectional context matters** — seeing left and right is better than left-only
5. **Scale matters** — bigger models learn better (within limits)

**Modern LLMs evolved BERT by:**
- Making models bigger (100M → 1T parameters)
- Switching to decoder-only (for generation)
- Training longer (4 days → months)
- Using RL and alignment (for safety)
- Adding in-context learning (no fine-tuning needed)

But the core principles — pre-train, transfer, fine-tune — remain the same.

---

*Phase 3 — concept 02 → you are here*  
*Previous concept → [01 — Attention Mechanisms](./01_attention_mechanisms.md)*  
*Next → Phase 4: LLM Internals + Fine-tuning*
