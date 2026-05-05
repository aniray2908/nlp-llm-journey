# Educational Rewriter GPT — Fine-tuning LLaMA for Educational Content

Fine-tuning a language model to rewrite confusing educational content is a fundamentally different task from pre-training. The model already knows language — the goal is to teach it a specific behaviour: take dense text, apply a targeted transformation, and produce something clearer. This concept note covers every decision made in Phase 5, from dataset design to deployment.

## Table of Contents

1. Why Fine-tune Instead of Prompt?
2. Choosing the Base Model
3. QLoRA — Efficient Fine-tuning at Scale
4. Dataset Design
5. Synthetic Data Generation
6. Prompt Format and Chat Templates
7. The 6 Rewrite Modes
8. Training Configuration
9. Evaluation Framework
10. Audience Suitability Score
11. Jargon Detection System
12. From Model to Product

---

## 1. Why Fine-tune Instead of Prompt?

The first question is always: do you need to fine-tune at all?

```
Prompting only:
  Send "Rewrite this more simply: {text}" to Claude/GPT-4
  Pros: Fast, no training, high quality
  Cons: Expensive per call, API dependency,
        no control over behaviour, not portable

Fine-tuning:
  Train a smaller model to do one thing very well
  Pros: Cheap inference, runs locally, consistent behaviour,
        deployable anywhere, publishable as a model
  Cons: Upfront training cost, lower ceiling than GPT-4
```

For the capstone use case — a deployed educational tool — fine-tuning makes sense:

```
1. The task is narrow and well-defined
   Six specific rewrite modes, educational text only
   Fine-tuning excels at narrow, consistent tasks

2. Inference cost matters
   A deployed app making thousands of rewrites
   API cost per call adds up fast

3. The model should be publishable
   A fine-tuned model on HuggingFace Hub is a portfolio artefact
   An API call wrapper is not

4. Consistency
   Fine-tuned models produce predictable outputs
   Prompting Claude directly has more variance
```

The decision: fine-tune LLaMA 3.2 3B on 846 curated rewrite pairs.

---

## 2. Choosing the Base Model

The base model choice has a large impact on output quality and training cost.

### The Options Considered

```
GPT-2 (124M):
  Pros: Tiny, fast, local
  Cons: Poor instruction following, incoherent outputs
  Verdict: Too small for this task

LLaMA 3.2 1B:
  Pros: Modern architecture, runs locally with QLoRA
  Cons: Instruction following still weak at 1B scale
  Tried: Val loss 1.28, output quality inconsistent

LLaMA 3.2 3B:
  Pros: Strong instruction following, modern architecture
        Runs on T4 GPU with QLoRA, good output quality
  Cons: 150 min training, slightly more memory
  Chosen: Val loss 1.065, 4-5/6 modes working well

LLaMA 3.2 7B:
  Pros: Best instruction following in the open-source family
  Cons: Needs A100 or multi-GPU, much longer training
  Future work: Upgrade path if 3B quality is insufficient
```

### Why LLaMA Over Other Families

```
LLaMA 3.2 architectural improvements over GPT-2:
  RoPE (Rotary Position Embedding)
    → Better long-sequence handling
    → Position information encoded in attention, not embeddings
  
  RMSNorm instead of LayerNorm
    → Faster computation, more stable training
    → No mean subtraction step
  
  SwiGLU instead of GELU
    → Better feed-forward activation
    → Empirically better on language tasks
  
  Grouped Query Attention
    → Faster inference (fewer key/value heads)
    → Same quality as full multi-head attention
```

### The 1B vs 3B Lesson

The failure of LLaMA 3.2 1B is worth documenting:

```
1B model qualitative results (one test passage):
  Default:       Copied input verbatim
  Simpler:       Copied input verbatim
  Add Example:   Wrong domain example
  Concise:       Copied input verbatim
  Step by Step:  Repetitive loops (Step 2 = Step 1 = Step 3)
  Add Analogy:   Nonsensical comparison

Root cause:
  1B parameters is below the threshold for reliable
  instruction following on multi-mode tasks
  The model learns "something happens" but not "what"

3B model qualitative results (same passage):
  Default:       Clear structure, accessible
  Simpler:       Mostly simpler, occasional jargon
  Add Example:   Domain-relevant examples added
  Concise:       49% length reduction consistently
  Step by Step:  Numbered steps, correct mechanism
  Add Analogy:   Relevant real-world comparisons

Lesson: Model scale has a hard threshold for instruction following.
        Below it, fine-tuning doesn't compensate.
```

---

## 3. QLoRA — Efficient Fine-tuning at Scale

Fine-tuning all 3B parameters of LLaMA requires enormous memory. QLoRA makes it feasible on a single T4 GPU.

### The Memory Problem

```
LLaMA 3.2 3B:
  Full precision (fp32):   ~12 GB just for weights
  Half precision (bf16):   ~6 GB for weights
  + Gradients:             ~6 GB
  + Optimiser states:      ~12 GB (Adam stores 2 moments)
  
  Total full fine-tune:    ~24-36 GB
  T4 GPU memory:            16 GB

  → Full fine-tuning is impossible on a T4
```

### QLoRA: Two Techniques Combined

**Step 1: Quantise the base model to 4-bit**

```
4-bit NF4 quantisation:
  Original weight: 32-bit float (4 bytes per number)
  Quantised weight: 4-bit integer (0.5 bytes per number)
  
  Memory reduction: 8x for weights
  3B model: 12 GB → 1.5 GB

  NF4 = NormalFloat4
    Designed for normally distributed weights
    Better than uniform 4-bit quantisation
    Minimises quantisation error for typical LLM weight distributions
```

**Step 2: Add LoRA adapters in bf16**

```
Freeze quantised base weights (never updated)
Add small trainable matrices A and B to attention layers:

  W_new = W_quantised + A × B × (alpha / rank)

  Where:
    W_quantised: frozen 4-bit base weights
    A:  (d_model × rank) trainable matrix
    B:  (rank × d_model) trainable matrix
    rank: 16 (much smaller than d_model = 3072)
```

### Memory with QLoRA

```
4-bit base model:        ~1.5 GB
LoRA adapters (fp16):    ~50 MB
Gradients (adapters only): ~50 MB
Optimiser states:        ~100 MB

Total:                   ~1.7 GB active memory
Peak during training:    ~8-10 GB (activations + batch)

T4 has 16 GB → comfortably fits with batch size 2
```

### LoRA Configuration Decisions

```
Rank = 16 (higher than Phase 4's rank 8):
  Phase 4 task: Learn GPT-2's WikiText style (narrow)
  Phase 5 task: Learn 6 distinct rewrite modes (wider)
  Higher rank = more expressiveness needed

Alpha = 32 = 2 × rank:
  Standard rule of thumb: alpha = 2 × rank
  Controls the scaling of LoRA updates
  Higher alpha = updates have more influence on output

Dropout = 0.05:
  Prevents LoRA adapters from overfitting
  Small value because dataset is small (612 examples)

Target modules = q_proj, v_proj, k_proj, o_proj:
  All four attention projections (vs Phase 4's q+v only)
  More expressive — all attention directions updated
  Necessary for 6-mode task vs single-style task

Trainable parameters: 3,407,872 (0.45% of 752M)
Memory reduction vs full fine-tune: ~14x
```

---

## 4. Dataset Design

The dataset is the most important decision in fine-tuning. Model quality is bounded by data quality.

### The Core Format

Each training example is a triplet:

```
{
  "input":  "confusing educational passage",
  "mode":   "simpler",
  "output": "clearer rewrite in requested mode"
}
```

### Source Passage Strategy

```
Target: 150 passages (ended up with 141 after cleaning)

Wikipedia (66 passages):
  Why: Educational intent, diverse domains, CC BY-SA license
       Dense technical sections are naturally confusing
  How: Wikipedia API, filtered for paragraph density,
       sentence length, absence of markup artifacts
  Domains: CS/ML, biology, physics, mathematics,
           economics, chemistry, medicine

arXiv (75 passages):
  Why: Academic abstracts are maximally jargon-heavy
       Represent the hardest rewriting challenge
       Open access, diverse fields
  How: arXiv API, filtered for 100-300 word abstracts
  Domains: ML, NLP, quantum computing, 
           economics, statistics, biology

Personal notes (0 passages):
  Originally planned for 20 passages
  Removed — single domain (NLP), subjective quality,
  insufficient to justify the complexity
```

### Why Mixed Sources

```
Wikipedia alone:
  Educational in intent, but not always confusing
  Risk: Model learns to rewrite "already clear" text

arXiv alone:
  Always confusing, but very uniform register (academic)
  Risk: Model only works on paper abstracts

Mix of both:
  Coverage of different types of confusing text
  Different registers, different domains
  Better generalisation to real educational content
```

### Dataset Splits — Stratified at Passage Level

```
Key design decision: split at PASSAGE level, not example level

Wrong (example-level split):
  Train set: passage 001 (default, simpler, concise)
  Test set:  passage 001 (add_example, step_by_step, add_analogy)
  
  Problem: Model has seen passage 001 during training
           Test evaluation is contaminated (data leakage)

Correct (passage-level split):
  Train set: passage 001 (all 6 modes)
  Test set:  passage 047 (all 6 modes)
  
  Passages are completely disjoint between splits
  Test evaluation is clean

Final splits:
  Train:      612 examples (99 passages, 70%)
  Validation: 138 examples (21 passages, 17%)
  Test:        96 examples (21 passages, 13%)
```

---

## 5. Synthetic Data Generation

Getting 846 high-quality rewrite pairs required a scalable approach.

### The Alpaca Methodology

Stanford Alpaca (2023) showed that large language models can generate high-quality instruction-following data for fine-tuning smaller models. This approach — using a capable model to teach a less capable one — has become standard practice:

```
Alpaca:    GPT-3 generates data → fine-tune LLaMA
Orca:      GPT-4 generates data → fine-tune LLaMA
Phase 5:   Claude generates data → fine-tune LLaMA 3.2 3B
```

This is legitimate because:
- The fine-tuned model is not competing with Claude
- The task (educational rewriting) is narrow and well-defined
- The methodology is published and peer-reviewed
- The dataset and process are fully disclosed

### The Generation Script

```python
# For each passage × each mode:
response = claude_client.messages.create(
    model="claude-sonnet-4-5",
    system=mode_system_prompt,
    messages=[{"role": "user", "content": f"Text: {passage}"}]
)
rewrite = response.content[0].text
```

Key design decisions in the generation script:

```
Auto-save after every passage:
  generation is a 90-minute run
  if it crashes at passage 100, resume from 100
  no wasted API calls

0.5 second delay between calls:
  respects API rate limits
  prevents 429 errors

Quality filter:
  skip rewrites under 5 words
  retry up to 3 times on failure
  log and skip on persistent failure

Cost: $3.12 for 846 examples
      ~$0.0037 per example
```

### Dynamic Prompting (50/50 Split)

```
Problem: At inference time, users might write
         simple instructions OR detailed ones.
         Model should handle both.

Solution: Generate training data with both:
  50% examples: short system prompt
    "You are an expert educational rewriter.
     Rewrite clearly. Output ONLY the rewrite."
  
  50% examples: detailed system prompt
    Full per-mode instructions with explicit constraints

Result: Model learns to follow both styles
        Prompt_style column tracks which was used
        Final split: 49% short / 51% detailed
```

### Manual Seed Examples

```
10-20 hand-written examples were planned as seeds
to anchor Claude's generation quality.

In practice: removed along with personal notes.
The Claude API generation was consistent enough
without seeds given the explicit per-mode prompts.

If quality had been lower, seeds would have been
included as few-shot examples in the generation prompt.
```

---

## 6. Prompt Format and Chat Templates

### Why Chat Format

LLaMA 3.2 was pre-trained with a specific chat template. Using the same format at fine-tuning time means:

```
Base model expectations (from pre-training):
  <|system|>    → sets context and behaviour
  <|user|>      → human input
  <|assistant|> → model response

Fine-tuning with same format:
  → Model already knows how to use these roles
  → Faster learning, less data needed
  → Consistent with inference-time usage
```

If you use a different format during fine-tuning, the model has to unlearn pre-training patterns first — wasting training capacity.

### The Format

```
<|system|>
You are an expert educational content rewriter.
Rewrite text clearly based on the requested mode.
Preserve the original meaning exactly.
Output ONLY the rewrite, nothing else.

<|user|>
Mode: Simpler
Text: {confusing passage}

<|assistant|>
{simpler rewrite}
```

### Dynamic Prompting at Inference

At inference time, each mode uses a specialised system prompt rather than a generic one:

```
Simpler mode system prompt:
  "Rewrite for a 16-year-old with no technical background.
   Replace ALL technical terms with everyday words.
   Maximum 15 words per sentence. Never use acronyms."

Step by Step mode system prompt:
  "Break down HOW THIS CONCEPT WORKS into numbered steps.
   Do NOT explain history. Do NOT add information not in original.
   Format strictly as: 1. ... 2. ... 3. ..."
```

This specialisation happens at inference only — training used a single mode label in the user message. This means:

```
Training:  Mode label teaches the behaviour
Inference: Detailed system prompt enforces the constraints

The model learned what "simpler" means during training.
The system prompt reminds it of the specific constraints.
```

---

## 7. The 6 Rewrite Modes

Each mode addresses a specific educational clarity failure.

```
Default:
  Failure addressed: generally unclear writing
  What it does:      improve structure, flow, vocabulary
  Metric:            LLM-as-judge clarity score

Simpler:
  Failure addressed: jargon and complexity barriers
  What it does:      replace technical terms, shorten sentences
  Metric:            Flesch-Kincaid grade level reduction

Add Example:
  Failure addressed: abstract concepts without grounding
  What it does:      append a concrete, domain-relevant example
  Metric:            presence of example indicator phrases

Concise:
  Failure addressed: verbose, redundant writing
  What it does:      remove filler, compress without losing meaning
  Metric:            output length / input length ratio

Step by Step:
  Failure addressed: processes and mechanisms not broken down
  What it does:      convert prose into numbered steps
  Metric:            presence of numbered step patterns

Add Analogy:
  Failure addressed: counterintuitive or purely abstract concepts
  What it does:      add a real-world comparison with explanation
  Metric:            presence of comparison phrases
```

### Modes That Were Removed

```
"Try Again":
  Problem: Not a clear training objective
           Model has no context for what was wrong
           Would just generate another random rewrite
           A UI instruction masquerading as a mode
  Removed: Replaced with nothing — the 6 remaining
           modes are sufficient and well-defined

"More Detailed":
  Problem: Vague — detailed in what direction?
           Risk: model just adds filler words
  Replaced: "Step by Step" — same intent, clearer constraint
            "Step by Step" for processes and mechanisms
            is more useful in educational content
```

---

## 8. Training Configuration

### Hyperparameter Decisions

```
Epochs = 3:
  Validation loss still decreasing at epoch 3
  Gap of 0.021 at epoch 3 (minimal overfitting)
  More epochs risked overfitting on 612 examples

Learning rate = 2e-4 with cosine schedule:
  Standard for LoRA fine-tuning
  Higher than full fine-tuning (fewer params to update)
  Cosine: smoothly decays to near-zero at end of training
  Warmup: 50 steps (5% of total) to stabilise early training

Batch size = 2 + gradient accumulation 8:
  Effective batch size = 16
  T4 has 16GB — batch 4 caused OOM with 3B model
  Reduced to 2, doubled accumulation steps
  Same gradient quality, half the memory per step

Max sequence length = 512:
  Covers 95%+ of examples (average: 121 input + 135 output words)
  Longer sequences use more memory quadratically (attention is O(n²))
  Truncation at 512 is safe for this dataset
```

### What Failed

```
Attempt 1: LLaMA 3.2 1B
  Abandoned: output quality too low for all modes
  Learned: 1B is below instruction-following threshold

Attempt 2: LLaMA 3.2 3B on T4 x2 (dual GPU)
  Failed: RuntimeError — QLoRA doesn't support multi-GPU
          Parameters split across cuda:0 and cuda:1
  Fixed: Set CUDA_VISIBLE_DEVICES=0 before model load
         Single GPU, full 16GB available

Attempt 3: fp16 training
  Failed: NotImplementedError — LLaMA 3.2 uses BFloat16 internally
          fp16 grad scaler incompatible
  Fixed: Set bf16=True instead of fp16=True

Attempt 4: warmup_ratio in SFTConfig
  Failed: Deprecated in TRL v5.2+
  Fixed: Use warmup_steps=50 directly

Each failure added a workaround that's now documented
in the training notebook for future reference.
```

### Training Results

```
Epoch 1: Train 1.737 | Val 1.183  (model learning fast)
Epoch 2: Train 1.149 | Val 1.086  (still improving)
Epoch 3: Train 1.044 | Val 1.065  (converging cleanly)

Train/val gap at epoch 3: 0.021
  → Minimal overfitting despite small dataset
  → Model generalises to unseen passages
  → 3 epochs was the right stopping point
```

---

## 9. Evaluation Framework

Three evaluation methods were used, each answering a different question.

### System 1: Mode-Specific Automated Metrics

```
Question: Does each mode do what it's supposed to?

Concise:       output_words / input_words < 1.0
               Result: 100% of test set shorter ✅

Simpler:       Flesch-Kincaid grade (output) < FK grade (input)
               Result: 98% more readable, avg -4.78 grade levels ✅

Step by Step:  regex detection of "1.", "Step 1", "**Step"
               Result: 100% of test set has numbered steps ✅

Add Example:   detection of "for example", "such as", "e.g.", etc.
               Result: 100% of test set has example phrases ✅

Add Analogy:   detection of "like a", "similar to", "just like", etc.
               Result: 89% of test set has analogy phrases ✅
```

### System 2: LLM-as-Judge

```
Question: Are the rewrites actually good?

Method: Sample 20% of test set (56 examples)
        Ask Claude to rate each on 4 dimensions:
          Clarity (1-5):        Is it clearer than the original?
          Mode Adherence (1-5): Does it follow the mode correctly?
          Accuracy (1-5):       Does it preserve the original meaning?
          Overall (1-5):        Is this a good educational rewrite?

Results:
  Clarity:        4.50/5.0
  Mode Adherence: 4.66/5.0  ← highest
  Accuracy:       4.59/5.0
  Overall:        4.53/5.0

Per mode (overall):
  Simpler:       5.00/5.0
  Step by Step:  4.83/5.0
  Add Example:   4.57/5.0
  Add Analogy:   4.40/5.0
  Default:       4.38/5.0
  Concise:       4.33/5.0

All modes above 4.0 = "Good" threshold
```

### System 3: Manual Spot-Check

```
Question: Does it pass a human sanity check?

Method: Review 14 examples (5% of test set) manually
        Compare against LLM-as-judge scores
        Flag systematic failures

Findings:
  Strong: Simpler (autoimmune disease), 5/5/5/5
          Step by Step (black holes), 5/5/4/5
          Add Example (Java code generation), 5/5/5/5
  
  Weak:   Add Example (quantum systems)
          → Hallucinated specific technical details
          Add Example (VR study)
          → Introduced unverified device names

Key insight: Add Example mode occasionally fabricates specifics.
             The model knows to add an example but not always
             what an accurate example looks like.
```

### Why One-Passage Testing Is Misleading

```
Initial qualitative test (1 passage, 1 run):
  Result: 4/6 modes working

LLM-as-judge (56 examples, diverse passages):
  Result: 6/6 modes above 4.0 threshold

The single-passage test was misleading because:
  The model performs differently across passage domains
  One passage (Transformers) is not representative
  Scale reveals quality that spot-checks miss
```

---

## 10. Audience Suitability Score

The audience score is a formula-based 1-10 metric computed from the rewrite at inference time. It requires no additional model — just text analysis.

### The Scale

```
10  Accessible to a curious 10-year-old
9   Middle school level
8   High school general audience
7   High school advanced
6   Undergraduate non-specialist
5   Undergraduate in related field
4   Undergraduate specialist
3   Bachelor's minimum in field
2   Graduate level
1   Domain expert / researcher
```

### The Formula

```
Score = readability (40%) + jargon density (25%) + 
        sentence complexity (20%) + concept density (15%)

Each component normalised to 0-10 before weighting.
Higher component score = more accessible.
```

### Component Design

```
Readability (40%):
  Flesch-Kincaid grade level
  Grade 0 → score 10 (very easy)
  Grade 16+ → score 1 (very hard)
  Highest weight: most reliable proxy for accessibility

Jargon density (25%):
  Average jargon score across all words
  Uses Simple English Wikipedia frequency tiers
  0.0 = no jargon (score 10), 1.0 = all jargon (score 1)

Sentence complexity (20%):
  Average sentence length + subordinate clause density
  Short sentences with few "which/that/although" = simpler

Concept density (15%):
  Noun phrase density per sentence
  Proxy for information load per sentence
  Lowest weight: noisiest signal
```

### Validation Results

```
Simple ("The dog sat in the park"):   9.4/10 ✅
Educational ("Photosynthesis..."):    7.3/10 ✅
Technical ("Backpropagation..."):     4.1/10 ✅
Very technical ("Eigendecomposition"): 5.2/10 ✅

Mode score deltas (vs original input):
  Simpler:       +1.55 ← largest improvement ✅
  Step by Step:  +0.70 ← second largest ✅
  Default:       -0.22 (adds structure, not simpler)
  Concise:       -0.20 (shorter but denser)
  Add Example:   -0.11 (longer, same complexity)
  Add Analogy:   -0.25 (longer, adds concepts)
```

---

## 11. Jargon Detection System

The jargon system provides the 25% weight in the audience score formula and is reusable as a standalone component.

### Why Simple English Wikipedia

```
Options considered:
  A: Top 10,000 common English words (fixed list)
     Problem: Context-agnostic, "Python" = jargon
  
  B: Domain-specific jargon lists per field
     Problem: Can't cover all domains, maintenance burden
  
  C: Simple English Wikipedia word frequencies
     Chosen: Written for non-native speakers and younger readers
             Principled benchmark for accessibility
             Domain-agnostic by construction
             Free to download
```

### Frequency Tiers

```
Simple English Wikipedia corpus:
  59 articles, 112,433 words

Tier thresholds (calibrated for corpus size):
  Common    (freq ≥ 50):   jargon score 0.0
  Familiar  (freq 10-49):  jargon score 0.3
  Technical (freq 2-9):    jargon score 0.7
  Jargon    (freq < 2):    jargon score 1.0

Binary (0/1) was too coarse.
Tiered scoring captures degrees of jargon:
  "learning" = familiar (0.3), not common or jargon
  "algorithm" = technical (0.7)
  "backpropagation" = jargon (1.0)
```

### Supplements

```
Simple Wikipedia has gaps — science-focused corpus
misses common everyday words that appear rarely
in technical articles ("sat", "birds", "park").

Supplements:
  Dolch word list (220 sight words, grades K-3)
  Fry word list (500 most common English words)
  Both given baseline frequency of 5,000
  (treated as always "common")

Effect: Everyday words correctly classified as common
        even when rare in the Wikipedia corpus
```

### Known Limitation

```
Simple text still scores ~0.22 jargon density:
  "The dog sat in the park and looked at the birds"
  "sat", "looked", "birds", "park" appear rarely
  in a science-focused 59-article corpus

This is an acceptable approximation.
The direction is always correct (simple < complex).
The absolute value is what needs calibration.
```

---

## 12. From Model to Product

### What Gets Deployed

```
HuggingFace Hub: ray-2908/educational-rewriter-lora
  LoRA adapters only (13 MB)
  Not the full 3B base model weights
  
  Why adapters only:
    Legally clean (your weights, not Meta's)
    Tiny file size (13 MB vs 6 GB)
    Standard practice for LoRA models
    User loads base model + your adapters

License: Apache 2.0
  Attribution required in derivative works
  Commercial use permitted
  Your name attached permanently
```

### The Inference Function

```python
def rewrite(model, tokenizer, text, mode="default"):
    """
    Returns:
        rewrite_text:      the rewritten passage
        output_score:      audience score (1-10)
        recommendation:    suggestion to simplify further, or None
    """
```

This is the interface the capstone imports. Everything — model loading, prompt formatting, audience scoring, recommendation generation — is encapsulated in `inference.py`.

### The Iterative Loop

```
User pastes text
  ↓
Model generates rewrite
  ↓
Audience score computed (e.g. 3.7/10 — Graduate level)
  ↓
Pop-up: "This content requires graduate-level knowledge.
         Want a version for undergraduates?"
  ↓
User clicks Show Me
  ↓
Simpler mode applied → new rewrite
  ↓
New score computed (e.g. 5.3/10 — Undergraduate level)
  ↓
New recommendation shown
  ↓
Repeat until satisfied or score ≥ 8
```

### What Phase 5 Delivers to the Capstone

```
Fine-tuned model:    ray-2908/educational-rewriter-lora
Inference function:  inference.py (rewrite, score, loop)
Jargon system:       data/jargon/jargon_scores.json
Dataset:             ray-2908/educational-rewriter-dataset
Evaluation results:  results/ (3 charts)

The capstone imports all of this and builds on top.
It does not retrain or modify the model.
```

---

## Summary

Fine-tuning LLaMA 3.2 3B for educational content rewriting required decisions at every stage:

```
Base model:    3B > 1B > GPT-2 for instruction following
Method:        QLoRA — 14x memory reduction, same quality
Dataset:       846 synthetic examples, Claude API, Alpaca method
Format:        Chat template, dynamic prompting, 50/50 short/detailed
Modes:         6 well-defined rewrite objectives
Training:      3 epochs, 2e-4 LR, effective batch 16, bf16
Evaluation:    3 systems — automated, LLM-as-judge, manual
Audience score: Formula-based 1-10 from text features
Jargon system: Simple English Wikipedia + frequency tiers
Deployment:    Apache 2.0, adapters only, HuggingFace Hub
```

The most important lesson from Phase 5 is that data quality determines model quality ceiling. The 3B model was not the limiting factor — the training data was. Better rewrite pairs with stricter domain-specificity constraints would produce better Add Example and Simpler outputs without any change to the model or training setup.

---
