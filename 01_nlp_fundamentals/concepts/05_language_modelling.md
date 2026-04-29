# Language Modelling Basics — n-grams, Perplexity

> **Path:** `01_nlp_fundamentals/concepts/`  
> **Topic:** Predicting the next word. The formal foundation of how LLMs are trained and evaluated.

---

## Table of Contents

1. [What is a Language Model?](#1-what-is-a-language-model)
2. [The Core Task — Conditional Probability](#2-the-core-task----conditional-probability)
3. [n-gram Language Models](#3-n-gram-language-models)
4. [Perplexity — Measuring Language Model Quality](#4-perplexity----measuring-language-model-quality)
5. [Smoothing & Backoff](#5-smoothing--backoff)
6. [Neural Language Models](#6-neural-language-models)
7. [Autoregressive Generation](#7-autoregressive-generation)
8. [The Sparsity Problem & Why Neural Models Win](#8-the-sparsity-problem--why-neural-models-win)
9. [Pre-training on Language Modelling](#9-pre-training-on-language-modelling)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. What is a Language Model?

A **language model** assigns a probability to sequences of words. It answers: "What's the probability of this text?"

```
P("The cat sat on the mat") = ?
P("The cat on sat the mat") = ? (grammatically wrong, should have lower probability)
P("Colorless green ideas sleep furiously") = ? (grammatical but nonsensical)
```

More practically, a language model **predicts the next word** given context:

```
Context: "The cat sat on the"
Model predicts:
  P(mat | context) = 0.6
  P(floor | context) = 0.2
  P(dog | context) = 0.05
  ... (rest of vocabulary)
```

To predict well, the model must understand:
- **Grammar** — what parts of speech can follow
- **Semantics** — what words are meaningful together
- **Facts** — what's true about the world
- **Context** — how earlier words constrain later ones

A model that predicts well has implicitly learned language.

---

## 2. The Core Task — Conditional Probability

Language modelling formalises this as learning **P(word | context)** — the probability of a word given everything before it.

### Chain rule decomposition

For a sequence of words, use the chain rule:

```
P(w₁, w₂, w₃, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁...wₙ₋₁)
```

Each word's probability depends on **all previous words**.

```
Example: "The cat sat"

P("The cat sat") = P("The") × P("cat" | "The") × P("sat" | "The cat")
                 = 0.01 × 0.5 × 0.8
                 = 0.004
```

### What makes a good model?

- **High probability for real text** — if a sentence is grammatical, P should be high
- **Low probability for nonsense** — if a sentence is ungrammatical or false, P should be low

This is the entire training objective: **assign high probability to real language, low probability to non-language**.

---

## 3. n-gram Language Models

The simplest approach: **count word sequences and convert to probabilities**.

### Unigram model

Just word frequencies:

```
P(word) = count(word) / total_words

Example:
  "the" appears 1000 times out of 10,000 words
  P("the") = 0.1
  P("cat") = 0.01
```

A unigram model doesn't use context — every word is equally likely given any context. Terrible for language modelling, but it's the baseline.

### Bigram model

Pairs of consecutive words:

```
P(word | previous_word) = count(previous, word) / count(previous)

Example: "the cat sat on the mat"
  Bigrams: (the, cat), (cat, sat), (sat, on), (on, the), (the, mat)
  
  P(cat | the) = count(the, cat) / count(the)
  P(sat | cat) = count(cat, sat) / count(cat)
```

A bigram model uses one word of context. Much better than unigram.

### Trigram model

Triples of consecutive words:

```
P(word | previous_two_words) = count(word₋₂, word₋₁, word) / count(word₋₂, word₋₁)

Example:
  P(sat | the cat) = count(the, cat, sat) / count(the, cat)
```

A trigram model uses two words of context. Generally better than bigram, with diminishing returns.

### Higher-order n-grams

You can keep going: 4-grams, 5-grams, etc. But there's a trade-off:

```
More context → better predictions (on paper)
But also → sparser counts, more unseen sequences, doesn't generalise
```

In practice, 5-grams is often a sweet spot before hitting sparsity problems.

---

## 4. Perplexity — Measuring Language Model Quality

**Perplexity** is a metric that measures how well a language model predicts text. **Lower perplexity is better.**

### Definition

```
Perplexity = exp(cross-entropy loss)
           = exp(- (1/N) Σ log P(wᵢ | context))

Where:
  N = number of words in test set
  P(wᵢ | context) = probability the model assigns to word wᵢ
  Σ = sum over all words in test set
```

### Intuition

Perplexity measures: **"On average, how many equally plausible words could come next?"**

```
Perplexity = 1:      Perfect model (always 100% sure)
Perplexity = 2:      Model is uncertain between 2 equally likely words
Perplexity = 10:     Model thinks ~10 words are equally likely
Perplexity = 50,000: Model is lost (whole vocabulary seems equally likely)
```

### Examples

```
Sentence: "The cat sat on the mat"

Perfect model: P(cat|The)=0.95, P(sat|The cat)=0.9, ...
  → Perplexity ≈ 1.05 (very confident)

Random model: P(any_word)=1/50000 for each position
  → Perplexity ≈ 50,000 (completely lost)

n-gram model: P(cat|The)=0.5, P(sat|The cat)=0.4, ...
  → Perplexity ≈ 5-20 (somewhat uncertain)
```

### Why perplexity matters

- It's **interpretable** — lower number = better predictor
- It's **comparable across models** — different architectures, same metric
- It **correlates with downstream performance** — models with better perplexity usually perform better on tasks like translation, summarisation, etc.

### Real numbers

```
Dataset: WikiText-103 (Wikipedia text)

GPT-2:   Perplexity ≈ 24
GPT-3:   Perplexity ≈ 20
BERT:    (language model variant) Perplexity ≈ 3-4
Claude:  (proprietary, exact number unknown) Likely ≈ 10-15

Your bigram model: Perplexity ≈ 5-20 on tiny corpus
  → Much worse than real LLMs because:
    - Tiny training corpus (few hundred words)
    - Limited context (only 2 previous words)
    - No learned parameters (just counts)
```

---

## 5. Smoothing & Backoff

The core problem with n-gram models: **unseen sequences**.

### The sparsity problem

With a 50,000-word vocabulary:
- **Unigrams:** 50,000 possible (manageable)
- **Bigrams:** 50,000² = 2.5B possible (most unseen)
- **Trigrams:** 50,000³ = 125T possible (virtually all unseen)

In a realistic corpus, most n-grams never appear. What's the probability of an unseen sequence?

```
Naive answer: P(unseen) = count / total = 0 / total = 0

Problem: log(0) = -∞, making perplexity infinite
Solution: We need smoothing
```

### Smoothing techniques

**Add-one smoothing** (Laplace smoothing):

```
P(word | context) = (count + 1) / (total + vocab_size)

Adds a small pseudocount to every sequence, never zero probability
```

**Backoff:**

```
If trigram not found, back off to bigram:
  P(word | w₋₂, w₋₁) = P(word | w₋₁) if trigram unseen

If bigram not found, back off to unigram:
  P(word | w₋₁) = P(word) if bigram unseen
```

Both are hacks. The real solution is **neural models** — learn dense representations that generalise to unseen contexts.

---

## 6. Neural Language Models

Instead of counting, use a neural network to predict the next word:

```
Input: Previous words (as embeddings)
       ↓
Hidden layers (RNN or Transformer)
       ↓
Output: Probability distribution over vocabulary
```

### RNN language model

```
"The cat sat" (as token IDs)
  ↓ Embedding lookup
[embed("the"), embed("cat"), embed("sat")]
  ↓ RNN processes sequence
  ↓ Output at position 3: probability distribution
  ↓ High probability for: "on", "in", "at", ...
  ↓ Low probability for: "meows", "quickly", ...
```

The RNN **remembers** the sequence (in theory) and uses it to predict.

### Transformer language model

```
"The cat sat" (as token IDs)
  ↓ Embedding lookup
[embed("the"), embed("cat"), embed("sat")]
  ↓ Attention: each position looks at all previous positions
  ↓ "sat" can directly see "the" and "cat" without bottleneck
  ↓ Output at position 3: probability distribution
```

The Transformer uses **attention** instead of recurrence, allowing each position to directly interact with all previous positions.

### Why neural wins

| Aspect | n-gram | Neural |
|---|---|---|
| **Generalisation** | Unseen sequences get zero (smoothing hack) | Learned representations generalise |
| **Context length** | Fixed (3 for trigram, 5 for 5-gram) | Flexible (up to sequence length) |
| **Parameters** | None learned (just counting) | Millions/billions learned |
| **Scalability** | Linear in vocab | Scales with data and compute |

---

## 7. Autoregressive Generation

Language models generate text **one token at a time** using the distribution they learned.

### Generation process

```
1. Start with prompt: "The cat"
2. Model predicts P(next_word | "The cat")
3. Sample/argmax to pick next word: "sat"
4. Now have: "The cat sat"
5. Model predicts P(next | "The cat sat")
6. Sample next word: "on"
7. Continue until stop token or max length
```

This is **autoregressive** — each token depends on all previous tokens. Every LLM works this way.

### Sampling vs argmax

**Greedy (argmax):**
```
Always pick the highest probability word
Result: deterministic, often repetitive
```

**Sampling:**
```
Sample from the probability distribution
Result: diverse, sometimes incoherent
```

**Temperature:** Control randomness
```
Low temperature (0.1): more greedy
High temperature (2.0): more random
```

Real LLMs use sophisticated techniques: top-k sampling, nucleus (top-p) sampling, etc. to balance quality and diversity.

---

## 8. The Sparsity Problem & Why Neural Models Win

The fundamental problem with n-grams:

```
Vocabulary size: 50,000
Trigram space: 50,000³ = 125 trillion possible sequences

Training corpus: 1 billion words
Trigrams observed: ~1 million (0.000001% of space)

Result: 99.999999% of trigrams are unseen
```

With sparse counts, you can't generalise:

```
Training: "The cat sat on the mat"
Test: "The dog sat on the floor"

Trigram (the, dog, sat): never seen in training
→ Probability = 0 (or smoothed-over guess)
→ Can't handle simple variations
```

### How neural models solve it

**Learned embeddings:** Words with similar meanings have similar vectors

```
embed("cat") ≈ embed("dog") ≈ embed("bird")
(all are animals, similar embeddings)
```

**Shared parameters:** The same neural network parameters apply to all words

```
Even if "dog" never appeared after "the", the network learned
from other sentences that: subject → verb patterns work
So it can predict reasonable verbs for new subjects
```

**Dense representations:** No sparsity

```
Every parameter is shared and learned
No "unseen" in the neural model sense — continuous space
```

This is why **neural language models revolutionised NLP** — they finally solved the sparsity problem.

---

## 9. Pre-training on Language Modelling

The single most powerful idea in modern NLP: **train a huge model to predict the next word on massive text, then use it for everything else.**

### The hypothesis

> "A model that predicts text well has learned language."

Evidence:
- **GPT-2** trained only on "predict next token" learns grammar, facts, coding, reasoning, translation
- **BERT** trained with masked language modelling learns semantic understanding
- **Scaling laws** show that perplexity correlates with downstream task performance

### Pre-training procedure

```
1. Take huge corpus (100GB+ of text)
2. Train model to minimise language modelling loss
3. Result: model has learned vast amounts about language

Why this works:
  - To predict well, model must understand grammar
  - Must learn facts about the world
  - Must learn reasoning patterns
  - Must learn how language encodes meaning
```

### Transfer learning

```
Pre-trained model:
  Embeddings learned on billions of tokens
  Attention patterns learned to parse structure
  Knowledge about facts and reasoning built in

Fine-tuning on specific task:
  Take pre-trained model
  Add task-specific layer
  Train on small supervised dataset
  Result: much better performance with less data
```

This is why **language models are the foundation of modern NLP** — they learn general language understanding that transfers to everything.

---

## 10. How This Connects to LLMs

| Stage | Method | Metric |
|---|---|---|
| **Early NLP** | n-gram models | Perplexity |
| **Neural era** | RNN/LSTM language models | Perplexity |
| **Modern** | Transformer language models | Perplexity |
| **LLMs** | Billion-parameter Transformers | Perplexity (+ downstream evals) |

### The LLM training pipeline

```
1. Pre-training:
   Objective: Minimise language modelling loss
   Data: Trillions of tokens from web, books, code, etc.
   Model: GPT-style Transformer with billions of parameters
   Result: Foundation model with broad knowledge

2. Fine-tuning (SFT):
   Objective: Predict high-quality next tokens on curated data
   Data: High-quality human-written examples
   Model: Same foundation model + light training
   Result: Model follows instructions better

3. RLHF:
   Objective: Optimise for human preferences (beyond accuracy)
   Data: Human feedback on outputs
   Model: Same foundation model + RL training
   Result: Model is helpful, harmless, honest

4. Inference:
   Objective: Generate coherent, helpful text
   Method: Autoregressive sampling from learned distribution
   Result: You get an answer
```

### Everything traces back to language modelling

```
Why is GPT smart?
  → It learned by predicting billions of tokens

Why can it do tasks it wasn't explicitly trained on (in-context learning)?
  → It learned generalised patterns about language and reasoning
  → Language modelling objective forces this generalisation

Why does more scale (bigger model, more data) improve performance?
  → Scaling laws in language modelling are phenomenal
  → Perplexity improves smoothly with scale (empirically)

Why does pre-training help so much?
  → Language modelling on huge text teaches general understanding
  → Transfers to any downstream task
```

### Summary

Every LLM you've used is:
1. **Trained** on language modelling (predict next token)
2. **Evaluated** using perplexity (lower is better)
3. **Generated text** using autoregressive sampling
4. **Capable** because language modelling forced it to learn language

The entire modern LLM revolution is built on the simple idea: **predict the next word well, and you've learned language.**

---

*Phase 1 — concept 05 → you are here*  
*Previous concept → [04 — Word Embeddings: Word2Vec, GloVe](./04_word_embeddings.md)*  
*Next → Phase 1 Capstone → [06 — Text Classification & Sentiment Analyser Project](../02_demo_sentiment_analyser/README.md)*
