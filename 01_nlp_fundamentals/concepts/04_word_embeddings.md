# Word Embeddings — Word2Vec, GloVe

> **Path:** `01_nlp_fundamentals/concepts/`  
> **Topic:** Dense vector representations of words where meaning is captured in geometry. The bridge from sparse BoW to contextual embeddings.

---

## Table of Contents

1. [The Problem with BoW](#1-the-problem-with-bow)
2. [What Are Word Embeddings?](#2-what-are-word-embeddings)
3. [How Embeddings Capture Semantics](#3-how-embeddings-capture-semantics)
4. [Word2Vec — Skip-gram and CBOW](#4-word2vec----skip-gram-and-cbow)
5. [GloVe — Global Vectors](#5-glove----global-vectors)
6. [Semantic Properties](#6-semantic-properties)
7. [Vector Arithmetic & Analogies](#7-vector-arithmetic--analogies)
8. [The Polysemy Problem](#8-the-polysemy-problem)
9. [Using Pre-trained Embeddings](#9-using-pre-trained-embeddings)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. The Problem with BoW

Bag of Words is sparse and high-dimensional:

| Problem | Impact |
|---|---|
| **50,000+ vocabulary** | 50,000 dimensions per document |
| **90%+ sparsity** | Mostly zeros |
| **No semantics** | "good" and "great" are completely different |
| **No synonymy** | Synonyms get separate vocabulary entries |
| **No relatedness** | Nothing captures that "cat" and "kitten" are similar |

BoW counts words but captures **no semantic meaning**. Two documents with synonyms get zero similarity.

---

## 2. What Are Word Embeddings?

Instead of counting words, **represent each word as a dense vector** where **similar words have similar vectors**.

```
BoW:       "king" → [0, 0, 1, 0, 0, 0, ..., 0]  (50,000-dim, sparse)
Embedding: "king" → [0.2, -0.5, 0.8, 0.1, ...]  (300-dim, dense)
```

### Key properties

- **Dense** — all dimensions are non-zero (or mostly non-zero)
- **Low-dimensional** — typically 50–1536 dimensions (vs 50,000+ for BoW)
- **Learned** — discovered by training on raw text
- **Semantic** — similar words cluster together in vector space

### Example

```
vector("king") ≈ [0.2, -0.5, 0.8, ...]
vector("queen") ≈ [0.25, -0.48, 0.75, ...]  ← very similar to king

vector("dog") ≈ [0.1, 0.3, -0.2, ...]       ← different from king
```

The vectors for "king" and "queen" are **close in space** because they mean similar things. The vectors for "king" and "dog" are **far apart**.

---

## 3. How Embeddings Capture Semantics

The core insight: **words that appear in similar contexts should have similar vectors**.

```
Example sentences:
"The king sat on his throne"
"The queen sat on her throne"
"The dog sat on the ground"

"king" and "queen" both appear before "sat on" → similar context → similar vectors
"dog" appears in different context → different vector
```

During training, the model **adjusts vectors** to make this work:
- If two words appear in similar contexts → push their vectors closer together
- If they don't → push them apart

The model learns this **purely from text patterns**, with **no labels or semantic information provided**.

---

## 4. Word2Vec — Skip-gram and CBOW

**Word2Vec** is a two-layer neural network trained on raw text. Two architectures:

### Skip-gram

**Predict context words from target word.**

```
Input: "the cat sat on the mat"

Training example:
  Target: "cat"
  Context (nearby words): ["the", "sat"]
  Model learns: cat → predicts "the" and "sat"

Another example:
  Target: "sat"
  Context: ["cat", "on"]
  Model learns: sat → predicts "cat" and "on"
```

The model trains to make the **dot product of similar words large**:

```
score(word_i, word_j) = vector(word_i) · vector(word_j)

If word_i and word_j co-occur:
  score should be large ✓

If they don't:
  score should be small ✓
```

The **hidden layer weights** become the embeddings — each word's row in the weight matrix is its vector representation.

### CBOW (Continuous Bag of Words)

The reverse: **predict target word from context words.**

```
Input: "the cat sat on the mat"

Training example:
  Context: ["the", "sat"]
  Target: "cat"
  Model learns: [the, sat] → predicts "cat"
```

Both skip-gram and CBOW learn similar embeddings, just through different prediction directions. Skip-gram typically works better.

### Why this works

The model doesn't know what "king" and "queen" mean. It just sees that they appear in similar contexts — same surrounding words. By making their vectors similar, it implicitly captures that they mean related things. **This is pure statistics becoming semantics.**

---

## 5. GloVe — Global Vectors

**GloVe** (Global Vectors for Word Representation) combines two ideas:

1. **Global statistics** — word co-occurrence counts across entire corpus
2. **Local context** — nearby words in small windows

### How it works

Pre-compute a **co-occurrence matrix**: for each pair of words, count how many times they appear together in the corpus.

```
"the" appears with "cat" 1000 times
"the" appears with "dog" 950 times
"cat" appears with "dog" 100 times
```

Then train embeddings to **reproduce the statistics** of this matrix:

```
vector(word_i) · vector(word_j) ≈ log(co-occurrence(word_i, word_j))
```

The vectors learn to make frequent co-occurrences have large dot products, rare ones have small dot products.

### GloVe vs Word2Vec

| Aspect | Word2Vec | GloVe |
|---|---|---|
| **Training** | Predict context (local) | Reproduce statistics (global + local) |
| **Speed** | Fast | Slightly slower |
| **Hyperparameters** | Context window | Context window + matrix weighting |
| **Results** | Excellent | Slightly better on some tasks |

In practice, both produce very similar embeddings. GloVe has a slight theoretical edge (uses global stats) but differences are marginal.

---

## 6. Semantic Properties

Once trained, embeddings capture linguistic relationships **without being told to**:

### Analogies

```
vector("Paris") - vector("France") + vector("Germany") ≈ vector("Berlin")

Intuition: The difference between Paris and France captures
"capital-of-country" relationship. Add that relationship to
Germany and you get its capital.
```

### Word relationships

```
vector("king") - vector("man") ≈ vector("queen") - vector("woman")

Both differences capture the "royalty" concept. The gender relationship
is orthogonal — you can add/subtract gender vectors independently.
```

### Clustering

```
Words near vector("king"):
  queen (0.76)
  prince (0.74)
  royal (0.71)
  monarchy (0.68)

All relate to royalty — the model learned this purely from co-occurrence.
```

### What dimensions mean

Unlike BoW where dimension *i* means "count of word *i*", embedding dimensions are **emergent and hard to interpret**:

- Dimension 0 might capture "gender" (somewhat)
- Dimension 5 might capture "royalty" (somewhat)
- But dimensions are mixed and interdependent

This is the **interpretability tradeoff** — dense embeddings are powerful but opaque.

---

## 7. Vector Arithmetic & Analogies

The most famous property: **vector arithmetic has meaning**.

### The king-queen-man-woman analogy

```
king - man + woman ≈ queen

Mathematically:
  vector(king) - vector(man) + vector(woman)

This sum is a vector. We find the word in the vocabulary
closest to this vector. Answer: queen (0.79 similarity)
```

### Why this works

- `king` and `queen` share similar non-gender aspects
- `man` and `woman` differ mainly on gender
- `king - man` removes the "man" aspects, leaving "royalty"
- `+ woman` adds back the gender, resulting in the female royalty

### Generality

Similar arithmetic works for many relationships:

```
Paris - France + Germany ≈ Berlin     (capitals)
bad - good + better ≈ worse            (comparatives)
run - running + walk ≈ walking         (verb conjugation)
```

These emerge purely from training on text — the model was never told about analogies.

---

## 8. The Polysemy Problem

**Word2Vec/GloVe are context-free** — each word gets one vector, no matter how many meanings it has.

### The problem

```
"bank" has two meanings:
  - Financial institution: "deposit money in the bank"
  - River edge: "sit by the river bank"

Word2Vec gives it ONE vector.
So the vector is a mixture of both meanings.

Most similar words:
  savings, credit, deposit  (financial)
  river, flow               (geographic)

The vector can't distinguish — it's confused between meanings.
```

### Why this matters

This is a fundamental limitation. Word2Vec can capture that "bank" is related to both finance and rivers, but **can't represent the meanings separately**.

### The solution

**Contextual embeddings** (BERT, GPT, etc.) give different vectors depending on context:

```
"I deposited money in the bank" 
  → bank = [0.1, 0.8, -0.2, ...]  (financial meaning)

"We walked along the river bank"
  → bank = [-0.3, 0.2, 0.7, ...]  (geographic meaning)

Same word, different vectors, different meanings captured.
```

This is a major leap in modern NLP — Transformers solve the polysemy problem.

---

## 9. Using Pre-trained Embeddings

You don't train Word2Vec from scratch. Instead, download **pre-trained embeddings** trained on billions of words:

```python
from gensim.models import KeyedVectors

# Load pre-trained
word_vectors = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True
)

# Look up a word
vector = word_vectors['king']  # 300-dimensional numpy array

# Find similar words
word_vectors.most_similar('king')
# → [('queen', 0.76), ('prince', 0.74), ...]

# Solve analogies
word_vectors.most_similar(positive=['king', 'woman'], negative=['man'])
# → [('queen', 0.79), ...]
```

### Pre-trained sources

| Source | Dimensions | Training data | Notes |
|---|---|---|---|
| Google News Word2Vec | 300 | Google News (3B words) | The classic |
| GloVe Wiki | 100–300 | Wikipedia + Gigaword | Good alternative |
| FastText | 300 | CommonCrawl | Handles out-of-vocab via subwords |

### Using in neural networks

```python
import torch.nn as nn

embedding_layer = nn.Embedding(
    num_embeddings=50000,   # vocabulary size
    embedding_dim=300       # dimension of each embedding
)

# Initialize with pre-trained weights
pretrained_weights = torch.tensor(word_vectors.vectors)
embedding_layer.weight.data.copy_(pretrained_weights)

# Freeze (don't update) or fine-tune (update slowly)
embedding_layer.weight.requires_grad = False  # freeze
# or
embedding_layer.weight.requires_grad = True   # fine-tune
```

This is exactly how **Transformers start** — a lookup table that maps token IDs to learned vectors.

---

## 10. How This Connects to LLMs

| Stage | Technique | Strength | Limitation |
|---|---|---|---|
| **BoW** | Word counts | Interpretable | No semantics, sparse |
| **Word2Vec/GloVe** | Fixed embeddings | Semantic similarity, dense | Context-free, polysemy |
| **BERT/GPT** | Contextual embeddings | Context-dependent meaning | Less interpretable |
| **LLMs** | Learned embeddings at scale | Everything | Complex to understand |

### The evolution

```
BoW: "good" and "great" are different
  ↓
Word2Vec: "good" and "great" are similar
  ↓
BERT: "good" and "great" cluster differently depending on context
  ↓
GPT: Learns embeddings while learning language, reasoning, knowledge
```

### LLMs use embeddings the same way

Every LLM starts with an **embedding layer** — token IDs to vectors. This layer is **learned during pre-training** rather than pre-computed on external data.

```
Token ID (e.g., 15496 for "hello")
  ↓  nn.Embedding lookup
Dense vector (768 or 1536 dims)
  ↓  + positional encoding
  ↓  Transformer blocks × N
  ↓  Output embeddings
Logits over vocabulary
```

The embedding space in GPT-4 is learned on trillions of tokens. The geometry captures **everything** — semantics, syntax, facts, reasoning. This is why attention over embeddings (the key mechanism of Transformers) is so powerful.

---

## Summary

**Word embeddings revolutionised NLP** by showing that **statistical patterns in text encode semantic meaning**. Training a model to predict context words forces it to learn that similar-context words should have similar vectors. This is pure geometry becoming meaning.

Word2Vec and GloVe have been superseded by contextual embeddings (Transformers), which solve the polysemy problem by generating different vectors depending on context. But the core idea — **dense vectors where similarity = semantic relatedness** — remains central to all modern NLP.

Understanding embeddings is understanding the foundation that everything modern NLP is built on.

---

*Phase 1 — concept 04 → you are here*  
*Previous concept → [03 — Bag of Words and TF-IDF](./03_bag_of_words_tfidf.md)*  
*Next concept → [05 — Language Modelling Basics: n-grams, Perplexity](./05_language_modelling.md)*
