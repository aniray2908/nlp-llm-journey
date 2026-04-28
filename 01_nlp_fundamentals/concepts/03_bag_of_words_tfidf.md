# Bag of Words & TF-IDF Representations

> **Path:** `01_nlp_fundamentals/concepts/`  
> **Topic:** Converting text into numerical vectors. Understanding two foundational text representation techniques that powered NLP before deep learning, and why they're still useful today.

---

## Table of Contents

1. [Why Represent Text as Numbers?](#1-why-represent-text-as-numbers)
2. [Bag of Words (BoW)](#2-bag-of-words-bow)
3. [Sparsity & High Dimensionality](#3-sparsity--high-dimensionality)
4. [Document Similarity](#4-document-similarity)
5. [TF-IDF — Smarter Weighting](#5-tf-idf----smarter-weighting)
6. [BoW vs TF-IDF](#6-bow-vs-tf-idf)
7. [Text Classification with BoW](#7-text-classification-with-bow)
8. [Document Clustering](#8-document-clustering)
9. [Limitations & When to Use](#9-limitations--when-to-use)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. Why Represent Text as Numbers?

Models need numbers. Before embeddings and Transformers, the standard approach was:

```
Raw text  →  Numerical vector  →  Model (classifier, clusterer, etc.)
```

Two foundational techniques for this conversion are **Bag of Words** and **TF-IDF**.

---

## 2. Bag of Words (BoW)

**Bag of Words** is the simplest text representation: count how many times each word appears in a document, ignoring order.

### How it works

```python
vocabulary = {the: 0, cat: 1, sat: 2, on: 3, mat: 4}

document = "the cat sat on the mat"

# Count each word
BoW vector = [2, 1, 1, 1, 1]
             (the=2, cat=1, sat=1, on=1, mat=1)
```

The document becomes a **sparse vector** — mostly zeros, with counts at word positions.

### More formally

Given a vocabulary of *n* unique words and a document *d*:

```
BoW(d) = [count(word_1), count(word_2), ..., count(word_n)]
```

The vector has length *n* (vocabulary size) and value at position *i* is the number of times word *i* appears in the document.

### Example with real documents

```python
documents = [
    "the cat sat on the mat",
    "the dog played in the park",
]

vocabulary: {and, cat, dog, in, mat, on, park, played, sat, the}
            (alphabetically: 10 unique words)

Doc 0: [0, 1, 0, 0, 1, 1, 0, 0, 1, 2]
       (cat=1, mat=1, on=1, sat=1, the=2)

Doc 1: [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
       (dog=1, in=1, park=1, played=1, the=1)
```

### Why "Bag of Words"?

The name is literal — you throw all the words into a bag, count them, and forget the order. "the cat sat" and "sat cat the" have the **same BoW vector**. This is both a feature (robustness to word order) and a major limitation (loses sequential information like "not good" vs "good not").

---

## 3. Sparsity & High Dimensionality

### The sparsity problem

Real text has a huge vocabulary. English has ~170,000 words. Any single document uses a tiny fraction of them.

```
4 documents, 17 unique words
→ BoW matrix: 4 rows × 17 columns
→ Most entries are zero

Sparsity = (zeros) / (total entries) ≈ 90%
```

Most BoW matrices are **90%+ zeros**. This wastes memory and computation.

### High dimensionality

With a 50,000-word vocabulary, each document becomes a 50,000-dimensional vector. Most dimensions are zero, but you still have to store and compute with them.

### Why this matters

- **Memory** — storing dense versions of sparse matrices is wasteful
- **Computation** — many ML algorithms are slow in very high dimensions (curse of dimensionality)
- **Generalisation** — models can struggle with so many features

This is why **dimensionality reduction** (PCA, SVD) and **sparse data structures** (scipy.sparse) are important in classical NLP.

---

## 4. Document Similarity

BoW vectors enable a natural notion of **similarity** via **cosine similarity**.

### Cosine similarity

The cosine of the angle between two vectors. Ranges from -1 (opposite) to 1 (identical).

```
similarity = (doc1 · doc2) / (||doc1|| × ||doc2||)
```

For text, cosine similarity between 0 and 1 — it measures the angle, not magnitude, so document length doesn't matter.

### Example

```
Doc A: "the cat sat on the mat"        → [0, 1, 0, 0, 1, 1, 0, 0, 1, 2]
Doc B: "the dog played in the park"    → [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
Doc C: "the cat and the dog"           → [1, 1, 1, 0, 0, 0, 0, 0, 0, 2]

Similarity(A, B) = low   (different words)
Similarity(A, C) = high  (share "cat", "the", "dog")
```

Two documents with many overlapping words get high similarity. This is the foundation for:
- Information retrieval / search
- Duplicate detection
- Document clustering

---

## 5. TF-IDF — Smarter Weighting

BoW treats all words equally: "the" (appears everywhere) gets the same weight as "transformer" (appears rarely).

**TF-IDF** fixes this: weight words by **how rare they are** across the entire corpus.

### The formula

```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

where:
  TF = term frequency = count of word in document
  IDF = inverse document frequency = log(total_docs / docs_containing_word)
```

### Intuition

- Common words (appear in many docs) → low IDF → low weight
- Rare words (appear in few docs) → high IDF → high weight
- Frequent in a specific doc → high TF → high weight

```
Example: 1000 documents

Word: "the"
  - Appears in 999 documents
  - IDF = log(1000 / 999) ≈ 0.001  (nearly 0 weight)

Word: "transformer"
  - Appears in 5 documents
  - IDF = log(1000 / 5) ≈ 5.3  (high weight)
```

### TF-IDF vector

Each document becomes a vector where **common words are damped and rare words are amplified**.

```
Doc A: "the cat sat on the mat"

BoW:     [2, 1, 1, 1, 1]   (raw counts)
TF-IDF:  [0.1, 0.8, 0.7, 0.6, 0.8]  (weighted by rarity)
```

The common word "the" went from 2 → 0.1. Content words stay high.

---

## 6. BoW vs TF-IDF

| Aspect | BoW | TF-IDF |
|---|---|---|
| **Weighting** | Raw count | Weighted by rarity |
| **Common words** | High weight | Low weight |
| **Rare words** | Normal weight | High weight |
| **Similarity** | All words equal | Informative words matter more |
| **Use case** | Simple baseline | Better for most tasks |
| **Interpretability** | Easy — just counts | Need to understand IDF |

### When to choose

**Use BoW when:**
- You want a simple, fast baseline
- All words are equally important (rare)
- You have very little data

**Use TF-IDF when:**
- You want to weight by informativeness
- Common words should matter less
- You're doing information retrieval or classification

For most practical tasks, **TF-IDF beats BoW**. But both are now **outperformed by embeddings** (Word2Vec, GloVe) and **Transformers** (BERT, GPT).

---

## 7. Text Classification with BoW

BoW vectors feed directly into simple classifiers like **Naive Bayes** or **SVM**.

### Pipeline

```
Raw text
  ↓  BoW vectorisation
Vector (1 × vocab_size)
  ↓  Naive Bayes / SVM / Logistic Regression
Class prediction
```

### Example: Sentiment analysis

```python
training_docs = [
    ("I love this movie", 1),      # positive
    ("Terrible waste of time", 0), # negative
    ("Amazing film!", 1),
    ("Complete garbage", 0),
]

Vectorise each → BoW vectors
Train Naive Bayes on (vector, label) pairs

Test on new doc: "This is great!"
→ BoW vector [0, 1, 0, 0, 1, ...]
→ Naive Bayes → probability(positive) = 0.92 → predict positive ✓
```

### Why it works

Even though BoW throws away word order, **sentiment words often predict the label**:
- Positive: "love", "amazing", "great", "excellent"
- Negative: "hate", "terrible", "bad", "garbage"

These words *independently* signal sentiment — order doesn't matter much.

### Limitations

Fails on **negation and intensity**:

```
"I love this movie"      → positive ✓
"I hate this movie"      → negative ✓
"I don't love this"      → should be negative, but sees "love" 🔴
"I really love this"     → positive (correct) but only by accident
"I sort of like this"    → positive (correct) but BoW doesn't understand "sort of"
```

---

## 8. Document Clustering

BoW vectors enable **unsupervised learning** — grouping similar documents without labels.

### KMeans + BoW/TF-IDF

```
Documents with BoW vectors
  ↓  TF-IDF weighting
Weighted vectors
  ↓  KMeans clustering (k=3)
Cluster assignments
```

No labels needed. The algorithm finds documents with similar word distributions and groups them.

### Example

```
Documents:
1. "cats are cute and fluffy"
2. "kittens are adorable pets"
3. "dogs love to play fetch"
4. "puppies are loyal and fun"

KMeans groups into:
Cluster A: docs 1,2 (cat-related)
Cluster B: docs 3,4 (dog-related)

Without any labels, pure word similarity created semantic clusters.
```

This is powerful for **unsupervised exploratory analysis** — discovering topics in a corpus without annotation.

---

## 9. Limitations & When to Use

### Major limitations

| Limitation | Impact |
|---|---|
| **Ignores word order** | "good not" and "not good" are identical |
| **Loses context** | Word meaning depends on surrounding words (BoW doesn't see it) |
| **Sparse & high-dimensional** | 50k vocab = 50k dimensions, 90%+ zeros |
| **Can't handle synonymy** | "good" and "excellent" are separate vocabulary entries |
| **Treats all words equally** (BoW) | "the" weighted same as "transformer" |

### Where BoW/TF-IDF still win

- **Sparse data** — classical ML with BoW works with very little training data
- **Interpretability** — you can see which words matter (just look at non-zero entries)
- **Speed** — BoW vectorisation is fast; Naive Bayes training is fast
- **Search & IR** — TF-IDF is still the baseline for search relevance ranking
- **Simple tasks** — sentiment, spam detection on clean data

### When to skip them

- **Deep learning tasks** — use embeddings (Word2Vec) or Transformers (BERT)
- **Complex semantics** — need contextual understanding
- **Modern NLP** — embeddings and neural networks are strictly better
- **LLMs** — BPE tokenisation + learned embeddings handle everything

---

## 10. How This Connects to LLMs

| Concept | Classical NLP | Modern LLMs |
|---|---|---|
| **Text representation** | BoW/TF-IDF vectors | Learned embeddings |
| **Vocabulary** | Fixed list (50k words) | BPE tokens (100k) |
| **Weighting** | TF-IDF (statistical) | Embedding weights (learned) |
| **Similarity** | Cosine on BoW | Cosine on embeddings |
| **Context** | None — document-level | Full — entire context window |

### The evolution

```
1990s-2010s: BoW/TF-IDF + Naive Bayes
    ↓
2010s: Word embeddings (Word2Vec, GloVe) + RNNs
    ↓
2015+: Transformers + contextual embeddings (BERT, GPT)
    ↓
2020+: Large language models (GPT-3, GPT-4)
```

Each step solved limitations of the previous:
- **BoW problem:** Ignores word order and context
- **Embeddings solution:** Dense vectors capture semantic similarity
- **Embedding problem:** No context — same word in different sentences gets same vector
- **Transformer solution:** Contextual embeddings — word representation depends on surrounding tokens
- **Transformer problem:** Limited by sequence length, single forward pass
- **LLM solution:** Huge models, autoregressive generation, in-context learning

### Why understand BoW/TF-IDF?

They're **conceptually foundational**:
- Teach what "representing text as numbers" means
- Show the core trade-off: simple & interpretable vs powerful & opaque
- Still useful as baselines and for certain tasks
- Help you understand why modern methods are improvements

LLMs are trained on the same principles — embeddings are just learned BoW-style vectors, attention is a sophisticated version of cosine similarity, and the vocabulary is learned via BPE instead of hand-picked.

---

## Summary

**Bag of Words** — count words, ignore order. Simple, interpretable, foundational.

**TF-IDF** — weight words by rarity. Better than BoW for most classical NLP tasks.

**Both are largely superseded** by embeddings and Transformers, which handle context and learn representations end-to-end.

**But understanding them is essential** — they're the conceptual ancestors of modern NLP, and knowing where we came from shows why modern methods are improvements.

---

*Phase 1 — concept 03 → you are here*  
*Previous concept → [02 — Text Normalisation](./02_text_normalisation.md)*  
*Next concept → [04 — Word Embeddings: Word2Vec, GloVe](./04_word_embeddings.md)*
