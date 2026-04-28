# Text Normalisation — Stemming, Lemmatisation, Stopwords

> **Path:** `01_nlp_fundamentals/concepts/`  
> **Topic:** Cleaning and standardising text before it enters a model. Understanding what classical NLP pipelines do, and why modern deep learning often skips these steps.

---

## Table of Contents

1. [Why Text Normalisation Exists](#1-why-text-normalisation-exists)
2. [Lowercasing & Whitespace](#2-lowercasing--whitespace)
3. [Punctuation & Special Characters](#3-punctuation--special-characters)
4. [Stemming](#4-stemming)
5. [Lemmatisation](#5-lemmatisation)
6. [Stemming vs Lemmatisation](#6-stemming-vs-lemmatisation)
7. [Stopword Removal](#7-stopword-removal)
8. [Complete Normalisation Pipeline](#8-complete-normalisation-pipeline)
9. [When NOT to Normalise](#9-when-not-to-normalise)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. Why Text Normalisation Exists

Raw text from the wild is messy and inconsistent:

```
"I LOVE ChatGPT!!! It's AMAZING..."
"The quick BROWN fox jumps over the lazy DOG."
"don't  worry   about  spacing"
```

Without normalisation, the same concept gets represented multiple ways:

```
vocab = {"LOVE": 0, "love": 1, "loving": 2, "loves": 3}
```

Four separate vocabulary entries for variants of the same root. Wasteful and the model doesn't recognise the connection.

Text normalisation makes text **consistent and removes noise**, allowing the model to focus on meaning rather than surface variations.

---

## 2. Lowercasing & Whitespace

The two simplest and most universally applied steps.

### Lowercasing

```python
text = "The Quick BROWN Fox"
text = text.lower()
# → "the quick brown fox"
```

**Pros:**
- Reduces vocabulary — "The" and "the" become one token
- Most common approach — standard in pre-2015 NLP

**Cons:**
- Loses information — "USA" and "usa" might mean different things
- Proper nouns — "Paris" (city) vs "paris" (lowercase) can matter
- Acronyms — "CNN" (news network) vs "cnn" (completely different meaning depending on context)

**Modern practice:** Some models preserve case; many lowercase everything. For LLMs, case is often preserved because the model learns what case patterns matter.

### Whitespace normalisation

```python
text = "don't  worry   about  spacing"
text = ' '.join(text.split())
# → "don't worry about spacing"
```

**Always do this.** Multiple spaces, tabs, and newlines are noise. Normalise to single spaces.

---

## 3. Punctuation & Special Characters

```python
import re

text = "Hello, world! How are you?"
text = re.sub(r'[^\w\s]', '', text)
# → "Hello world How are you"
```

**Pros:**
- Reduces vocabulary — punctuation becomes noise
- Faster processing
- Common in classical NLP

**Cons:**
- Loses information — "don't" becomes "dont" (loses structure)
- Sentiment markers gone — "!!!" conveys excitement, now it's lost
- Emojis — critical for modern social media text, completely removed

**Modern practice:** Keep punctuation. Tokenisers handle it. Models learn that "!" and "?" are meaningful.

---

## 4. Stemming

**Stemming** is a simple rule-based approach: chop off common suffixes and prefixes to reduce words to a root form (the "stem").

### How it works

The **Porter Stemmer** applies a cascade of rules:

```
SSES → SS     (caresses → caress)
IES → I       (ponies → poni)
ATIONAL → ATE (relational → relate)
...
```

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runs", "ran", "runner"]
for w in words:
    print(f"{w:12} → {stemmer.stem(w)}")

# running      → run
# runs         → run
# ran          → ran       ← irregular verb, not stemmed
# runner       → runner    ← different suffix, not stemmed
```

### The problems

| Problem | Example | Impact |
|---|---|---|
| **Non-words** | "singular" → "singul" | Output isn't a real word |
| **Irregular verbs** | "ran" stays "ran" | Misses some variants |
| **Over-stemming** | "conflated" → "conflat" | Removes too much |
| **Under-stemming** | "runner" stays "runner" | Misses variants |

### Fast but imperfect

Stemming is **fast** (no dictionary needed) but **inconsistent and brittle**. Good for 1990s search engines, rarely used in modern NLP.

---

## 5. Lemmatisation

**Lemmatisation** is smarter: use a dictionary and linguistic knowledge to reduce words to their **lemma** — the canonical base form that appears in the dictionary.

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "runner"]
for w in words:
    print(f"{w:12} → {lemmatizer.lemmatize(w, pos='v')}")

# running      → run
# runs         → run
# ran          → run        ← handles irregular verbs!
# runner       → runner     ← noun, stays as is
```

### Part of speech (POS) matters

The same word can have different lemmas:

```python
lemmatizer.lemmatize("running", pos='v')  # verb   → "run"
lemmatizer.lemmatize("running", pos='n')  # noun   → "running"

lemmatizer.lemmatize("better", pos='v')   # verb   → "bet"
lemmatizer.lemmatize("better", pos='a')   # adj    → "good"  ← irregular!
```

Lemmatisation uses a dictionary, so it:
- **Handles irregular verbs** — "ran" → "run" (not possible with rules alone)
- **Produces real words** — every lemma exists in the dictionary
- **Respects morphology** — understands stem + suffix structure

But it requires linguistic knowledge and is **slower** than stemming.

---

## 6. Stemming vs Lemmatisation

| Aspect | Stemming | Lemmatisation |
|---|---|---|
| **Method** | Rule-based suffix/prefix removal | Dictionary + linguistic rules |
| **Speed** | Fast | Slower (lookups) |
| **Accuracy** | Lower; misses irregulars | Higher; handles exceptions |
| **Output** | Stem (often not a real word) | Lemma (always a real word) |
| **Example: "better"** | "better" (no change) | "good" (irregular!) |
| **Example: "were"** | "were" (no change) | "be" (correct) |
| **Modern use** | Rarely | Occasionally, mostly legacy |

**For modern NLP:** Lemmatisation is technically superior but often not worth the complexity. Transformers with embeddings handle word variants automatically.

---

## 7. Stopword Removal

**Stopwords** are common function words with little semantic content: "the", "a", "is", "in", "and", "or", "but", etc.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

text = "The quick brown fox jumps over the lazy dog"
filtered = ' '.join([w for w in text.lower().split() if w not in stop_words])

# → "quick brown fox jumps lazy dog"
```

### When it helps

**Information retrieval (search)** — "the quick brown fox" → "quick brown fox" focuses on content words.

**Bag-of-words classifiers** — reduces noise when counting word frequencies.

**Some text statistics** — when measuring vocabulary size or diversity.

### When it hurts

**Sentiment analysis:**
```
"I love this movie"   → "love movie"
"I do not love this movie" → "love movie"  ← negation lost!
```

**Negation and intensity:**
```
"not good"  → "good"     ← opposite meaning!
"very bad"  → "bad"      ← intensity lost
"don't want" → "want"    ← flips the meaning
```

**Language models:**
```
Modern models see all tokens. Removing "the" changes the distribution
the model is learning from. It's usually better to just let the model
learn that "the" is common and low-signal.
```

### The critical failure cases

Your demo showed exactly this:

```
"I do not love this movie"  →  "love movie"
"This is not good"          →  "good"
```

Both become positive when they should be negative. **Stopword removal breaks sentiment analysis.**

---

## 8. Complete Normalisation Pipeline

In practice, you combine multiple steps:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def normalise_text(text, lowercase=True, remove_special=True,
                   remove_stopwords=False, lemmatise=False):
    """
    Complete text normalisation pipeline.
    
    Args:
        text: input text
        lowercase: convert to lowercase
        remove_special: remove punctuation and numbers
        remove_stopwords: filter common words
        lemmatise: reduce to base forms
    
    Returns:
        list of normalised tokens
    """
    
    # 1. Lowercase
    if lowercase:
        text = text.lower()
    
    # 2. Normalise whitespace
    text = ' '.join(text.split())
    
    # 3. Remove special characters
    if remove_special:
        text = re.sub(r'[^a-z\s]', '', text) if lowercase else re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenise
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    
    # 6. Lemmatise
    if lemmatise:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    
    return tokens
```

### Example outputs from your demo

```
Original: "I LOVE ChatGPT!!! It's AMAZING... Running tests now!"

Raw tokenisation
→ ['i', 'love', 'chatgpt', '!', '!', '!', 'it', "'s", 'amazing', '...', 'running', 'tests', 'now', '!']

Clean + lowercase
→ ['i', 'love', 'chatgpt', 'its', 'amazing', 'running', 'tests', 'now']

No stopwords
→ ['love', 'chatgpt', 'amazing', 'running', 'tests']

Full pipeline (+ lemmatisation)
→ ['love', 'chatgpt', 'amaze', 'run', 'test']
```

Each step removes information. The question is: *is that information useful for your task?*

---

## 9. When NOT to Normalise

This is critical for modern NLP.

### Deep learning models

Modern neural networks (especially Transformers) learn better with **raw data**. They automatically learn:
- Which words are common (and therefore low-signal)
- Which punctuation matters
- How case carries information

Removing information early forces the model to learn with handicaps.

### Language models specifically

**For LLMs:**
1. **Do** lowercase (often)
2. **Do** normalise whitespace
3. **Do** tokenise with BPE

**Don't:**
- Remove punctuation — models need to predict and see it
- Remove stopwords — changes the natural language distribution
- Stem/lemmatise — BPE handles word variants automatically

### Sentiment and negation tasks

Never remove stopwords before sentiment analysis. The model needs to see "not", "very", "no", "but" — these carry semantic weight.

### Multilingual text

Stemming and lemmatisation are **English-specific**. For other languages you'd need language-specific stemmers, which is expensive. Better to skip it and let the model handle it.

---

## 10. How This Connects to LLMs

| Concept | Modern LLM practice |
|---|---|
| **Stemming** | Not used. BPE subword tokens handle morphology. |
| **Lemmatisation** | Not used. Embeddings learn that variants are similar. |
| **Stopword removal** | Not used. Model learns what's signal and what's noise. |
| **Lowercasing** | Sometimes — depends on the model. Preserving case can help. |
| **Whitespace normalisation** | Always. Multiple spaces are noise. |
| **Punctuation removal** | Never. Models need to predict punctuation. |

### The full LLM pipeline

```
Raw text
    ↓
Normalise whitespace (only)
    ↓  tokenisation (BPE)
Token IDs [9906, 1917, 0, ...]
    ↓  nn.Embedding lookup
Token vectors shape (seq_len, embed_dim)
    ↓  + positional encoding
    ↓  Transformer blocks × N
    ↓  linear projection
Logits over vocabulary shape (seq_len, vocab_size)
```

Classical NLP had heavy preprocessing. Modern deep learning **learns to handle these things automatically** — you just need to give it the raw signal.

### Why classical vs modern differ

**Classical NLP (pre-2015):**
- Limited vocabulary, sparse features
- Needed aggressive preprocessing to reduce noise
- Rule-based or simple statistical models
- "Reduce the problem to data the model can handle"

**Deep Learning (2015+):**
- Learned embeddings, dense representations
- Can handle raw, messy data
- Billions of parameters to learn from signal
- "Give the model signal and let it learn what matters"

This is why your demo showed that on a tiny sentiment dataset, stopword removal didn't help — the model can already learn to ignore common words. On larger data, the gap widens: removing information always hurts capacity.

---

## Summary — Three Rules

1. **Always normalise whitespace.** Multiple spaces are always noise.

2. **Never remove stopwords for deep learning.** Let the model learn. Only use it for classical bag-of-words or search.

3. **Don't stem or lemmatise for Transformers.** BPE and embeddings handle word variants automatically and better than rules ever could.

For LLMs: clean whitespace, tokenise with BPE, and let the model do the rest.

---

*Phase 1 — concept 02 → you are here*  
*Previous concept → [01 — Tokenisation](./01_tokenisation.md)*  
*Next concept → [03 — Bag of Words and TF-IDF](./03_bag_of_words_tfidf.md)*
