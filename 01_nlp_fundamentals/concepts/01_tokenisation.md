# Tokenisation — Word, Subword (BPE), Character Level

> **Path:** `01_nlp_fundamentals/concepts/`  
> **Topic:** How raw text becomes the integer sequences that models actually consume.

---

## Table of Contents

1. [Why Tokenisation Exists](#1-why-tokenisation-exists)
2. [Approach 1 — Word-level Tokenisation](#2-approach-1----word-level-tokenisation)
3. [Approach 2 — Character-level Tokenisation](#3-approach-2----character-level-tokenisation)
4. [Approach 3 — Subword Tokenisation (BPE)](#4-approach-3----subword-tokenisation-bpe)
5. [How BPE Works — The Algorithm](#5-how-bpe-works----the-algorithm)
6. [BPE in Practice — `tiktoken`](#6-bpe-in-practice----tiktoken)
7. [Tokenisation Quirks](#7-tokenisation-quirks)
8. [Special Tokens](#8-special-tokens)
9. [Vocabulary Size Tradeoffs](#9-vocabulary-size-tradeoffs)
10. [From Tokens to Model Input](#10-from-tokens-to-model-input)
11. [How This Connects to LLMs](#11-how-this-connects-to-llms)

---

## 1. Why Tokenisation Exists

Computers can't read words — they need numbers. Tokenisation is the process of converting raw text into a sequence of integers that a model can process.

```
"Hello world!"  →  [9906, 1917, 0]  →  model
```

But *how* you split the text into tokens — that's where it gets interesting. There are three main approaches, and the choice has cascading effects on vocabulary size, sequence length, model behaviour, and even what tasks the model can do well.

---

## 2. Approach 1 — Word-level Tokenisation

The most intuitive idea: split on spaces and punctuation, give each unique word an ID.

```python
text  = "the cat sat on the mat"
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ids   = [vocab[w] for w in text.split()]
# → [0, 1, 2, 3, 0, 4]
```

### Problems

| Problem | Why it matters |
|---|---|
| **Vocabulary explosion** | English alone has 170,000+ words. Add names, jargon, typos, multilingual text — vocabulary becomes unmanageable |
| **Out-of-vocabulary (OOV)** | Any unseen word becomes `<UNK>`, losing all information. "ChatGPT" would be `<UNK>` to a pre-2022 model |
| **Morphology blindness** | "run", "running", "ran", "runner" become totally separate tokens with no shared meaning |

Word-level was standard before ~2018 but has been almost entirely abandoned for the reasons above.

---

## 3. Approach 2 — Character-level Tokenisation

Go the opposite direction — split into individual characters:

```python
text   = "hello"
vocab  = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
ids    = [vocab[c] for c in text]
# → [7, 4, 11, 11, 14]
```

### Pros

- **Tiny vocabulary** — just letters, digits, punctuation ≈ ~100 tokens total
- **No OOV problem** — any text in any language can be represented
- **Robust to typos** — a misspelling is still a valid sequence of characters

### Cons

- **Long sequences** — "Hello world" becomes 11 tokens instead of 2. Transformer attention scales **quadratically** with sequence length, making this expensive
- **Hard to learn meaning** — the model has to figure out that `h-e-l-l-o` represents a unit of meaning, rather than learning the word "hello" directly

Character models exist (and are interesting!) but rarely used for large-scale LLMs.

### A character tokeniser in 15 lines

```python
class CharTokeniser:
    def __init__(self, text):
        self.chars      = sorted(set(text))
        self.stoi       = {c: i for i, c in enumerate(self.chars)}
        self.itos       = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)
```

This is a complete, working tokeniser. BPE is just a smarter version of this same pattern.

---

## 4. Approach 3 — Subword Tokenisation (BPE)

The modern standard. The key insight:

> **Split rare words into pieces, keep common words whole.**

```
"the"          →  ["the"]
"transformers" →  ["transform", "ers"]
"ChatGPT"      →  ["Chat", "G", "PT"]
"unhappiness"  →  ["un", "h", "appiness"]
```

This gives you the best of both worlds — manageable vocabulary, no OOV problem, and meaningful units. The dominant algorithm is **BPE — Byte Pair Encoding**.

---

## 5. How BPE Works — The Algorithm

BPE learns a vocabulary by starting with characters and iteratively merging the most frequent adjacent pairs.

### Step-by-step example

Starting corpus: `"low low low lowest newest"`

**Step 1 — Start with characters**
```
[l,o,w] [l,o,w] [l,o,w] [l,o,w,e,s,t] [n,e,w,e,s,t]
```

**Step 2 — Count adjacent pairs**
```
(l,o): 4    (o,w): 4    (w,e): 2    (e,s): 2    (s,t): 2    ...
```

**Step 3 — Merge the most frequent pair → `lo`**
```
[lo,w] [lo,w] [lo,w] [lo,w,e,s,t] [n,e,w,e,s,t]
```

**Step 4 — Repeat. Next most frequent pair is `(lo,w)` → `low`**
```
[low] [low] [low] [low,e,s,t] [n,e,w,e,s,t]
```

**Keep merging** until you reach your target vocabulary size (e.g. 50,000 tokens).

### What you end up with

A vocabulary where common words become single tokens, rare words split into meaningful pieces, and morphologically related words often share roots — purely from statistical patterns, no linguistic rules required.

> **Important caveat:** BPE is statistical pattern matching, not linguistic analysis. It splits "unhappiness" as `["un", "h", "appiness"]` because `appiness` happened to be a frequent substring — not because it understands the prefix-stem-suffix structure.

---

## 6. BPE in Practice — `tiktoken`

OpenAI's `tiktoken` library implements the BPE tokeniser used by GPT models.

### Installation and basic use

```python
!pip install tiktoken

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4's tokeniser

text = "Hello, how are you doing today?"
ids  = enc.encode(text)
# → [9906, 11, 1268, 527, 499, 3815, 3432, 30]

# Decode back
enc.decode(ids)
# → "Hello, how are you doing today?"

# See actual tokens
[enc.decode([i]) for i in ids]
# → ['Hello', ',', ' how', ' are', ' you', ' doing', ' today', '?']
```

### Notice the leading spaces

`' how'`, `' are'`, `' you'` — the leading space is **part of the token**. This is intentional and encodes word boundary information directly inside each token.

This means `Hello` and ` Hello` (with space) are **different token IDs**. They look identical to humans but the model sees them as separate symbols.

### Common encoders

| Encoding | Used by | Vocab size |
|---|---|---|
| `cl100k_base` | GPT-4, GPT-3.5-turbo | 100,277 |
| `gpt2` | GPT-2, GPT-3 | 50,257 |
| `o200k_base` | GPT-4o | 200,000 |

Different models use different tokenisers and they are **not interchangeable** — always use the tokeniser that matches your model.

---

## 7. Tokenisation Quirks

These are real, surprising, and they matter for prompt engineering.

### Spaces matter

```python
enc.encode("Token")    # → [3404]      one token
enc.encode(" Token")   # → [9857]      different token!
enc.encode("  Token")  # → [220, 9857] separator + word
```

Trailing or extra whitespace in prompts can quietly hurt LLM performance because you're feeding the model less common token sequences.

### Case matters

```python
enc.encode("hello")    # → [15339]
enc.encode("Hello")    # → [9906]         entirely different ID
enc.encode("HELLO")    # → [51812, 1623]  splits into 2 tokens
```

ALL CAPS isn't a "loud version" to the model — it's an unusual sequence the model saw less often during training. One reason caps-heavy prompts can produce worse responses.

### Numbers are unpredictable

```python
enc.encode("1")           # → 1 token
enc.encode("100")         # → 1 token
enc.encode("1000")        # → 2 tokens   ← splits!
enc.encode("1234567890")  # → 4 tokens: ['123', '4', '5678', '90']
```

Numbers split into seemingly arbitrary chunks. The model doesn't see the digits — it sees opaque token IDs.

### Why LLMs struggle with maths

```python
enc.encode("1234 + 5678 = 6912")
# → ['123', '4', ' +', ' ', '567', '8', ' =', ' ', '691', '2']
```

The model has to recognise that `123` + `4` = the number 1234, perform addition on it, then output `691` + `2` as a coherent answer. **This is brutal at the token level.** Most LLM arithmetic weaknesses trace back to this.

This is also why LLMs struggle with:
- **Spelling** — they don't see individual letters
- **Reversing strings** — same reason
- **Counting characters** — "How many r's in strawberry?" requires character-level vision they don't have

---

## 8. Special Tokens

Every tokeniser includes special tokens for structural purposes:

| Token | Used by | Purpose |
|---|---|---|
| `<\|endoftext\|>` | GPT | Marks end of a document; signals model to stop generating |
| `[CLS]` | BERT | Classification token, prepended to sequences |
| `[SEP]` | BERT | Separates two sentences |
| `[PAD]` | All | Padding to make batched sequences same length |
| `[UNK]` | Older models | Fallback for OOV (rare in BPE) |
| `[MASK]` | BERT | Token replaced during masked language modelling |

These act as structural anchors — the model learns to use them as stop signals, separators, and summary representations.

---

## 9. Vocabulary Size Tradeoffs

| Vocab size | Tokens per word (avg) | Pros | Cons |
|---|---|---|---|
| Small (~1k) | ~3-5 | Tiny embedding table | Long sequences, hard to learn |
| Medium (~32k) | ~1.3 | Balanced | — |
| Large (~100k) | ~1.1 | Short sequences, multilingual efficiency | Larger embedding table (millions of params) |

| Model | Vocab size |
|---|---|
| GPT-2 | 50,257 |
| LLaMA 2 | 32,000 |
| GPT-4 | 100,277 |
| GPT-4o | ~200,000 |

The trend is toward larger vocabularies — they reduce sequence length (and therefore compute cost) and handle non-English text more efficiently.

### Cross-language efficiency

```python
"Hello world"        →   2 tokens (English)
"불고기 먹고 싶다"     →  ~7 tokens (Korean, GPT-4)
"불고기 먹고 싶다"     →  ~12 tokens (GPT-2)
```

GPT-2's tokeniser was English-heavy. GPT-4's covers more languages efficiently. This directly affects API cost and inference speed for non-English workloads.

---

## 10. From Tokens to Model Input

Tokenisation produces a list of IDs. Models take **batched, padded tensors**.

```python
import torch

sentences = [
    "The cat sat on the mat",
    "Transformers changed NLP forever",
    "Tokenisation is step zero",
]

# Encode each sentence
encoded = [enc.encode(s) for s in sentences]
# [[791, 8415, ...], [9140, ...], [3404, 2065, ...]]

# Pad all to same length so they fit in a tensor
max_len   = max(len(e) for e in encoded)
pad_id    = 0   # or a dedicated pad token
padded    = [e + [pad_id] * (max_len - len(e)) for e in encoded]

# Convert to tensor
batch = torch.tensor(padded)
print(batch.shape)
# → torch.Size([3, 7])  (batch_size=3, seq_len=7)
```

This `(batch_size, seq_len)` tensor is the **direct input to `nn.Embedding` in a Transformer.** Every LLM forward pass starts here.

---

## 11. How This Connects to LLMs

| Concept | Where it appears in LLMs |
|---|---|
| BPE vocabulary | The embedding table — one row per token, ~50k–100k rows |
| Token IDs | Integers fed into `nn.Embedding` at the start of every forward pass |
| Sequence length | Directly determines memory and compute (quadratic in attention) |
| Special tokens | `<\|endoftext\|>` triggers stop; `[PAD]` enables batching |
| Tokeniser choice | GPT-4 ≠ LLaMA — vocabularies are not interchangeable |
| Subword splits | Why LLMs struggle with spelling, character counting, and arithmetic |
| Leading spaces in tokens | Why prompt whitespace formatting affects output quality |

### The full pipeline

```
Raw text
    ↓  tokenisation (BPE)
Token IDs  [9906, 1917, 0, ...]
    ↓  nn.Embedding lookup
Token vectors  shape (seq_len, embed_dim)
    ↓  + positional encoding
    ↓  Transformer blocks × N
    ↓  linear projection
Logits over vocabulary  shape (seq_len, vocab_size)
    ↓  CrossEntropyLoss vs next token
Loss
```

Tokenisation is **step zero** — everything else depends on getting it right.

### Three things to remember

1. **BPE is statistical pattern matching, not linguistic analysis.** It can split "unhappiness" awkwardly because it doesn't understand morphology.

2. **Tokenisation is the source of many LLM weaknesses** — spelling, maths, character counting, reverse-string puzzles. Models never see clean characters or digits.

3. **Prompt formatting matters at the token level.** Trailing spaces, unusual capitalisation, and unconventional punctuation can quietly degrade output quality — even though the prompt looks normal to humans.

---

*Phase 0 → Complete*  
*Phase 1 — concept 01 → you are here*  
*Next concept → [02 — Text Normalisation: Stemming, Lemmatisation, Stopwords](./02_text_normalisation.md)*
