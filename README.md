# NLP → LLM Journey

A self-paced learning repo documenting my path from NLP fundamentals to building a mini GPT from scratch. Each phase combines written concept notes with hands-on Colab notebooks — building both the theory and the practical fluency needed to work seriously with modern language models.

---

## Status

🌱 **Currently in Phase 1 — NLP Fundamentals**

| Phase | Status | Topic |
|---|---|---|
| 00 | ✅ Complete | PyTorch warm-up |
| 01 | 🌱 In progress | NLP fundamentals |
| 02 | ⏳ Upcoming | Demo project — sentiment analyser |
| 03 | ⏳ Upcoming | Transformer deep dive |
| 04 | ⏳ Upcoming | LLM internals + fine-tuning |
| 05 | ⏳ Upcoming | Capstone — build a mini GPT from scratch |

---

## Structure

Every phase follows the same pattern:

```
phase_folder/
├── concepts/   ← detailed written notes per topic
└── demos/     ← hands-on Colab notebooks per topic
```

The `concepts/` files are designed to be standalone references — written so they can be revisited months later and still make sense. The `demos/` notebooks are where each concept gets practised on real code.

---

## Phase 0 — PyTorch Warm-up ✅

Building the PyTorch fundamentals that everything else depends on.

| # | Topic | Demo built |
|---|---|---|
| 01 | Tensors, slicing, broadcasting, GPU basics | — |
| 02 | Autograd | — |
| 03 | Linear regression from scratch | Manual gradient descent vs `nn.Linear` |
| 04 | nn.Module and custom layers | Two-layer MLP fitting `sin(x)` |
| 05 | DataLoader, training loop, optimisers | Binary classifier on gaussian clusters |
| 06 | MLP classifier on MNIST | ~98% test accuracy with confusion matrix analysis |

→ See [`00_pytorch_warmup/README.md`](./00_pytorch_warmup/README.md) for details.

---

## Phase 1 — NLP Fundamentals 🌱

Classical NLP — the foundation LLMs are built on.

| # | Topic | Status |
|---|---|---|
| 01 | Tokenisation — word, subword (BPE), character level | ✅ |
| 02 | Text normalisation — stemming, lemmatisation, stopwords | ⏳ |
| 03 | Bag of Words and TF-IDF representations | ⏳ |
| 04 | Word embeddings — Word2Vec, GloVe | ⏳ |
| 05 | Language modelling basics — n-grams, perplexity | ⏳ |
| 06 | Text classification pipeline | ⏳ |

---

## Stack

- **Language:** Python 3.11+
- **Framework:** PyTorch
- **NLP:** HuggingFace Transformers, tiktoken
- **Tools:** Jupyter / Google Colab, NumPy, Matplotlib, scikit-learn

---

## Why This Repo Exists

LLMs and NLP are one of the most in-demand skill sets in tech right now. There's no shortage of people who can call the OpenAI API — but there's a real shortage of engineers who understand what's happening underneath. This repo is my attempt to land firmly in the second camp: build everything from scratch, document it properly, and make sure each concept is understood deeply rather than copied.

The end goal is the **Phase 5 capstone**: a mini GPT trained from scratch on a custom dataset, with a full write-up explaining every architectural decision.
