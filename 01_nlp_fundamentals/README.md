# Phase 1 — NLP Fundamentals

> Classical NLP is the foundation that everything modern — Transformers, LLMs, embeddings — is built on. Understanding tokenisation, embeddings, and language modelling deeply is what separates engineers who use LLMs from those who truly understand them.

---

## What's in This Phase

This phase covers the classical NLP pipeline and the core concepts that modern language models still rely on. Nothing here is outdated — it's foundational.

| # | Topic | Type | What you'll learn |
|---|---|---|---|
| 01 | Tokenisation | Learn + Build | Word, subword (BPE), character-level; how text becomes numbers |
| 02 | Text normalisation | Learn + Build | Stemming, lemmatisation, stopwords; when to clean and when to skip |
| 03 | Bag of Words & TF-IDF | Learn + Build | Sparse vector representations; why they work and their limits |
| 04 | Word embeddings | Learn + Build | Word2Vec, GloVe, semantic similarity; geometry of meaning |
| 05 | Language modelling | Learn + Build | n-grams, perplexity; the training signal behind all LLMs |
| 06 | Text classification | Learn + Build | Complete pipeline: data → features → model → evaluate |

---

## Folder Structure

```
01_nlp_fundamentals/
├── concepts/
│   ├── 01_tokenisation.md
│   ├── 02_text_normalisation.md
│   ├── 03_bag_of_words_tfidf.md
│   ├── 04_word_embeddings.md
│   ├── 05_language_modelling.md
│   └── 06_text_classification_pipeline.md
└── demos/
    ├── 01_tokenisation.ipynb
    ├── 02_text_normalisation.ipynb
    ├── 03_bag_of_words_tfidf.ipynb
    ├── 04_word_embeddings.ipynb
    ├── 05_language_modelling.ipynb
    └── 06_text_classification_pipeline.ipynb
```

Each concept has:
- **Detailed markdown notes** in `concepts/` — reference material you can revisit anytime
- **Hands-on Colab notebook** in `demos/` — where you implement and experiment

---

## Key Concepts Covered

### Tokenisation
How text becomes a sequence of token IDs. Three approaches:
- **Word-level:** Simple but vocabulary explosion
- **Subword (BPE):** The modern standard; balances vocabulary size and sequence length
- **Character-level:** Always works but sequences get very long

Every LLM uses some form of tokenisation. Understanding BPE is critical — it's what GPT uses.

### Text Normalisation
Cleaning text for downstream processing. When to apply what:
- **Lowercasing & whitespace:** Always
- **Punctuation removal:** Depends on task
- **Stemming/lemmatisation:** Classical ML only; skip for deep learning
- **Stopword removal:** Usually skip for modern models

The key insight: **modern deep learning models learn better when you feed them raw text.** Classical preprocessing is a sign you're using old techniques.

### Bag of Words & TF-IDF
The simplest vector representations:
- **BoW:** Just word counts; sparse, high-dimensional
- **TF-IDF:** Weight words by rarity; usually better than BoW

Neither is used in modern NLP, but understanding them teaches you what "representing text as numbers" means. They're the ancestors of embeddings.

### Word Embeddings
**The revolution that proved text representation matters.** Two key techniques:
- **Word2Vec:** Predict context words; learns from co-occurrence
- **GloVe:** Combines local context and global statistics

Embeddings are dense (300-dim vs 50k-dim for BoW), low-dimensional, and **semantically meaningful** — similar words cluster together.

Limitations: context-free (same word, one vector), polysemy (can't distinguish "bank" meanings).

Modern Transformers solve this with **contextual embeddings.**

### Language Modelling
**The training objective behind all LLMs:** Predict the next word.

Three approaches:
- **n-grams:** Count word sequences; simple, sparse, doesn't scale
- **Neural models (RNNs):** Learn distributed representations; suffers from long-term dependencies
- **Transformers:** Attend to all previous tokens; no bottleneck, scales to billions of parameters

Perplexity measures how well a model predicts. It's interpretable: "how many equally plausible words could come next?"

This is how LLMs are trained at scale.

### Text Classification
**The complete pipeline:** Raw text → features → model → predictions → evaluation.

Five common mistakes:
1. Evaluating on training data
2. Fitting vectoriser on test data (data leakage)
3. Using accuracy on imbalanced data
4. Hyperparameter tuning on test set
5. Not checking for data leakage

Understanding this pipeline is non-negotiable. Most ML failures trace back to one of these mistakes.

---

## Projects Built

**Tokenisation:** Explored BPE, compared token counts across languages, saw how rare words split into subwords.

**Text normalisation:** Compared stemming vs lemmatisation, saw how stopword removal breaks sentiment analysis.

**BoW & TF-IDF:** Built complete feature extraction pipeline, compared accuracy across vectorisers.

**Word embeddings:** Loaded pre-trained GloVe, visualised word clusters with t-SNE, solved analogies (king - man + woman = queen).

**Language modelling:** Built unigram, bigram, trigram models, computed perplexity, generated text, saw sparsity problem.

**Text classification:** Built three classifiers (BoW+LogReg, TF-IDF+LogReg, TF-IDF+NaiveBayes), compared metrics, plotted ROC curves.

---

## Reading Material

**Required:** "Speech and Language Processing" (3rd edition) by Jurafsky & Martin, Chapters 1–6

Available free online at: https://web.stanford.edu/~jurafsky/slp3/

This is the gold standard textbook for NLP. Reading the first 6 chapters covers:
- Ch 1: Introduction and Tokenisation
- Ch 2: Regular Expressions, Text Normalisation, Edit Distance
- Ch 3: N-gram Language Models
- Ch 4: Naive Bayes and Sentiment Classification
- Ch 5: Logistic Regression
- Ch 6: Vector Semantics and Embeddings

These chapters directly parallel what you've learned in Phase 1. Reading them cements your understanding and introduces formal notation and additional algorithms.

---

## What You Know Now

By completing Phase 1, you understand:

✓ How text becomes numbers (tokenisation)  
✓ Why and when to clean text (normalisation)  
✓ How classical ML represents text (BoW, TF-IDF)  
✓ Why embeddings are better (semantics, density)  
✓ How LLMs are trained (language modelling)  
✓ How to build classifiers that actually work (pipeline, evaluation)  
✓ Where data leakage hides and how to prevent it  

You also know the limitations of each approach and when to upgrade:
- BoW → embeddings (semantics)
- Embeddings → contextual embeddings (context)
- RNNs → Transformers (scalability)
- Task-specific models → pre-trained LLMs (generality)

---

## What's Next

**Phase 2 — Sentiment Analyser Capstone**

Take everything from Phase 1 and build an **end-to-end sentiment classifier:**
- Load real review data (IMDB or Amazon)
- Build a baseline with TF-IDF + logistic regression
- Upgrade to GloVe embeddings + PyTorch LSTM
- Deploy with a simple inference script
- Document everything in a polished README

This is your first real portfolio piece. It shows you can take a real dataset, build a full pipeline, and ship it.

After that:

**Phase 3 — Transformer Deep Dive**  
Implement attention, multi-head attention, positional encoding, full encoder blocks. Fine-tune BERT.

**Phase 4 — LLM Internals**  
Decoder-only models, causal masking, LoRA fine-tuning, RAG.

**Phase 5 — Capstone**  
Build a mini GPT from scratch. The project that sets you apart.

---

## Tips for Success in This Phase

1. **Run every notebook yourself.** Don't just read; code along. That's where learning happens.

2. **Experiment in the notebooks.** Change hyperparameters, try different settings, see what breaks and why.

3. **Read the concept notes carefully.** They're written to be references you return to. Bookmark the sections you find confusing.

4. **Do the reading.** "Speech and Language Processing" is the best NLP textbook. The first 6 chapters are foundational.

5. **Notice the evolution:** BoW → embeddings → contextual embeddings → LLMs. Each step solves problems of the previous. This arc appears throughout NLP.

6. **Internalise the pipeline:** Data → Split → Preprocess → Vectorise → Train → Predict → Evaluate. You'll use this for every classification task, forever.

7. **Learn to spot data leakage.** It's the #1 mistake. Always ask: "Could the test set have influenced this?"

---

## Philosophy

Classical NLP (pre-2015) is not outdated — it's foundational. Modern deep learning didn't replace it; it **improved** on it while keeping the same principles:

- Tokenisation is still essential (just more sophisticated with BPE)
- Feature engineering is still critical (embeddings are learned features)
- Language modelling is still the training signal (just with Transformers instead of n-grams)
- The pipeline is still the same (data → features → model → evaluate)

Engineers who understand this history can navigate the modern ML landscape. Those who skip it struggle when something doesn't work.

---

*Phase 0 → PyTorch warm-up (complete)*  
*Phase 1 → NLP fundamentals (you are here)*  
*Phase 2 → Demo project: sentiment analyser (next)*
