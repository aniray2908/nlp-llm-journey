# Text Classification Pipeline — Features → Model → Evaluate

> **Path:** `01_nlp_fundamentals/concepts/`  
> **Topic:** The complete workflow for building a text classifier. From raw text to evaluation metrics, understanding every step and where things can go wrong.

---

## Table of Contents

1. [What is Text Classification?](#1-what-is-text-classification)
2. [The Pipeline Overview](#2-the-pipeline-overview)
3. [Step 1 — Data Collection & Splitting](#3-step-1----data-collection--splitting)
4. [Step 2 — Preprocessing](#4-step-2----preprocessing)
5. [Step 3 — Feature Extraction](#5-step-3----feature-extraction)
6. [Step 4 — Model Training](#6-step-4----model-training)
7. [Step 5 — Prediction](#7-step-5----prediction)
8. [Step 6 — Evaluation](#8-step-6----evaluation)
9. [Common Mistakes & Data Leakage](#9-common-mistakes--data-leakage)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. What is Text Classification?

**Text classification** is assigning text to one of several predefined categories.

```
Input:  "This movie was absolutely amazing!"
Output: Positive

Input:  "Terrible waste of time"
Output: Negative

Input:  "The cat slept peacefully"
Output: Neutral
```

### Common applications

| Task | Input | Output |
|---|---|---|
| **Sentiment analysis** | Review text | Positive, Negative, Neutral |
| **Spam detection** | Email | Spam, Not spam |
| **Topic classification** | Article | Politics, Sports, Tech, ... |
| **Intent detection** | User query | Book flight, Check weather, ... |
| **Toxicity detection** | Comment | Toxic, Safe |

All follow the same pipeline — only the data and classes differ.

---

## 2. The Pipeline Overview

Every text classification project follows this structure:

```
1. RAW TEXT
   ↓
2. DATA SPLITTING
   Train/val/test split (always before any processing!)
   ↓
3. PREPROCESSING
   Tokenise, normalise, clean
   ↓
4. FEATURE EXTRACTION
   BoW, TF-IDF, embeddings, or learned representations
   ↓
5. MODEL TRAINING
   Train classifier on training features and labels
   ↓
6. PREDICTION
   Apply model to test features
   ↓
7. EVALUATION
   Measure quality on test set
   ↓
8. ANALYSIS
   Understand errors, improve, iterate
```

Each step has choices and tradeoffs. The quality of the final model depends on decisions at every stage.

---

## 3. Step 1 — Data Collection & Splitting

### The cardinal rule

**Split your data BEFORE any preprocessing or feature extraction.** This prevents data leakage.

### Train/Val/Test split

Standard practice:

```
Total dataset: 1000 samples
  ↓
Train (80%): 800 samples  → training the model
Val (10%):   100 samples  → tuning hyperparameters (optional)
Test (10%):  100 samples  → final evaluation (never touch!)
```

### Why this matters

- **Training set** — model learns from this
- **Validation set** — you tune hyperparameters, pick models, check early stopping
- **Test set** — final evaluation; never train on it or make decisions based on intermediate results

### Stratified splitting

When classes are imbalanced, use stratification:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # maintain class balance
)
```

Stratification ensures each split has roughly the same class distribution as the original dataset.

---

## 4. Step 2 — Preprocessing

Clean and normalise the text before feature extraction.

### Tokenisation

```python
text = "The cat sat on the mat."
tokens = text.lower().split()
# → ['the', 'cat', 'sat', 'on', 'the', 'mat.']
```

### Lowercasing

```python
text = "ChatGPT is AMAZING"
text = text.lower()
# → "chatgpt is amazing"
```

Reduces vocabulary. Trade-off: loses information (proper nouns, emphasis).

### Removing punctuation

```python
import re
text = "Hello, world! How are you?"
text = re.sub(r'[^\w\s]', '', text)
# → "Hello world How are you"
```

Optional but often helpful. Trade-off: "don't" becomes "dont" (loses structure).

### Optional: Remove stopwords

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop_words]
```

For classical ML: often helps by reducing noise.  
For deep learning: usually skip it — let the model learn.

### Optional: Stem/Lemmatise

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
tokens = [stemmer.stem(t) for t in tokens]
```

Reduces "running", "runs", "ran" to "run".  
For classical ML: sometimes helps.  
For deep learning: BPE and embeddings handle this.

### Critical: Fit preprocessing on train only

```python
# ❌ WRONG: Normaliser sees both train and test
normaliser.fit(X_train + X_test)

# ✅ RIGHT: Normaliser sees only training data
normaliser.fit(X_train)
train_processed = normaliser.transform(X_train)
test_processed = normaliser.transform(X_test)
```

This applies to stopword lists, stemming rules, anything learned from data.

---

## 5. Step 3 — Feature Extraction

Convert preprocessed text into numerical features.

### Option A: Bag of Words (BoW)

Count how many times each word appears:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)  # fit on train only!
X_test = vectorizer.transform(X_test_text)       # apply to test
```

```
Document: "the cat the dog"
Vocabulary: {cat: 0, dog: 1, the: 2}
BoW vector: [1, 1, 2]
```

**Pros:** Fast, interpretable, works with small data  
**Cons:** Sparse, high-dimensional, no semantics

### Option B: TF-IDF

Weight words by how rare they are across the corpus:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
```

TF-IDF often outperforms BoW on classical ML tasks.

**Pros:** Weights informative words higher, still interpretable  
**Cons:** Still sparse, high-dimensional

### Option C: Word Embeddings

Use pre-trained embeddings (Word2Vec, GloVe):

```python
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(...)

def doc_embedding(text):
    words = text.split()
    vectors = [word_vectors[w] for w in words if w in word_vectors]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

X_train = np.array([doc_embedding(t) for t in X_train_text])
X_test = np.array([doc_embedding(t) for t in X_test_text])
```

**Pros:** Dense, low-dimensional, captures semantics  
**Cons:** Need pre-trained model, less interpretable

### Option D: Learned embeddings

Let a neural network learn embeddings end-to-end:

```python
# PyTorch model with embedding layer
model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.LSTM(...),
    nn.Linear(..., num_classes)
)
```

**Pros:** Optimised for your specific task  
**Cons:** Needs more data, less interpretable

### Critical: Fit vectoriser on train only

```python
# ❌ WRONG
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train + X_test)  # vocabulary includes test data!

# ✅ RIGHT
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
```

If you fit the vectoriser on both train and test, **you've leaked information from test into training**. The model will perform worse in production.

---

## 6. Step 4 — Model Training

Pick a model and train it on features and labels.

### Simple: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
```

Fast, interpretable, works with high-dimensional sparse data (TF-IDF).

### Robust: SVM

```python
from sklearn.svm import SVC

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
```

Often outperforms logistic regression, but slower to train.

### Fast: Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)
```

Very fast, works well with BoW/TF-IDF, makes simplifying assumptions.

### Powerful: Neural Network

```python
# LSTM in PyTorch
model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.LSTM(hidden_size, batch_first=True),
    nn.Linear(hidden_size, num_classes)
)

# Training loop...
```

Most powerful, needs more data, slower to train.

### When to choose which

| Scenario | Best choice |
|---|---|
| **Little data (<1000 samples)** | Logistic regression, Naive Bayes |
| **Medium data (1k-100k)** | SVM, Logistic regression |
| **Lots of data (>100k)** | Neural networks, Transformers |
| **Interpretability required** | Logistic regression, Naive Bayes |
| **Maximum performance** | Ensemble, Transformers |
| **Speed matters** | Naive Bayes, Logistic regression |

**Start simple.** Complex models only if simple ones don't work.

---

## 7. Step 5 — Prediction

Apply the trained model to new data.

### Predict class

```python
y_pred = clf.predict(X_test)
# → [0, 1, 1, 0, 1, ...]
```

Hard prediction — the class the model thinks is most likely.

### Predict probability

```python
y_pred_proba = clf.predict_proba(X_test)
# → [[0.8, 0.2], [0.3, 0.7], ...]
#    (probability of class 0, probability of class 1)
```

Soft prediction — confidence scores for each class.

### Adjust threshold

By default, threshold is 0.5:

```
If P(positive) > 0.5 → predict positive
Otherwise → predict negative
```

For imbalanced data or different error costs, adjust:

```python
y_pred = (y_pred_proba[:, 1] > 0.3).astype(int)  # lower threshold
# More predictions as positive, higher recall, lower precision
```

Lower threshold → more false positives, higher recall  
Higher threshold → fewer false positives, lower recall

---

## 8. Step 6 — Evaluation

Measure model quality on held-out test set.

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Percentage of correct predictions. **Only use when classes are balanced.**

### Precision

```
Precision = TP / (TP + FP)

"Of the predictions I made as positive, how many were correct?"
```

High precision = few false alarms.  
Use when **false positives are expensive** (e.g., spam filter — don't block real emails).

### Recall

```
Recall = TP / (TP + FN)

"Of the actual positives, how many did I find?"
```

High recall = catch most positives.  
Use when **false negatives are expensive** (e.g., disease detection — don't miss cases).

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Harmonic mean of precision and recall.
```

Use when you want **balanced precision and recall**.

### Confusion Matrix

```
                Predicted
              Negative  Positive
Actual Neg      TN        FP
       Pos      FN        TP
```

Shows exactly where the model makes mistakes. More informative than accuracy alone.

### ROC-AUC

```
AUC = Area Under the ROC Curve
```

Measures model's ability to distinguish between classes across all thresholds.

**AUC = 0.5:** Random guessing  
**AUC = 1.0:** Perfect classifier  
**AUC = 0.7:** Reasonable  
**AUC = 0.9:** Excellent

### Which metrics to use?

| Scenario | Metric |
|---|---|
| **Balanced classes** | Accuracy, F1, AUC |
| **Imbalanced classes** | Precision, Recall, F1, AUC (not accuracy!) |
| **Cost asymmetry** | Precision if FP expensive; Recall if FN expensive |
| **General health check** | Confusion matrix + F1 |

---

## 9. Common Mistakes & Data Leakage

### Mistake 1: Evaluating on training data

```python
# ❌ WRONG
clf.fit(X, y)
y_pred = clf.predict(X)  # predicting on training data!
accuracy = accuracy_score(y, y_pred)  # inflated!

# ✅ RIGHT
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  # unseen data
accuracy = accuracy_score(y_test, y_pred)
```

Training accuracy is always higher — model memorised the training data.

### Mistake 2: Fitting vectoriser on test data

```python
# ❌ WRONG: vocabulary learned from test data
vectorizer = TfidfVectorizer()
X_test = vectorizer.fit_transform(X_test_text)

# ✅ RIGHT: vocabulary learned only from training data
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train_text)
X_train = vectorizer.transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
```

**Data leakage:** Information from test set influenced the model (indirectly).

### Mistake 3: Using accuracy on imbalanced data

```python
# Dataset: 950 negative, 50 positive samples

# ❌ WRONG
clf = DummyClassifier(strategy='constant', constant=0)  # always predict negative
accuracy = clf.score(X_test, y_test)  # → 95% (!!)

# ✅ RIGHT
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

On imbalanced data, a naive classifier can achieve high accuracy while predicting nothing.

### Mistake 4: Hyperparameter tuning on test set

```python
# ❌ WRONG
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    score = accuracy_score(y_test, (y_pred_proba > threshold))
    if score > best_score:
        best_threshold = threshold  # overfitting to test set!

# ✅ RIGHT
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    score = accuracy_score(y_val, (y_val_proba > threshold))
    if score > best_score:
        best_threshold = threshold  # tune on validation set
# Then evaluate on test set with fixed threshold
```

Tune on validation set, evaluate on test set.

### Mistake 5: Not checking for data leakage

```
Data leakage = information from outside the training set
somehow influences the model

Examples:
  - Fitting vectoriser on train + test (mistake #2)
  - Using features that include the label (e.g., timestamp from when label was assigned)
  - Preprocessing on full data before splitting
  - Using future information to predict the past
```

Always ask: "Could the test set have influenced training?"

---

## 10. How This Connects to LLMs

| Stage | Task | Method |
|---|---|---|
| **Classical ML** | Text classification | BoW/TF-IDF + Logistic Regression |
| **Modern ML** | Text classification | Word embeddings + LSTM/CNN |
| **Transformers** | Text classification | BERT fine-tuning |
| **LLMs** | Everything | In-context learning, few-shot, zero-shot |

### The evolution

```
Early NLP: Hard features (BoW) → simple models
Modern NLP: Learned features (embeddings) → neural models
Transformers: Contextual features learned end-to-end
LLMs: Pre-trained on trillions of tokens, fine-tune for any task
```

### Text classification with LLMs

Modern approach:

```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased')
result = classifier("This movie was amazing!")
# → [{'label': 'POSITIVE', 'score': 0.9998}]
```

No manual feature extraction, no vectoriser — the model handles everything.

### But understanding classical ML matters

Even though you'd use transformers in production, understanding the full pipeline teaches you:

- Why data splitting matters
- What data leakage is and why it's catastrophic
- How features influence model quality
- When simple models are better than complex ones
- How to evaluate properly

These principles apply whether you're using TF-IDF or a billion-parameter LLM.

---

## Summary

**The text classification pipeline is the bridge between raw text and predictions:**

```
Data → Split → Preprocess → Vectorise → Train → Predict → Evaluate
```

Each step matters. Most errors come from **not splitting properly** or **data leakage**. Always:

1. Split before any processing
2. Fit on train only
3. Evaluate on held-out test set
4. Use appropriate metrics for your problem

Master this pipeline, and you can build text classifiers with any feature representation or model architecture.

---

*Phase 1 — concept 06 → Complete!*  
*Previous concept → [05 — Language Modelling Basics](./05_language_modelling.md)*  
*Next → [Phase 2 — Demo Project: Sentiment Analyser](../02_demo_sentiment_analyser/README.md)*
