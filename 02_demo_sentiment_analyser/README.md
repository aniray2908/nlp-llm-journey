# Phase 2 — Sentiment Analyser

> **First portfolio piece:** An end-to-end sentiment classifier. Take a real dataset, build two models, compare them, and deploy. This project demonstrates the complete NLP pipeline from data to inference.

---

## What This Project Does

**Sentiment Analysis on Movie Reviews**

Given a movie review as input, the model predicts whether it's positive or negative.

```
Input:  "This movie was absolutely amazing! Best film I've seen in years."
Output: Positive (82% confidence)

Input:  "Terrible waste of time. Couldn't finish it."
Output: Negative (91% confidence)
```

---

## Dataset

**IMDB Movie Reviews**
- Source: Hugging Face `datasets` library
- Size: 2,000 reviews (1,000 positive, 1,000 negative)
- Test set: 500 reviews (250 positive, 250 negative)
- Balanced: 50/50 positive/negative split

Each review is 1-5 sentences describing a movie, labeled as positive (1) or negative (0).

---

## Models & Results

### Model 1: Baseline — TF-IDF + Logistic Regression

**Approach:**
- Feature extraction: TF-IDF vectorisation (max 1000 features)
- Model: Logistic Regression
- Training time: <1 second
- Inference time: <1ms per review

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.820 |
| Precision | 0.801 |
| Recall | 0.851 |
| F1 | 0.825 |

### Model 2: Upgrade — GloVe Embeddings + LSTM

**Approach:**
- Feature extraction: Pre-trained GloVe embeddings (100-dim)
- Model: LSTM (1 layer, 64 hidden units, dropout=0.3)
- Training time: ~30 seconds (15 epochs with early stopping)
- Inference time: ~5ms per review

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.656 |
| Precision | 0.665 |
| Recall | 0.732 |
| F1 | 0.696 |

---

## Model Comparison

### Which Model Wins?

**Baseline (TF-IDF + LogReg) significantly outperforms the LSTM upgrade.**

```
                    TF-IDF + LogReg    GloVe + LSTM
Accuracy:           0.820              0.656 (↓ 20%)
Precision:          0.801              0.665 (↓ 17%)
Recall:             0.851              0.732 (↓ 14%)
F1:                 0.825              0.696 (↓ 16%)
```

### Why Does the Baseline Win?

1. **GloVe averaging loses information** — Averaging all word embeddings collapses sequence information. The LSTM loses the structure it needs to shine.

2. **Small dataset favours simple models** — With only 2,000 training samples, logistic regression has less capacity to overfit. LSTMs need more data (ideally 10k+) to justify their complexity.

3. **TF-IDF captures sentiment signals** — Word frequency is highly predictive for sentiment analysis. Words like "amazing", "terrible", "excellent", "awful" appear frequently in positive/negative reviews. TF-IDF captures this directly.

4. **LSTM is overfitting** — Validation loss increases after epoch ~6, indicating the model memorises training data without generalising.

### Key Learning

**This demonstrates a crucial ML principle: complex models aren't always better.** Start simple, measure, then upgrade only if needed. A baseline should always be your first step.

**For this dataset, the best approach would be:**
- Keep TF-IDF + LogReg as production baseline (accurate, fast, interpretable)
- Or upgrade to BERT fine-tuning (Phase 3) which would likely beat both

---

## Training Process

### Baseline Training

```python
# 1. Vectorise training reviews with TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(reviews_train)

# 2. Train logistic regression
clf = LogisticRegression()
clf.fit(X_train, labels_train)

# 3. Evaluate on test set
predictions = clf.predict(X_test)
accuracy = accuracy_score(labels_test, predictions)
```

### LSTM Training

```python
# 1. Load pre-trained GloVe embeddings
word_vectors = api.load("glove-wiki-gigaword-100")

# 2. Convert reviews to embedding sequences
# Each word → 100-dim vector
# Each review → (seq_len, 100)

# 3. Train LSTM with early stopping
# Stop when validation loss stops improving

# 4. Evaluate on test set
predictions = model.predict(X_test)
accuracy = accuracy_score(labels_test, predictions)
```

---

## How to Use

### Option 1: Jupyter Notebook

```bash
jupyter notebook demos/01_sentiment_analyser.ipynb
```

Run through all cells to reproduce the entire pipeline:
- Load and explore IMDB data
- Train baseline model
- Train LSTM model
- Compare results

### Option 2: Python Inference Script

```bash
python inference.py
```

Interactive command-line tool:
```
Enter a movie review: "This movie was fantastic!"
Prediction: Positive (confidence: 0.85)
```

### Option 3: Gradio Web Demo (Interactive)

```bash
python gradio_demo.py
```

Opens a web interface at `http://localhost:7860`:
- Paste a review
- Click "Submit"
- See prediction and confidence

---

## Project Structure

```
02_demo_sentiment_analyser/
├── README.md                          (this file)
├── demos/
│   └── 01_sentiment_analyser.ipynb   (full training notebook)
├── inference.py                       (standalone prediction script)
├── gradio_demo.py                     (web UI)
└── requirements.txt                   (dependencies)
```

---

## Dependencies

```
numpy
pandas
scikit-learn
torch
torchvision
gensim
matplotlib
seaborn
datasets
gradio
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Files in Detail

### `demos/01_sentiment_analyser.ipynb`

The complete project notebook with 12 cells:

1. **Imports & Setup** — Load libraries, check GPU
2. **Load IMDB Data** — Download dataset, balance classes
3. **Split Data** — Train/test split (80/20)
4. **Baseline Model** — TF-IDF + LogReg training and evaluation
5. **Load Embeddings** — Pre-trained GloVe vectors
6. **PyTorch Dataset** — Custom dataset class for LSTM
7. **LSTM Model** — Define architecture
8. **Training Loop** — Train with early stopping
9. **Evaluate LSTM** — Test set evaluation
10. **Compare Models** — Side-by-side comparison and plots
11. **Inference Function** — Predict on new reviews
12. **Summary** — Recap of findings

**Runtime:** ~5 minutes on GPU, ~20 minutes on CPU

### `inference.py`

Minimal standalone script to load the trained baseline model and predict on new reviews. No dependencies on the notebook.

```python
# Load TF-IDF vectoriser and logistic regression model
# Accept user input or file input
# Return prediction and confidence
```

### `gradio_demo.py`

Interactive web interface for sentiment analysis. Zero code — just paste a review and get a prediction. Great for demos and sharing.

---

## Results & Insights

### Confusion Matrix (Baseline Model)

On the 500-sample test set:
- True Negatives: 212 ✓
- False Positives: 53 (predicted positive, actually negative)
- False Negatives: 41 (predicted negative, actually positive)
- True Positives: 194 ✓

The model is slightly better at catching negatives (recall=0.85) than positives (recall=0.79).

### What the Model Gets Right

✅ Strong positive reviews ("amazing", "excellent", "loved")  
✅ Strong negative reviews ("terrible", "awful", "horrible")  
✅ Neutral sentiments with emotional words

### What the Model Struggles With

❌ Sarcasm ("Yeah, great movie" = negative but looks positive)  
❌ Mixed sentiments ("Good acting but bad plot")  
❌ Double negatives ("Not bad" = positive but contains "bad")  
❌ Subtle opinions ("It was okay")

---

## What I Learned

**1. Start with a baseline**

Always build the simplest model first. It's fast, interpretable, and often competitive. TF-IDF + LogReg took 10 seconds, LSTM took 2 minutes — and LogReg won.

**2. More complex ≠ better**

Neural networks are powerful but need large datasets and careful tuning. On 2,000 reviews, logistic regression is optimal.

**3. Feature engineering matters**

TF-IDF automatically weights informative words. Averaging embeddings threw away the benefits. With better pooling (attention) or different embeddings (BERT), the LSTM would likely win.

**4. Data leakage is real**

The IMDB dataset initially loaded only negative reviews. Careful exploration saved the project.

**5. Early stopping prevents overfitting**

Without early stopping, the LSTM trained for 15 epochs and overfit terribly. With early stopping, it stopped at epoch 6 when validation loss improved.

---

## Improvements & Next Steps

**To improve this project:**

1. **Better embedding pooling** — Use max pooling or attention instead of averaging
2. **Bidirectional LSTM** — See context from both directions
3. **More data** — LSTM needs 10k+ samples to shine
4. **Pre-trained models** — BERT fine-tuning (Phase 3) would likely win
5. **Ensemble** — Combine baseline + LSTM predictions
6. **Error analysis** — Debug misclassified reviews to understand failure modes

**Phase 3 will explore BERT and Transformers**, which are strictly better than both baseline and LSTM for this task.

---

## How This Fits Into the Journey

| Phase | Topic | This project |
|---|---|---|
| Phase 0 | PyTorch fundamentals | Built the training loops and models |
| Phase 1 | NLP fundamentals | Applied tokenisation, embeddings, classification pipeline |
| **Phase 2** | **Demo project** | **Brings it all together into a real product** |
| Phase 3 | Transformers | Will implement BERT for even better results |
| Phase 4 | LLM internals | Will fine-tune LLMs on custom data |
| Phase 5 | Capstone | Build mini-GPT from scratch |

---

## Credits & Resources

- **IMDB Dataset:** Hugging Face Datasets library
- **GloVe Embeddings:** Pre-trained vectors by Stanford NLP Group
- **Training methodology:** "Speech and Language Processing" by Jurafsky & Martin

---

## Author

Built as part of the NLP → LLM learning journey.

**Repository:** [nlp-llm-journey](https://github.com/yourusername/nlp-llm-journey)

---

*Phase 2 Complete. Next: Phase 3 — Transformer Deep Dive* 🚀
