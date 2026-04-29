"""
inference.py — Sentiment Analysis Inference Script

Loads the trained TF-IDF + Logistic Regression baseline model
and predicts sentiment for new movie reviews.

Usage:
    python inference.py
    
Then enter reviews interactively, or modify for batch processing.
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Global state (in production, load these from saved files)
vectorizer = None
model = None

def load_model_and_vectorizer(vectorizer_obj, model_obj):
    """
    Load the trained vectorizer and model.
    
    In production, you'd load from disk:
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
    """
    global vectorizer, model
    vectorizer = vectorizer_obj
    model = model_obj

def predict_sentiment(review_text):
    """
    Predict sentiment for a single review.
    
    Args:
        review_text (str): Movie review text
        
    Returns:
        dict: {
            'prediction': 'Positive' or 'Negative',
            'confidence': float (0.0 to 1.0),
            'score_negative': float,
            'score_positive': float,
            'text': str (input review)
        }
    """
    if vectorizer is None or model is None:
        raise RuntimeError("Model not loaded. Call load_model_and_vectorizer() first.")
    
    # Vectorise the review
    X = vectorizer.transform([review_text])
    
    # Get probabilities
    proba = model.predict_proba(X)[0]
    
    # Get prediction
    pred = model.predict(X)[0]
    
    # Format output
    prediction = 'Positive' if pred == 1 else 'Negative'
    confidence = proba[pred]
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'score_negative': float(proba[0]),
        'score_positive': float(proba[1]),
        'text': review_text
    }

def batch_predict(reviews_list):
    """
    Predict sentiment for multiple reviews.
    
    Args:
        reviews_list (list): List of review strings
        
    Returns:
        list: List of prediction dicts
    """
    return [predict_sentiment(review) for review in reviews_list]

def interactive_mode():
    """
    Run interactive mode — user enters reviews, get predictions.
    """
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSER — Interactive Mode")
    print("=" * 70)
    print("Enter movie reviews and get sentiment predictions.")
    print("Type 'quit' to exit.\n")
    
    while True:
        review = input("Enter a movie review: ").strip()
        
        if review.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not review:
            print("Please enter a review.\n")
            continue
        
        result = predict_sentiment(review)
        
        print(f"\n{'─' * 70}")
        print(f"Review: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"  Negative: {result['score_negative']:.3f}")
        print(f"  Positive: {result['score_positive']:.3f}")
        print(f"{'─' * 70}\n")

if __name__ == "__main__":
    # In production, load saved models:
    # vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    # model = pickle.load(open('model.pkl', 'rb'))
    # load_model_and_vectorizer(vectorizer, model)
    
    # For demo in Colab, models are already in memory
    # Just call interactive_mode() after loading models above
    
    print("Sentiment Analyser Inference Script")
    print("Models should be loaded before calling interactive_mode()")
