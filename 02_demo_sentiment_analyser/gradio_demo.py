"""
gradio_demo.py — Sentiment Analysis Web Interface

Interactive Gradio demo for sentiment analysis.
Run with: python gradio_demo.py

Opens web interface at http://localhost:7860
"""

import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Global state (loaded at startup)
vectorizer = None
model = None

def load_models(vec, clf):
    """Load models into global state."""
    global vectorizer, model
    vectorizer = vec
    model = clf

def predict_fn(review_text):
    """
    Gradio interface function.
    Takes review text, returns formatted prediction.
    """
    if not review_text.strip():
        return "Please enter a review."
    
    if vectorizer is None or model is None:
        return "Error: Models not loaded."
    
    # Vectorise
    X = vectorizer.transform([review_text])
    
    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    # Format output
    sentiment = 'Positive 😊' if pred == 1 else 'Negative 😞'
    confidence = proba[pred]
    
    # Create detailed output
    output = f"""
**Sentiment:** {sentiment}
**Confidence:** {confidence:.1%}

---

**Score Breakdown:**
- Negative: {proba[0]:.3f}
- Positive: {proba[1]:.3f}
"""
    return output

def create_interface():
    """Create and return Gradio interface."""
    interface = gr.Interface(
        fn=predict_fn,
        inputs=gr.Textbox(
            label="Enter a Movie Review",
            placeholder="Example: This movie was amazing!",
            lines=4
        ),
        outputs=gr.Markdown(),
        title="🎬 Movie Sentiment Analyser",
        description="Predict whether a movie review is positive or negative using a TF-IDF + Logistic Regression model trained on IMDB data.",
        examples=[
            ["This movie was absolutely fantastic! Best film I've seen in years."],
            ["Terrible waste of time. Couldn't finish it."],
            ["It was okay, nothing special but entertaining enough."],
            ["Brilliant cinematography and outstanding performances. Highly recommend!"],
            ["Horrible plot and bad acting. Save your money."],
        ],
        theme="default"
    )
    return interface

if __name__ == "__main__":
    # In production, load saved models:
    # import pickle
    # vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    # model = pickle.load(open('model.pkl', 'rb'))
    
    # For now, create and launch
    interface = create_interface()
    interface.launch(share=True)  # share=True creates a public link
