import os
import pickle
from .preprocess import clean_text

# Resolve model/vectorizer paths relative to THIS file's location, same
# convention train.py's OUT_DIR uses — not relative to the process's
# current working directory, which varies depending on how/where
# main.py is invoked from (e.g. /app inside Docker vs backend/ locally).
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..')


def predict_news(text, model=None, vectorizer=None):
    """
    Predict whether a news article is REAL or FAKE.
    Loads saved model from disk if not passed directly.
    """
    if model is None or vectorizer is None:
        with open(os.path.join(_MODEL_DIR, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)

    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    label    = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    conf     = max(proba) * 100

    icon = "✅" if label == "REAL" else "🚨"
    print(f"  {icon}  {label}  ({conf:.1f}% confidence)")
    return label, conf