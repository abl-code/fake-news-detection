import pickle
from src.preprocess import clean_text

def predict_news(text, model=None, vectorizer=None):
    """
    Predict whether a news article is REAL or FAKE.
    Loads saved model from disk if not passed directly.
    """
    # Load from disk if not provided
    if model is None or vectorizer is None:
        with open('model.pkl',      'rb') as f: model      = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)

    cleaned  = clean_text(text)
    features = vectorizer.transform([cleaned])
    label    = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    conf     = max(proba) * 100

    icon = "✅" if label == "REAL" else "🚨"
    print(f"  {icon}  {label}  ({conf:.1f}% confidence)")
    return label, conf