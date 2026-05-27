import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backend.src.train   import load_data, train_model
from backend.src.predict import predict_news

if __name__ == '__main__':

    # ── Step 1: Load and preprocess data ──────────────────────
    print("STEP 1 — Loading data")
    df = load_data(os.path.join('backend', 'data'))

    # ── Step 2: Train all models, select best ─────────────────
    print("\nSTEP 2 — Training models")
    model, vectorizer, results = train_model(df, model_choice='best')

    # ── Step 3: Test with sample headlines ────────────────────
    print("\nSTEP 3 — Sample predictions")
    print("-" * 50)

    headlines = [
        "Federal Reserve raises interest rates to combat inflation",
        "BREAKING: Government putting mind control chemicals in tap water EXPOSED",
        "Senate passes bipartisan infrastructure bill after months of debate",
        "SHOCKING: Secret elite globalist meeting confirms depopulation agenda",
        "Scientists confirm record ocean temperatures for third year running",
        "BOMBSHELL: Hillary Clinton arrested — mainstream media hiding the truth",
    ]

    for headline in headlines:
        print(f"\n  Input : {headline}")
        predict_news(headline, model, vectorizer)