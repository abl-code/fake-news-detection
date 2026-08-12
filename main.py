import os
from backend.src.train             import load_data, train_model
from backend.src.predict           import predict_news
from backend.src.train_ai_detector import train_ai_detector

if __name__ == '__main__':

    # ── Step 1: Fake news model ────────────────────────────────
    print("STEP 1 — Loading fake-news data")
    df = load_data(os.path.join('backend', 'data'))

    print("\nSTEP 2 — Training fake-news model (Random Forest)")
    model, vectorizer, results = train_model(df)

    print("\nSTEP 3 — Sample predictions")
    print("-" * 50)
    headlines = [
        "Federal Reserve raises interest rates to combat inflation",
        "BREAKING: Government putting mind control chemicals in tap water EXPOSED",
        "Senate passes bipartisan infrastructure bill after months of debate",
        "SHOCKING: Secret elite globalist meeting confirms depopulation agenda",
    ]
    for headline in headlines:
        print(f"\n  Input : {headline}")
        predict_news(headline, model, vectorizer)

    # ── Step 4: AI-text detector (separate pipeline) ───────────
    print("\nSTEP 4 — Training AI-text detector")
    train_ai_detector()

    print("\nAll models trained and saved.")