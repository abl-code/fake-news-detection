from src.train   import load_data, train_model
from src.predict import predict_news

if __name__ == '__main__':

    # ── Step 1: Load and preprocess data ──────────────────────
    print("STEP 1 — Loading data")
    df = load_data('data')

    # ── Step 2: Train the model ────────────────────────────────
    print("\nSTEP 2 — Training model")
    model, vectorizer = train_model(df)

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