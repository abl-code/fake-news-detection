import os
from backend.src.train             import train_all_languages
from backend.src.train_ai_detector import train_ai_detector

if __name__ == '__main__':

    # ── Step 1+2: Fake-news models, all languages ──────────────
    print("STEP 1/2 — Loading data + training fake-news models (Random Forest)")
    train_all_languages(
        data_dir=os.path.join('backend', 'data'),
        languages=('en', 'hi'),
    )

    # ── Step 3: AI-text detector (separate pipeline, unchanged) ─
    print("\nSTEP 3 — Training AI-text detector")
    train_ai_detector()

    print("\nAll models trained and saved.")