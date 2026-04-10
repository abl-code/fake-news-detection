import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import (
    accuracy_score, classification_report, confusion_matrix
)
from src.preprocess import clean_text


def load_data(data_dir='data'):
    """Load Indian news as REAL, pair with Fake.csv for FAKE labels."""

    # ── OPTION 1: Indian news dataset (Times of India etc.) ────
    try:
        print("Loading Indian news dataset...")

        news1_path = os.path.join(data_dir, 'news_summary.csv')
        news2_path = os.path.join(data_dir, 'news_summary_more.csv')
        fake_path  = os.path.join(data_dir, 'Fake.csv')

        if not os.path.exists(news1_path):
            raise Exception("Indian news CSV not found")

        # Load both Indian news files
        news1_df = pd.read_csv(news1_path, encoding='iso-8859-1')
        news2_df = pd.read_csv(news2_path, encoding='iso-8859-1')
        indian_df = pd.concat([news1_df, news2_df], ignore_index=True)

        # Use headlines + ctext as content
        indian_df['title'] = indian_df['headlines'].fillna('')
        indian_df['text']  = indian_df['ctext'].fillna('')
        indian_df['label'] = 'REAL'
        indian_df = indian_df[['title', 'text', 'label']]

        print(f"Indian news loaded: {len(indian_df)} articles")

        # Load fake news from Fake.csv to pair with Indian real news
        if os.path.exists(fake_path):
            fake_df = pd.read_csv(fake_path)
            fake_df['label'] = 'FAKE'

            # Sample same number of fake articles as real for balance
            n = min(len(indian_df), len(fake_df))
            indian_df = indian_df.sample(n, random_state=42)
            fake_df   = fake_df.sample(n,   random_state=42)

            # Rename fake columns to match
            if 'title' not in fake_df.columns:
                fake_df = fake_df.rename(columns={fake_df.columns[0]: 'title'})
            fake_df['text'] = fake_df.get('text', '')

            fake_df = fake_df[['title', 'text', 'label']]
            df = pd.concat([indian_df, fake_df], ignore_index=True)
            print(f"Paired with {n} fake articles for balance")

        else:
            df = indian_df
            print("No Fake.csv found — using Indian news only")

    # ── OPTION 2: Fall back to True.csv + Fake.csv ─────────────
    except Exception as e:
        print(f"Indian news load failed ({e}) — trying True/Fake CSVs...")
        true_path = os.path.join(data_dir, 'True.csv')
        fake_path = os.path.join(data_dir, 'Fake.csv')

        if os.path.exists(true_path) and os.path.exists(fake_path):
            true_df          = pd.read_csv(true_path)
            fake_df          = pd.read_csv(fake_path)
            true_df['label'] = 'REAL'
            fake_df['label'] = 'FAKE'
            df               = pd.concat([true_df, fake_df], ignore_index=True)
            print(f"Local CSV dataset loaded: {len(df)} articles")

        # ── OPTION 3: Fall back to synthetic data ──────────────
        else:
            print("No CSVs found — using built-in sample dataset...")
            true_df = pd.DataFrame({'title': [
                "Federal Reserve raises interest rates to combat inflation",
                "NASA confirms successful Mars rover landing",
                "Senate passes bipartisan infrastructure bill after months of debate",
                "WHO declares end to public health emergency as cases decline",
                "Scientists discover new treatment for drug-resistant bacteria",
                "Supreme Court rules on landmark digital privacy case",
                "Treasury announces new economic sanctions against foreign entity",
                "FDA approves new cancer immunotherapy drug after trials",
                "UN Security Council holds emergency session on climate report",
                "Stock markets fall sharply as inflation data exceeds expectations",
            ], 'text': [''] * 10})
            fake_df = pd.DataFrame({'title': [
                "BREAKING: Government putting mind control chemicals in tap water EXPOSED",
                "SHOCKING: Bill Gates microchip found in vaccine — share before deleted",
                "BOMBSHELL: Deep state planning false flag attack this weekend URGENT",
                "MUST READ: Secret elite globalist meeting confirms depopulation agenda",
                "EXPOSED: 5G towers activating vaccine ingredients inside your body now",
                "BREAKING: Hillary Clinton arrested — mainstream media hiding the truth",
                "REVEALED: UN troops positioned in underground bunkers across US cities",
                "BOMBSHELL: New study proves climate change fabricated by globalist agenda",
                "SHOCKING: Mainstream media anchor admits all news is scripted on hot mic",
                "ALERT: New digital dollar will let government freeze your spending instantly",
            ], 'text': [''] * 10})
            true_df['label'] = 'REAL'
            fake_df['label'] = 'FAKE'
            df = pd.concat([true_df, fake_df], ignore_index=True)

    # ── Shuffle + clean ─────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['content'] = (
        df['title'].fillna('') + ' ' + df['text'].fillna('')
    ).apply(clean_text)

    print(f"Dataset ready: {len(df)} articles | "
          f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")
    return df


def train_model(df):
    """Vectorize text with TF-IDF and train a Multinomial Naive Bayes model."""

    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # TF-IDF: converts text into numerical feature vectors
    vectorizer = TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),   # single words + two-word phrases
        sublinear_tf = True      # log-scale term frequency
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Train Naive Bayes
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print("="*50)

    print(classification_report(
        y_test, y_pred,
        target_names=['FAKE', 'REAL'],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("  Confusion matrix explained:")
    print("  [ TN  FP ]   TN = real correctly identified as real")
    print("  [ FN  TP ]   TP = fake correctly identified as fake")
    print("               FP = real wrongly called fake")
    print("               FN = fake wrongly called real")

    # Save model and vectorizer to disk for use in predict.py
    with open('model.pkl',      'wb') as f: pickle.dump(model,      f)
    with open('vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)
    print("\nModel saved: model.pkl + vectorizer.pkl")

    return model, vectorizer