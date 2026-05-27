import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.ensemble                import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration             import CalibratedClassifierCV
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import accuracy_score, classification_report, confusion_matrix

# Resolve preprocess regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text


def load_data(data_dir='data'):
    """
    Load dataset with FULL article text (not just headlines).
    Priority:
      1. ISOT True.csv + Fake.csv  — full Reuters/PolitiFact articles (best for full-text training)
      2. Indian news CSVs + Fake.csv  — headlines + ctext body
      3. Synthetic fallback
    """

    true_path  = os.path.join(data_dir, 'True.csv')
    fake_path  = os.path.join(data_dir, 'Fake.csv')
    news1_path = os.path.join(data_dir, 'news_summary.csv')
    news2_path = os.path.join(data_dir, 'news_summary_more.csv')

    # ── OPTION 1: ISOT full-text dataset (preferred) ────────────
    if os.path.exists(true_path) and os.path.exists(fake_path):
        print("Loading ISOT full-text dataset (True.csv + Fake.csv)...")
        true_df          = pd.read_csv(true_path)
        fake_df          = pd.read_csv(fake_path)
        true_df['label'] = 'REAL'
        fake_df['label'] = 'FAKE'

        # ISOT columns: title, text, subject, date
        # Combine title + full text for richer features
        for df in [true_df, fake_df]:
            df['title'] = df.get('title', pd.Series([''] * len(df))).fillna('')
            df['text']  = df.get('text',  pd.Series([''] * len(df))).fillna('')

        df = pd.concat([true_df, fake_df], ignore_index=True)
        print(f"ISOT dataset: {len(df)} articles | "
              f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")

    # ── OPTION 2: Indian news + Fake.csv ────────────────────────
    elif os.path.exists(news1_path):
        print("Loading Indian news dataset...")
        news1_df  = pd.read_csv(news1_path, encoding='iso-8859-1')
        news2_df  = pd.read_csv(news2_path, encoding='iso-8859-1') if os.path.exists(news2_path) else pd.DataFrame()
        indian_df = pd.concat([news1_df, news2_df], ignore_index=True)

        indian_df['title'] = indian_df['headlines'].fillna('')
        indian_df['text']  = indian_df['ctext'].fillna('')   # full article body
        indian_df['label'] = 'REAL'
        indian_df = indian_df[['title', 'text', 'label']]
        print(f"Indian news: {len(indian_df)} articles")

        if os.path.exists(fake_path):
            fake_df          = pd.read_csv(fake_path)
            fake_df['label'] = 'FAKE'
            fake_df['title'] = fake_df.get('title', pd.Series([''] * len(fake_df))).fillna('')
            fake_df['text']  = fake_df.get('text',  pd.Series([''] * len(fake_df))).fillna('')
            fake_df = fake_df[['title', 'text', 'label']]

            n         = min(len(indian_df), len(fake_df))
            indian_df = indian_df.sample(n, random_state=42)
            fake_df   = fake_df.sample(n,   random_state=42)
            df        = pd.concat([indian_df, fake_df], ignore_index=True)
            print(f"Balanced: {n} real + {n} fake = {len(df)} total")
        else:
            df = indian_df

    # ── OPTION 3: Synthetic fallback ────────────────────────────
    else:
        print("No CSVs found — using built-in sample dataset...")
        true_texts = [
            ("Federal Reserve raises interest rates",
             "The Federal Reserve raised its benchmark interest rate by 0.25 percentage points on Wednesday, "
             "the latest in a series of increases aimed at combating inflation that remains well above the "
             "central bank's 2 percent target. Fed Chair Jerome Powell said the decision was unanimous."),
            ("NASA confirms Mars rover landing",
             "NASA's Perseverance rover successfully touched down on Mars on Thursday, landing in the "
             "Jezero Crater as planned. Scientists celebrated at mission control as telemetry confirmed "
             "the rover survived the harrowing entry, descent and landing sequence."),
            ("Senate passes infrastructure bill",
             "The Senate passed a sweeping $1.2 trillion infrastructure bill on Tuesday with broad "
             "bipartisan support, sending the legislation to the House. The bill funds roads, bridges, "
             "broadband internet and public transit across the United States."),
        ]
        fake_texts = [
            ("BREAKING: Mind control chemicals in tap water EXPOSED",
             "A whistleblower from inside the government has revealed that fluoride added to the public "
             "water supply contains nanobots designed to make citizens docile and controllable. "
             "Mainstream media is suppressing this BOMBSHELL story. Share before it gets deleted!"),
            ("SHOCKING: Microchip found in vaccine",
             "A lab analysis of COVID vaccines has confirmed the presence of microscopic tracking devices "
             "linked to Bill Gates and the globalist agenda. Multiple doctors have been silenced for "
             "speaking out. The deep state does not want you to know this truth."),
            ("BOMBSHELL: Secret elite meeting confirms depopulation",
             "Leaked documents from a secret meeting of world leaders confirm a coordinated plan to "
             "reduce global population by 90 percent using engineered food shortages and tainted "
             "pharmaceuticals. This is not a conspiracy theory — it is happening NOW."),
        ]
        rows = ([{'title': t, 'text': b, 'label': 'REAL'} for t, b in true_texts] +
                [{'title': t, 'text': b, 'label': 'FAKE'} for t, b in fake_texts])
        df = pd.DataFrame(rows)

    # ── Combine title + full text, clean ────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Full-text content: title gets 2× weight by repeating it
    df['content'] = (
        df['title'].fillna('') + ' ' +
        df['title'].fillna('') + ' ' +   # title repeated for extra weight
        df['text'].fillna('')
    ).apply(clean_text)

    # Drop rows where cleaning produced nothing useful
    df = df[df['content'].str.split().str.len() >= 5].reset_index(drop=True)

    print(f"\nDataset ready: {len(df)} articles | "
          f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")
    return df


def _build_candidates():
    return {
        "Naive Bayes":         MultinomialNB(alpha=0.1),      # lower alpha suits larger vocab
        "Logistic Regression": LogisticRegression(
                                   max_iter=1000, C=5.0,
                                   solver='lbfgs', random_state=42),
        "Linear SVM":          CalibratedClassifierCV(
                                   LinearSVC(max_iter=3000, C=1.0,
                                             random_state=42)),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(
                                   n_estimators=150, learning_rate=0.1,
                                   max_depth=4, random_state=42),
    }


def train_model(df, model_choice='best'):
    """
    Train all candidate models on full-text TF-IDF features.
    Saves ALL models to disk so the API can switch between them at runtime.

    model_choice: 'best' | a specific model name
    Returns: (best_model, vectorizer, results_dict)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # Larger feature space for full articles
    vectorizer = TfidfVectorizer(
        max_features = 10000,   # doubled from 5000
        ngram_range  = (1, 2),
        sublinear_tf = True,
        min_df       = 2,       # ignore terms appearing in only 1 doc
        max_df       = 0.95,    # ignore near-universal terms
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    candidates = _build_candidates()
    results    = {}

    print("\n" + "=" * 60)
    print("  MULTI-MODEL COMPARISON")
    print("=" * 60)

    for name, clf in candidates.items():
        print(f"\n  Training: {name} ...", end=" ", flush=True)
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=['FAKE', 'REAL'],
            output_dict=True, zero_division=0
        )
        results[name] = {
            'model':     clf,
            'accuracy':  acc,
            'report':    report,
            'confusion': confusion_matrix(y_test, y_pred),
        }
        print(f"Accuracy = {acc*100:.2f}%")

    # ── Leaderboard ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'MODEL':<25} {'ACCURACY':>10}  {'F1-FAKE':>8}  {'F1-REAL':>8}")
    print("-" * 60)
    for name, r in sorted(results.items(),
                           key=lambda x: x[1]['accuracy'], reverse=True):
        rpt = r['report']
        print(f"  {name:<25} {r['accuracy']*100:>9.2f}%  "
              f"{rpt['FAKE']['f1-score']:>8.3f}  "
              f"{rpt['REAL']['f1-score']:>8.3f}")
    print("=" * 60)

    # ── Select final model ──────────────────────────────────────
    if model_choice in results:
        final_name  = model_choice
        final_model = results[model_choice]['model']
    else:
        final_name  = max(results, key=lambda k: results[k]['accuracy'])
        final_model = results[final_name]['model']

    print(f"\n  Selected model: {final_name}")

    # Detailed report for the winner
    y_pred_final = final_model.predict(X_test_tfidf)
    print("\n" + "=" * 60)
    print(f"  FINAL MODEL — {final_name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred_final,
                                 target_names=['FAKE', 'REAL'], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))

    # ── Save vectorizer + ALL models ────────────────────────────
    # Always save into backend/ (one level up from this file) so
    # app.py can find them regardless of where main.py is called from.
    OUT_DIR = os.path.join(os.path.dirname(__file__), '..')

    with open(os.path.join(OUT_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save each model individually so API can hot-swap
    for name, r in results.items():
        safe_name = name.lower().replace(' ', '_')
        with open(os.path.join(OUT_DIR, f'model_{safe_name}.pkl'), 'wb') as f:
            pickle.dump(r['model'], f)

    # Also save the winner as model.pkl for backwards compatibility
    with open(os.path.join(OUT_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)

    print(f"\nSaved: vectorizer.pkl + model_*.pkl for all {len(results)} models")
    print(f"Default (model.pkl) = {final_name}")

    return final_model, vectorizer, results