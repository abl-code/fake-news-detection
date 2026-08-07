import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import LogisticRegression
from sklearn.tree                    import DecisionTreeClassifier
from sklearn.ensemble                import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import accuracy_score, classification_report, confusion_matrix

from .preprocess import clean_text


def _load_isot_pair(true_path, fake_path, source_label):
    """Load one True/Fake CSV pair in ISOT format (title, text, subject, date)."""
    true_df          = pd.read_csv(true_path)
    fake_df          = pd.read_csv(fake_path)
    true_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'

    for df in [true_df, fake_df]:
        df['title'] = df.get('title', pd.Series([''] * len(df))).fillna('')
        df['text']  = df.get('text',  pd.Series([''] * len(df))).fillna('')

    df = pd.concat([true_df, fake_df], ignore_index=True)[['title', 'text', 'label']]
    print(f"  {source_label}: {len(df)} articles | "
          f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")
    return df


def load_data(data_dir='data'):
    """
    Load and COMBINE every dataset found under data_dir:
      1. data/True.csv               + data/Fake.csv
      2. data/News _dataset/True.csv + data/News _dataset/Fake.csv
      3. data/news_summary.csv       + data/news_summary_more.csv  (REAL headlines,
         paired 1:1 with sampled FAKE.csv rows to keep classes balanced)
    Any source that isn't present on disk is simply skipped.
    """

    frames = []

    print("Scanning for datasets...")

    # ── ISOT pair at data root ──────────────────────────────────
    root_true = os.path.join(data_dir, 'True.csv')
    root_fake = os.path.join(data_dir, 'Fake.csv')
    if os.path.exists(root_true) and os.path.exists(root_fake):
        frames.append(_load_isot_pair(root_true, root_fake, 'data/True.csv + data/Fake.csv'))

    # ── ISOT pair inside News _dataset subfolder ────────────────
    sub_dir  = os.path.join(data_dir, 'News _dataset')
    sub_true = os.path.join(sub_dir, 'True.csv')
    sub_fake = os.path.join(sub_dir, 'Fake.csv')
    if os.path.exists(sub_true) and os.path.exists(sub_fake):
        frames.append(_load_isot_pair(sub_true, sub_fake, 'News _dataset/True.csv + Fake.csv'))

    # ── Indian news headlines (REAL) balanced against Fake.csv ──
    news1_path = os.path.join(data_dir, 'news_summary.csv')
    news2_path = os.path.join(data_dir, 'news_summary_more.csv')
    if os.path.exists(news1_path):
        news1_df  = pd.read_csv(news1_path, encoding='iso-8859-1')
        news2_df  = pd.read_csv(news2_path, encoding='iso-8859-1') if os.path.exists(news2_path) else pd.DataFrame()
        indian_df = pd.concat([news1_df, news2_df], ignore_index=True)

        indian_df['title'] = indian_df['headlines'].fillna('')
        indian_df['text']  = indian_df['ctext'].fillna('')
        indian_df['label'] = 'REAL'
        indian_df = indian_df[['title', 'text', 'label']]
        print(f"  Indian news (news_summary*.csv): {len(indian_df)} REAL articles")

        # Balance against a FAKE source if one exists
        fake_source = root_fake if os.path.exists(root_fake) else (sub_fake if os.path.exists(sub_fake) else None)
        if fake_source:
            fake_df          = pd.read_csv(fake_source)
            fake_df['label'] = 'FAKE'
            fake_df['title'] = fake_df.get('title', pd.Series([''] * len(fake_df))).fillna('')
            fake_df['text']  = fake_df.get('text',  pd.Series([''] * len(fake_df))).fillna('')
            fake_df = fake_df[['title', 'text', 'label']]

            n         = min(len(indian_df), len(fake_df))
            indian_df = indian_df.sample(n, random_state=42)
            fake_df   = fake_df.sample(n,   random_state=42)
            print(f"  Balancing Indian news with {n} extra sampled FAKE rows")
            frames.append(pd.concat([indian_df, fake_df], ignore_index=True))
        else:
            frames.append(indian_df)

    if not frames:
        raise FileNotFoundError(
            f"No datasets found under '{data_dir}'. Expected True.csv/Fake.csv "
            f"(at data root and/or in a 'News _dataset' subfolder) and/or "
            f"news_summary.csv."
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"\nCombined raw dataset: {len(df)} articles from {len(frames)} source(s) | "
          f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")

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
        "Decision Tree":       DecisionTreeClassifier(
                                   max_depth=20, random_state=42),
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

    model_choice: 'best' | 'Gradient Boosting' (default, matches backend
                  DEFAULT_MODEL) | any other specific model name
    Returns: (final_model, vectorizer, results_dict)
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