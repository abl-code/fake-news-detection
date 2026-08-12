import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble                import RandomForestClassifier
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import accuracy_score, classification_report, confusion_matrix

from .preprocess import clean_text

def _strip_reuters_tag(text):
    """Remove the '(Reuters) -' wire-service dateline that appears in
    ~99% of REAL articles and 0% of FAKE ones — pure source leakage,
    not a truthfulness signal."""
    if not isinstance(text, str):
        return text
    return re.sub(r'^.{0,80}?\(Reuters\)\s*-\s*', '', text)
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
    Load and COMBINE every dataset found under data_dir. Any source not
    present on disk is skipped.
    """
    frames = []
    print("Scanning for datasets...")

    root_true = os.path.join(data_dir, 'True.csv')
    root_fake = os.path.join(data_dir, 'Fake.csv')
    if os.path.exists(root_true) and os.path.exists(root_fake):
        frames.append(_load_isot_pair(root_true, root_fake, 'data/True.csv + data/Fake.csv'))

    if not frames:
        raise FileNotFoundError(
            f"No datasets found under '{data_dir}'. Expected True.csv/Fake.csv."
        )
    # NOTE: 'subject' column deliberately excluded from features — it's
    # ~100% deterministic on label (leakage), see check_leakage.py.
    df = pd.concat(frames, ignore_index=True)[['title', 'text', 'label']]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['text'] = df['text'].apply(_strip_reuters_tag)

    df['content'] = (
        df['title'].fillna('') + ' ' +
        df['title'].fillna('') + ' ' +
        df['text'].fillna('')
    ).apply(clean_text)

    df = df[df['content'].str.split().str.len() >= 5].reset_index(drop=True)

    print(f"\nDataset ready: {len(df)} articles | "
          f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")
    return df


def train_model(df):
    """
    Train a single Random Forest classifier on TF-IDF features.
    Saves vectorizer.pkl and model.pkl to backend/.
    Returns: (model, vectorizer, results_dict) — results_dict kept as a
    dict (single entry) so callers/lab-report tooling expecting that
    shape from the old multi-model version still work.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(
        max_features = 10000,
        ngram_range  = (1, 2),
        sublinear_tf = True,
        min_df       = 2,
        max_df       = 0.95,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print("\n" + "=" * 60)
    print("  TRAINING — Random Forest")
    print("=" * 60)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=['FAKE', 'REAL'],
        output_dict=True, zero_division=0
    )

    print(f"\nAccuracy = {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    OUT_DIR = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(OUT_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(OUT_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    print("\nSaved: vectorizer.pkl, model.pkl (Random Forest)")

    results = {
        'Random Forest': {
            'model': clf, 'accuracy': acc, 'report': report,
            'confusion': confusion_matrix(y_test, y_pred),
        }
    }
    return clf, vectorizer, results