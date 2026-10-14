import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble                import RandomForestClassifier
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import accuracy_score, classification_report, confusion_matrix

from .preprocess import clean_text

# ── Per-language artifact naming ─────────────────────────────────────
# English keeps the original unsuffixed filenames (model.pkl,
# vectorizer.pkl) so nothing that already loads those two files by name
# (app.py's old load path, test_predictions.py's load_artifacts()) breaks.
# Every other language gets a suffixed pair: model_hi.pkl, vectorizer_hi.pkl.
def _artifact_names(lang):
    if lang == 'en':
        return 'model.pkl', 'vectorizer.pkl'
    return f'model_{lang}.pkl', f'vectorizer_{lang}.pkl'


def _strip_reuters_tag(text):
    """Remove the '(Reuters) -' wire-service dateline that appears in
    ~99% of REAL articles and 0% of FAKE ones in the ISOT (English)
    dataset — pure source leakage, not a truthfulness signal.
    English-only artifact; no-op for other languages/datasets unless a
    similar tag is confirmed present and leaking there too."""
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


def _load_hinfakenews(csv_path, source_label='HinFakeNews'):
    """
    Load the HinFakeNews (IndiaAI) dataset — a single CSV, not a
    True/Fake pair like ISOT. ~68k Hindi news articles (political,
    social, communal domains): real articles sourced from RNI-registered
    outlets, fake from IFCN-approved fact-checkers.

    Expected columns: title, article content (or 'content'/'text'), url,
    and a binary label column where 0 = FAKE, 1 = REAL. Column names are
    matched case-insensitively / with common variants since the exact
    header spelling in the downloaded CSV isn't guaranteed to match the
    dataset-page description verbatim.
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    def _find(*candidates):
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    title_col = _find('title')
    text_col  = _find('article content', 'content', 'article_content', 'text')
    label_col = _find('label', 'binary label', 'binary_label')

    if text_col is None or label_col is None:
        raise ValueError(
            f"{source_label}: couldn't find expected columns in {csv_path}. "
            f"Found columns: {list(df.columns)}. Expected a title column, a "
            f"content/text column, and a 0/1 label column — update the "
            f"_find(...) candidates above to match this file's actual headers."
        )

    out = pd.DataFrame({
        'title': df[title_col].fillna('') if title_col else '',
        'text':  df[text_col].fillna(''),
        'label': df[label_col].map({0: 'FAKE', 1: 'REAL', '0': 'FAKE', '1': 'REAL'}),
    })

    unmapped = out['label'].isna().sum()
    if unmapped:
        print(f"  [warn] {source_label}: {unmapped} rows had an unrecognized "
              f"label value (not 0/1) and were dropped.")
        out = out.dropna(subset=['label'])

    print(f"  {source_label}: {len(out)} articles | "
          f"{(out.label=='REAL').sum()} real, {(out.label=='FAKE').sum()} fake")
    return out.reset_index(drop=True)

def _load_hfdnd(hfdnd_dir, source_label='HFDND'):
    """
    Load the Hindi Fake News Detection Dataset (HFDND) — MIT-licensed,
    Manipal Institute of Technology. Unlike HinFakeNews, this dataset has
    NO title column: just an index, a `text` field, an int `label`
    (1 = Fake, 0 = True per the dataset's Kaggle listing), and a
    precomputed `wcount` we don't need (we compute our own post-cleaning
    word count in load_data()).
    """
    import glob
    candidates = sorted(glob.glob(os.path.join(hfdnd_dir, '*.csv')))
    if not candidates:
        raise FileNotFoundError(f"{source_label}: no .csv file found under {hfdnd_dir}")
    csv_path = candidates[0]
    if len(candidates) > 1:
        print(f"  [warn] {source_label}: multiple CSVs found in {hfdnd_dir} "
              f"({[os.path.basename(c) for c in candidates]}), using {os.path.basename(csv_path)}")

    df = pd.read_csv(csv_path)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    def _find(*candidates):
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    text_col  = _find('text')
    label_col = _find('label')

    if text_col is None or label_col is None:
        raise ValueError(
            f"{source_label}: couldn't find expected columns in {csv_path}. "
            f"Found columns: {list(df.columns)}. Expected a 'text' column and "
            f"a 0/1 'label' column."
        )

    out = pd.DataFrame({
        'title': '',   # HFDND has no title field — text-only dataset
        'text':  df[text_col].fillna(''),
        'label': df[label_col].map({1: 'FAKE', 0: 'REAL', '1': 'FAKE', '0': 'REAL'}),
    })

    unmapped = out['label'].isna().sum()
    if unmapped:
        print(f"  [warn] {source_label}: {unmapped} rows had an unrecognized "
              f"label value (not 0/1) and were dropped.")
        out = out.dropna(subset=['label'])

    dupes = out.duplicated(subset=['text']).sum()
    if dupes:
        print(f"  [info] {source_label}: {dupes} duplicate text rows present "
              f"(not auto-dropped — verify intentionally before training if this seems high).")

    print(f"  {source_label} ({os.path.basename(csv_path)}): {len(out)} articles | "
          f"{(out.label=='REAL').sum()} real, {(out.label=='FAKE').sum()} fake")
    return out.reset_index(drop=True)
def load_data(data_dir='data', lang='en'):
    """
    Load and COMBINE every dataset found under data_dir/<lang>/. Any
    source not present on disk is skipped.

    English keeps its original flat layout (data/True.csv, data/Fake.csv)
    for backward compatibility. Non-English languages use a
    data/<lang>/ subfolder so multiple language datasets can coexist
    without filename collisions — e.g. data/hi/True.csv, data/hi/Fake.csv.
    """
    frames = []
    print(f"Scanning for datasets (lang={lang})...")

    if lang == 'en':
        root_true = os.path.join(data_dir, 'True.csv')
        root_fake = os.path.join(data_dir, 'Fake.csv')
        if os.path.exists(root_true) and os.path.exists(root_fake):
            frames.append(_load_isot_pair(root_true, root_fake, 'data/True.csv + data/Fake.csv'))
    elif lang == 'hi':
        # Text-only (no title), int label where
        # 1 = Fake, 0 = True.
        hfdnd_dir = os.path.join(data_dir, 'HFDND')
        if os.path.isdir(hfdnd_dir):
            frames.append(_load_hfdnd(hfdnd_dir))


        # Optional additional Hindi source(s) in ISOT-pair shape, if you
        # later add a second dataset alongside HinFakeNews for more
        # coverage/domain diversity. Skipped if not present.
        lang_true = os.path.join(data_dir, lang, 'True.csv')
        lang_fake = os.path.join(data_dir, lang, 'Fake.csv')
        if os.path.exists(lang_true) and os.path.exists(lang_fake):
            frames.append(_load_isot_pair(lang_true, lang_fake,
                                           f'data/{lang}/True.csv + data/{lang}/Fake.csv'))
    else:
        lang_true = os.path.join(data_dir, lang, 'True.csv')
        lang_fake = os.path.join(data_dir, lang, 'Fake.csv')
        if os.path.exists(lang_true) and os.path.exists(lang_fake):
            frames.append(_load_isot_pair(lang_true, lang_fake,
                                           f'data/{lang}/True.csv + data/{lang}/Fake.csv'))

    if not frames:
        if lang == 'en':
            expected = f"'{data_dir}/True.csv' + '{data_dir}/Fake.csv'"
        elif lang == 'hi':
            expected = (f"'{data_dir}/HFDND/*.csv' "
                        f"(and/or '{data_dir}/hi/True.csv' + '{data_dir}/hi/Fake.csv')")
        else:
            expected = f"'{data_dir}/{lang}/True.csv' + '{data_dir}/{lang}/Fake.csv'"
        raise FileNotFoundError(
            f"No datasets found for lang='{lang}'. Expected {expected}. "
            f"See README's dataset setup section for sourcing options."
        )

    # NOTE: 'subject' column deliberately excluded from features — it's
    # ~100% deterministic on label in ISOT (leakage), see check_leakage.py.
    # Re-verify this holds for any new non-English dataset before reusing
    # the same exclusion — leakage sources are dataset-specific.
    df = pd.concat(frames, ignore_index=True)[['title', 'text', 'label']]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if lang == 'en':
        df['text'] = df['text'].apply(_strip_reuters_tag)

    df['content'] = (
        df['title'].fillna('') + ' ' +
        df['title'].fillna('') + ' ' +
        df['text'].fillna('')
    ).apply(lambda t: clean_text(t, lang=lang))

    df = df[df['content'].str.split().str.len() >= 5].reset_index(drop=True)

    print(f"\nDataset ready ({lang}): {len(df)} articles | "
          f"{(df.label=='REAL').sum()} real, {(df.label=='FAKE').sum()} fake")
    return df


def train_model(df, lang='en'):
    """
    Train a single Random Forest classifier on TF-IDF features for the
    given language. Saves the language's vectorizer/model pair to
    backend/ (see _artifact_names()).

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
    print(f"  TRAINING — Random Forest ({lang})")
    print("=" * 60)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=['FAKE', 'REAL'],
        output_dict=True, zero_division=0
    )

    print(f"\nAccuracy ({lang}) = {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_name, vec_name = _artifact_names(lang)
    OUT_DIR = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(OUT_DIR, vec_name), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(OUT_DIR, model_name), 'wb') as f:
        pickle.dump(clf, f)

    print(f"\nSaved: {vec_name}, {model_name} (Random Forest, {lang})")

    results = {
        f'Random Forest ({lang})': {
            'model': clf, 'accuracy': acc, 'report': report,
            'confusion': confusion_matrix(y_test, y_pred),
        }
    }
    return clf, vectorizer, results


def train_all_languages(data_dir='data', languages=('en',)):
    """
    Convenience wrapper: load_data + train_model for each language in
    `languages`, skipping (with a warning) any language whose dataset
    isn't present on disk rather than aborting the whole run.
    Called from main.py.
    """
    all_results = {}
    for lang in languages:
        try:
            df = load_data(data_dir=data_dir, lang=lang)
        except FileNotFoundError as e:
            print(f"\n[skip] {lang}: {e}")
            continue
        _, _, results = train_model(df, lang=lang)
        all_results.update(results)
    return all_results
