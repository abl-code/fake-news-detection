"""
train_ai_detector.py — Trains a HUMAN vs AI-GENERATED text classifier.

Separate from the fake-news pipeline on purpose: this is a STYLE
detection problem, not a content/factual one, so it uses different
features (char n-grams + stopwords KEPT + stylometric features)
and a different label space (HUMAN / AI) entirely.

Dataset: Hello-SimpleAI/HC3 (Human ChatGPT Comparison Corpus)

LEAKAGE NOTE (found via manual sample inspection, not caught by
held-out accuracy — the model hit 100.00% on the raw HC3 text):
HC3's human_answers are pulled from a pre-tokenized/detokenized corpus
and contain systematic spacing artifacts absent from chatgpt_answers,
e.g. "word ." instead of "word.", "wo n't" instead of "won't", stray
merges like "Thenagain". A char n-gram model trivially learns these
tokenization fingerprints (" .", " n't", etc.) as near-perfect HUMAN
predictors — this is a preprocessing-pipeline artifact, not a genuine
writing-style signal, and would not generalize to real-world human
text. `_normalize_spacing()` below fixes this before any feature
extraction. Class length distributions also differ substantially
(HUMAN std ~178 words vs AI std ~53 words) — this is a real property
of HC3's collection methodology (ChatGPT answers are naturally more
length-consistent than scraped Reddit answers) rather than a bug, and
is documented here rather than silently engineered away, since
downstream detectors trained on HC3 generally share this limitation.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _normalize_spacing(text):
    """
    Fix detokenization artifacts present in HC3's human_answers (and, to
    guard against reintroducing asymmetry, applied uniformly to both
    classes). Without this, punctuation-spacing patterns leak the data
    source rather than reflecting genuine human-vs-AI writing style.
    """
    if not isinstance(text, str):
        return text
    # "word ." / "word ," / "word !" -> "word." / "word," / "word!"
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # "wo n't" -> "won't"
    text = re.sub(r"\bwo\s+n't\b", "won't", text, flags=re.IGNORECASE)
    # "did n't" / "is n't" / etc. -> "didn't" / "isn't"
    text = re.sub(r"\s+n't\b", "n't", text)
    # " 's" -> "'s"  (possessive / contraction)
    text = re.sub(r"\s+'s\b", "'s", text)
    # " 're" / " 've" / " 'll" / " 'd" -> "'re" / "'ve" / "'ll" / "'d"
    text = re.sub(r"\s+'(re|ve|ll|d)\b", r"'\1", text)
    # collapse any double spaces created by the above
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def load_hc3(max_per_class=15000):
    """Pull HC3 'default' config, flatten into (text, label) pairs."""
    print("Downloading HC3 dataset from Hugging Face...")
    ds = load_dataset("Hello-SimpleAI/HC3", "default", revision="refs/convert/parquet")["train"]

    human_texts, ai_texts = [], []
    for row in ds:
        for h in row.get("human_answers", []) or []:
            if h and len(h.split()) >= 15:
                human_texts.append(_normalize_spacing(h))
        for a in row.get("chatgpt_answers", []) or []:
            if a and len(a.split()) >= 15:
                ai_texts.append(_normalize_spacing(a))

    human_texts = human_texts[:max_per_class]
    ai_texts    = ai_texts[:max_per_class]

    print(f"  HUMAN: {len(human_texts)}  |  AI: {len(ai_texts)}")

    df = pd.DataFrame({
        "text":  human_texts + ai_texts,
        "label": ["HUMAN"] * len(human_texts) + ["AI"] * len(ai_texts),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def stylometric_features(texts):
    """
    Hand-crafted style signals that classical AI-text detectors rely on.
    Returns an (n_samples, 5) numpy array:
      0: avg sentence length (words)
      1: sentence length variance (AI text tends to be too uniform)
      2: avg word length
      3: type-token ratio (lexical diversity)
      4: punctuation diversity (unique punctuation chars / total chars)
    """
    rows = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s]
        sent_lens = [len(s.split()) for s in sentences] or [0]

        words = text.split()
        word_lens = [len(w) for w in words] or [0]

        avg_sent_len = np.mean(sent_lens)
        var_sent_len = np.var(sent_lens)
        avg_word_len = np.mean(word_lens)

        unique_words = set(w.lower().strip('.,!?;:"\'') for w in words)
        ttr = len(unique_words) / max(len(words), 1)

        punct_chars = set(c for c in text if c in '.,!?;:-"\'()')
        punct_diversity = len(punct_chars) / max(len(text), 1) * 100

        rows.append([avg_sent_len, var_sent_len, avg_word_len, ttr, punct_diversity])

    return np.array(rows)


def build_features(texts, word_vec=None, char_vec=None, fit=False):
    """Combine word TF-IDF + char TF-IDF + stylometric features into one sparse matrix."""
    if fit:
        word_vec = TfidfVectorizer(
            max_features=6000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, max_df=0.95,
            stop_words=None,          # KEEP stopwords — function words matter for style
        )
        char_vec = TfidfVectorizer(
            max_features=4000, analyzer='char_wb', ngram_range=(3, 5),
            sublinear_tf=True, min_df=2,
        )
        X_word = word_vec.fit_transform(texts)
        X_char = char_vec.fit_transform(texts)
    else:
        X_word = word_vec.transform(texts)
        X_char = char_vec.transform(texts)

    X_style = csr_matrix(stylometric_features(texts))
    X = hstack([X_word, X_char, X_style]).tocsr()
    return X, word_vec, char_vec


def train_ai_detector():
    df = load_hc3()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    print("\nBuilding features (word TF-IDF + char TF-IDF + stylometrics)...")
    X_train, word_vec, char_vec = build_features(X_train_text, fit=True)
    X_test, _, _ = build_features(X_test_text, word_vec, char_vec, fit=False)

    print("\nTraining Random Forest (HUMAN vs AI)...")
    clf = RandomForestClassifier(n_estimators=250, max_depth=30, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy = {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['AI', 'HUMAN'], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    OUT_DIR = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(OUT_DIR, 'ai_detector_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(OUT_DIR, 'ai_detector_word_vec.pkl'), 'wb') as f:
        pickle.dump(word_vec, f)
    with open(os.path.join(OUT_DIR, 'ai_detector_char_vec.pkl'), 'wb') as f:
        pickle.dump(char_vec, f)

    print("\nSaved: ai_detector_model.pkl, ai_detector_word_vec.pkl, ai_detector_char_vec.pkl")
    return clf, word_vec, char_vec


if __name__ == '__main__':
    train_ai_detector()