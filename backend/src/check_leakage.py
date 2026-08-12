"""
check_leakage.py — Diagnoses whether the fake-news RF model is learning
source/formatting artifacts (e.g. Reuters wire-story prefixes) rather
than genuine content signal. Read-only: doesn't touch your saved models.

Run: python -m backend.src.check_leakage   (from repo root)
"""

import os
import re
import pandas as pd
from collections import Counter

DATA_DIR = os.path.join('backend', 'data')


def load_raw():
    true_df = pd.read_csv(os.path.join(DATA_DIR, 'True.csv'))
    fake_df = pd.read_csv(os.path.join(DATA_DIR, 'Fake.csv'))
    true_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'
    return pd.concat([true_df, fake_df], ignore_index=True)


def check_reuters_prefix(df):
    """ISOT REAL articles are almost all Reuters wire copy and often start
    with a dateline like 'WASHINGTON (Reuters) -'. If that prefix predicts
    the label almost perfectly, the model can shortcut on formatting
    instead of content."""
    pattern = re.compile(r'\(Reuters\)', re.IGNORECASE)
    df['has_reuters_tag'] = df['text'].fillna('').apply(lambda t: bool(pattern.search(t)))

    print("\n--- Reuters-tag vs label ---")
    ct = pd.crosstab(df['has_reuters_tag'], df['label'])
    print(ct)
    real_with_tag = ct.loc[True, 'REAL'] / ct['REAL'].sum() * 100 if True in ct.index else 0
    fake_with_tag = ct.loc[True, 'FAKE'] / ct['FAKE'].sum() * 100 if True in ct.index else 0
    print(f"\n{real_with_tag:.1f}% of REAL articles contain '(Reuters)'")
    print(f"{fake_with_tag:.1f}% of FAKE articles contain '(Reuters)'")
    if real_with_tag > 80 and fake_with_tag < 20:
        print("⚠️  LEAKAGE SIGNAL: '(Reuters)' tag alone is a near-perfect label predictor.")
    return ct


def check_subject_column(df):
    """ISOT's 'subject' column often perfectly separates REAL/FAKE
    (e.g. 'politicsNews' vs 'News'/'left-news') even though it's
    metadata, not content — another leakage vector if it ever
    leaked into training features."""
    if 'subject' not in df.columns:
        print("\n(no 'subject' column found — skip)")
        return
    print("\n--- subject vs label ---")
    ct = pd.crosstab(df['subject'], df['label'])
    print(ct)
    for subj in ct.index:
        row = ct.loc[subj]
        total = row.sum()
        dominant = row.idxmax()
        pct = row.max() / total * 100
        if pct > 95:
            print(f"⚠️  subject='{subj}' is {pct:.1f}% {dominant} — near-perfect leakage if used as a feature.")


def check_title_length_bias(df):
    """Sanity check: are FAKE headlines systematically longer/shorter
    or more punctuation-heavy in a way a short manual test headline
    wouldn't match?"""
    df['title_len']       = df['title'].fillna('').apply(lambda t: len(t.split()))
    df['title_has_caps']  = df['title'].fillna('').apply(lambda t: bool(re.search(r'\b[A-Z]{4,}\b', t)))
    df['title_has_bang']  = df['title'].fillna('').apply(lambda t: '!' in t)

    print("\n--- Title style stats by label ---")
    print(df.groupby('label')[['title_len']].mean())
    print(df.groupby('label')['title_has_caps'].mean() * 100, "% ALL-CAPS words")
    print(df.groupby('label')['title_has_bang'].mean() * 100, "% contain '!'")


def check_manual_headlines(df):
    """Directly test the exact headlines from main.py's Step 3 against
    the raw dataset's own average length/style to see if they're
    out-of-distribution for REAL articles specifically."""
    real_avg_len = df[df.label == 'REAL']['title'].fillna('').apply(lambda t: len(t.split())).mean()
    fake_avg_len = df[df.label == 'FAKE']['title'].fillna('').apply(lambda t: len(t.split())).mean()

    test_headlines = [
        "Federal Reserve raises interest rates to combat inflation",
        "Senate passes bipartisan infrastructure bill after months of debate",
    ]
    print("\n--- Manual test headline lengths vs dataset averages ---")
    print(f"Dataset REAL avg title length: {real_avg_len:.1f} words")
    print(f"Dataset FAKE avg title length: {fake_avg_len:.1f} words")
    for h in test_headlines:
        print(f"  '{h}' -> {len(h.split())} words")


if __name__ == '__main__':
    df = load_raw()
    print(f"Loaded {len(df)} articles ({(df.label=='REAL').sum()} REAL, {(df.label=='FAKE').sum()} FAKE)")

    check_reuters_prefix(df)
    check_subject_column(df)
    check_title_length_bias(df)
    check_manual_headlines(df)

    print("\nDone. If leakage signals fired above, this explains why held-out")
    print("accuracy is ~99.8% yet short hand-written headlines get misclassified:")
    print("the model learned Reuters-wire formatting/source artifacts, not")
    print("generalizable claims about truthfulness. Worth a paragraph in your")
    print("report's Limitations section — this is a known, published issue")
    print("with the ISOT dataset, not a bug in your pipeline.")
    