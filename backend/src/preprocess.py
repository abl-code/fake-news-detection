r"""
preprocess.py — Text cleaning pipeline.

Now language-aware. Originally English-only (`re.sub(r'[^a-z\s]', '',
text)`), which silently stripped every non-Latin character — meaning any
Hindi (Devanagari) text fed through the old clean_text() came out as an
empty string. That's the root blocker for Hindi support, not a config
change: fixed here by branching on `lang` and giving each language its
own script-preserving regex + stopword list.

Malayalam is NOT handled yet — no labeled real/fake Malayalam dataset is
in hand (see project notes), so there is no Malayalam branch here. Adding
one is a ~5-line change (Malayalam Unicode block is \u0D00-\u0D7F) once
data exists; deliberately not stubbed in now to avoid a branch nothing
trains or tests against.

Supported languages: 'en' (default), 'hi'.
"""

import re

# ── Stopwords ─────────────────────────────────────────────────────────
# English: kept as a static list (compact, no NLTK dependency at import
# time). Swap for nltk.corpus.stopwords if you already pull NLTK elsewhere.
EN_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'to', 'of', 'in', 'on', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
    'this', 'that', 'these', 'those', 'it', 'its', 'as', 'so', 'than',
    'too', 'very', 'can', 'will', 'just', 'not', 'no', 'nor', 'do', 'does',
    'did', 'has', 'have', 'had', 'i', 'you', 'he', 'she', 'we', 'they',
}

# Hindi stopwords — common function words (postpositions, pronouns,
# conjunctions, auxiliary verbs). Sourced from the standard AI4Bharat /
# stopwords-iso Hindi list, trimmed to the highest-frequency subset. Swap
# in the full AI4Bharat list for production if recall on rarer stopwords
# matters more than keeping this file dependency-free.
HI_STOPWORDS = {
    'का', 'के', 'की', 'को', 'कि', 'कर', 'करते', 'करता', 'करने', 'किया',
    'है', 'हैं', 'था', 'थे', 'थी', 'हो', 'होता', 'होती', 'होंगे', 'हुआ',
    'हुई', 'और', 'या', 'एक', 'यह', 'वह', 'ये', 'वे', 'इस', 'उस', 'इन',
    'उन', 'भी', 'तो', 'ही', 'न', 'नहीं', 'पर', 'से', 'में', 'तक', 'लिए',
    'साथ', 'अपने', 'अपनी', 'उनके', 'उनकी', 'जो', 'जब', 'तब', 'क्या',
    'क्यों', 'कैसे', 'कहा', 'कहते', 'गया', 'गई', 'रहा', 'रही', 'रहे',
}

# ── Per-language Unicode script ranges ──────────────────────────────────
# Devanagari block (covers Hindi, plus Marathi/Sanskrit — fine, since only
# Hindi is routed here).
_HI_SCRIPT_RE = re.compile(r'[^\u0900-\u097F\s]')
_EN_SCRIPT_RE = re.compile(r'[^a-z\s]')
_WHITESPACE_RE = re.compile(r'\s+')
_HI_CHAR_RE = re.compile(r'[\u0900-\u097F]')

def _clean_english(text):
    text = text.lower()
    text = _EN_SCRIPT_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    tokens = [w for w in text.split() if w not in EN_STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


def _clean_hindi(text):
    # No .lower() — Devanagari has no case. Strip everything outside the
    # Devanagari block instead of the old Latin-only regex, which is what
    # was silently deleting Hindi text entirely.
    text = _HI_SCRIPT_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    tokens = [w for w in text.split() if w not in HI_STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


_CLEANERS = {
    'en': _clean_english,
    'hi': _clean_hindi,
}


def clean_text(text, lang='en'):
    """
    Clean `text` for the given language before TF-IDF vectorization.

    lang: 'en' (default, backward-compatible with every existing call
          site that doesn't pass lang) or 'hi'.

    Unrecognized lang values fall back to the English cleaner rather than
    raising, so a bad/missing language-detection result degrades to the
    old behavior instead of crashing the request.
    """
    if not isinstance(text, str):
        return ''
    cleaner = _CLEANERS.get(lang, _clean_english)
    return cleaner(text)


def detect_language(text):
    """
    Lightweight language detection ahead of model routing.

    Uses fasttext's lid.176 model (recommended over langdetect for
    robustness on short/code-mixed text — see project scoping notes).
    Falls back to a cheap Unicode-block heuristic if fasttext/the model
    file isn't available, so this never hard-fails the request path.

    Returns: 'en' | 'hi' (only languages this pipeline currently trains
    models for — anything else fasttext detects still collapses to the
    nearer of these two via the heuristic fallback, since there's no
    third model to route to yet).
    """
    try:
        import fasttext
        # Model file expected at backend/lid.176.bin — download from
        # https://fasttext.cc/docs/en/language-identification.html
        import os
        model_path = os.path.join(os.path.dirname(__file__), '..', 'lid.176.bin')
        model = getattr(detect_language, '_ft_model', None)
        if model is None:
            model = fasttext.load_model(model_path)
            detect_language._ft_model = model
        pred_label, _ = model.predict(text.replace('\n', ' '), k=1)
        code = pred_label[0].replace('__label__', '')
        return 'hi' if code == 'hi' else 'en'
    except Exception:
        # Heuristic fallback: if the text contains a meaningful fraction
        # of Devanagari characters, call it Hindi; else English.
        devanagari_count = len(_HI_CHAR_RE.findall(text))
        return 'hi' if devanagari_count > max(len(text) * 0.15, 5) else 'en'