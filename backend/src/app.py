"""
app.py  —  Flask backend for Fake News Detector
Endpoints:
  POST /predict        { "url": "...", "lang": "en" | "hi" (optional) }
  POST /predict-text   { "text": "...", "title": "...", "lang": "en" | "hi" (optional) }
  GET  /health          -> status
"""

import os, re, pickle, requests

from flask       import Flask, request, jsonify
from flask_cors  import CORS
from bs4         import BeautifulSoup
from urllib.parse import urlparse
import json
from .safe_fetch import fetch_public

from .preprocess import clean_text, detect_language
# ── near the top, alongside VECTORIZER / MODEL loading ──
import numpy as np
from scipy.sparse import hstack, csr_matrix
from .train_ai_detector import stylometric_features  # reuse feature logic

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024
origins = [origin.strip() for origin in os.getenv('CORS_ORIGINS', '').split(',') if origin.strip()]
if origins:
    CORS(app, origins=origins)

@app.before_request
def validate_payload():
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error='Provide a JSON object.'), 400
        for key in ('url', 'text', 'title', 'lang'):
            if key in data and not isinstance(data[key], str):
                return jsonify(error=f'{key} must be a string.'), 400
        if data.get('lang') and data['lang'].strip().lower() not in ('en', 'hi'):
            return jsonify(error='Supported languages: en, hi.'), 400

@app.errorhandler(413)
def payload_too_large(error):
    return jsonify(error='Article exceeds the 128 KB request limit.'), 413

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, '..')

# Languages the fake-news classifier has (or may have) trained artifacts
# for. Add to this list as new languages get trained — nothing else in
# this file needs to change to pick them up, since loading is name-based.
SUPPORTED_LANGS = ['en', 'hi']


def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


AI_MODEL    = load_pickle('ai_detector_model.pkl')
AI_WORD_VEC = load_pickle('ai_detector_word_vec.pkl')
AI_CHAR_VEC = load_pickle('ai_detector_char_vec.pkl')
print(f"AI detector: {'loaded' if AI_MODEL else 'NOT FOUND - run main.py first'}")


def run_ai_detection(text, lang):
    """
    Returns None if the AI detector isn't available, OR if `lang` isn't
    English — the AI detector is trained exclusively on HC3 (English
    human vs. ChatGPT answers), so there's no meaningful result to
    return for other languages. Callers treat None as "omit AI fields
    from the response entirely" (see run_prediction()'s `if ai_result:`
    check) rather than surfacing a null/placeholder result.
    """
    if lang != 'en':
        return None
    if not (AI_MODEL and AI_WORD_VEC and AI_CHAR_VEC):
        return None
    X_word  = AI_WORD_VEC.transform([text])
    X_char  = AI_CHAR_VEC.transform([text])
    X_style = csr_matrix(stylometric_features([text]))
    X = hstack([X_word, X_char, X_style]).tocsr()

    label   = AI_MODEL.predict(X)[0]
    proba   = AI_MODEL.predict_proba(X)[0]
    classes = list(AI_MODEL.classes_)
    ai_prob = float(proba[classes.index('AI')]) * 100
    return {
        'ai_label':      label,                       # "AI" or "HUMAN"
        'ai_probability': round(ai_prob, 1),
        'ai_confidence': round(float(max(proba)) * 100, 1),
    }


def _artifact_names(lang):
    if lang == 'en':
        return 'model.pkl', 'vectorizer.pkl'
    return f'model_{lang}.pkl', f'vectorizer_{lang}.pkl'


def load_fake_news_models():
    """
    Load every available per-language (model, vectorizer) pair into a
    dict keyed by lang code, e.g. {'en': (model, vec), 'hi': (model, vec)}.
    A language with missing files is silently omitted (not an error) —
    checked at request time via lang not in MODELS.
    """
    models = {}
    for lang in SUPPORTED_LANGS:
        model_name, vec_name = _artifact_names(lang)
        model = load_pickle(model_name)
        vec   = load_pickle(vec_name)
        if model is not None and vec is not None:
            models[lang] = (model, vec)
    return models


MODELS = load_fake_news_models()
for lang in SUPPORTED_LANGS:
    status = 'loaded' if lang in MODELS else 'NOT FOUND'
    print(f"Fake-news model [{lang}]: {status}")

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# ── Platform detection ───────────────────────────────────────────
SOCIAL_PLATFORMS = {
    'x.com': 'x', 'twitter.com': 'x',
    'facebook.com': 'facebook', 'fb.watch': 'facebook',
    'instagram.com': 'instagram',
}

def detect_platform(netloc):
    host = netloc.replace('www.', '').replace('m.', '')
    for domain, name in SOCIAL_PLATFORMS.items():
        if host == domain or host.endswith('.' + domain):
            return name
    return None

def fetch_x_via_oembed(url):
    """X/Twitter's public oEmbed endpoint returns post HTML without auth,
    for PUBLIC posts only. This is the only reliable no-auth path for X."""
    try:
        from urllib.parse import urlencode
        endpoint = 'https://publish.twitter.com/oembed?' + urlencode({'url': url, 'omit_script': 'true'})
        data = json.loads(fetch_public(endpoint, HEADERS))
        soup = BeautifulSoup(data.get('html', ''), 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return {
            'title': data.get('author_name', 'X post'),
            'body_text': text,
            'preview': text[:500] + ('...' if len(text) > 500 else ''),
            'domain': 'x.com',
            'word_count': len(text.split()),
        }
    except Exception:
        raise ConnectionError(
            "Couldn't retrieve that X/Twitter post. It may be private, "
            "deleted, or the post has too little text — try pasting the "
            "text directly instead."
        )

def fetch_article(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("URL must start with http:// or https://")

    platform = detect_platform(parsed.netloc)

    if platform == 'x':
        return fetch_x_via_oembed(url)

    if platform in ('facebook', 'instagram'):
        raise ConnectionError(
            f"{platform.capitalize()} posts require you to be logged in to view, "
            "so this app can't scrape them directly (no public API access without "
            "credentials). Please switch to 'Paste text' mode and copy the post "
            "text in instead."
        )

    # ── Generic news-site scraping (unchanged) ──
    try:
        html = fetch_public(url, HEADERS)
    except ValueError:
        raise
    except Exception:
        raise ConnectionError('Could not retrieve the article. Try pasting its text.')

    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script','style','nav','footer','header','aside','form','noscript']):
        tag.decompose()

    title_tag  = soup.find('title') or soup.find('h1')
    title      = title_tag.get_text(strip=True) if title_tag else ''
    content_el = (soup.find('article') or soup.find('main') or
                  soup.find('div', class_=re.compile(r'article|content|post|story|body', re.I)) or
                  soup.body)
    body_text  = content_el.get_text(separator=' ', strip=True) if content_el else ''
    body_text  = re.sub(r'\s+', ' ', body_text).strip()
    return {
        'title': title, 'body_text': body_text,
        'preview': body_text[:500] + ('...' if len(body_text) > 500 else ''),
        'domain': parsed.netloc.replace('www.', ''),
        'word_count': len(body_text.split()),
    }

def run_prediction(text, title, lang=None):
    """
    lang: 'en' | 'hi' | None. If None, auto-detected from `text` via
    detect_language(). An explicit lang from the request always wins
    over auto-detection — the caller (e.g. a user-set UI toggle) knows
    better than a statistical guess on possibly short/noisy input.
    """
    if lang is None:
        lang = detect_language(text)

    if lang not in MODELS:
        available = ', '.join(sorted(MODELS)) or 'none'
        raise RuntimeError(
            f"No trained model available for lang='{lang}'. "
            f"Available: {available}. Run main.py to train it first."
        )
    model, vectorizer = MODELS[lang]

    combined  = title + ' ' + title + ' ' + text
    cleaned   = clean_text(combined, lang=lang)
    features  = vectorizer.transform([cleaned])
    label     = model.predict(features)[0]
    proba     = model.predict_proba(features)[0]
    classes   = list(model.classes_)
    fake_prob = float(proba[classes.index('FAKE')]) * 100
    real_prob = float(proba[classes.index('REAL')]) * 100

    result = {
        'label':      label,
        'confidence': round(float(max(proba)) * 100, 1),
        'fake_prob':  round(fake_prob, 1),
        'real_prob':  round(real_prob, 1),
        'model_used': 'Random Forest',
        'lang':       lang,
    }

    # AI-text detection runs on the RAW (uncleaned) text — style signal
    # depends on stopwords/punctuation that clean_text() strips out.
    # Gated to English only inside run_ai_detection() — see its docstring.
    ai_result = run_ai_detection(text, lang)
    if ai_result:
        result.update(ai_result)

    return result

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    url  = (data.get('url') or '').strip()
    lang = (data.get('lang') or '').strip().lower() or None
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        article = fetch_article(url)
    except (ValueError, ConnectionError) as e:
        return jsonify({'error': str(e)}), 422
    if article['word_count'] < 20:
        return jsonify({'error': 'Not enough text extracted. The site may block scraping — try pasting the text instead.'}), 422
    try:
        pred = run_prediction(article['body_text'], article['title'], lang=lang)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({**pred, 'title': article['title'], 'preview': article['preview'],
                    'domain': article['domain'], 'word_count': article['word_count'],
                    'platform': detect_platform(urlparse(url).netloc) or 'web'})

@app.route('/predict-text', methods=['POST'])
def predict_text():
    data  = request.get_json(silent=True) or {}
    text  = (data.get('text') or '').strip()
    title = (data.get('title') or '').strip()
    lang  = (data.get('lang') or '').strip().lower() or None
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text.split()) < 10:
        return jsonify({'error': 'Please provide at least 10 words of article text.'}), 400
    try:
        pred = run_prediction(text, title, lang=lang)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    preview = text[:500] + ('...' if len(text) > 500 else '')
    return jsonify({**pred, 'title': title or 'Pasted text', 'preview': preview,
                    'domain': 'manual input', 'word_count': len(text.split()),
                    'platform': 'manual'})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if 'en' in MODELS else 'unavailable',
        'fake_news_models': {lang: True for lang in MODELS},
        'fake_news_models_missing': [l for l in SUPPORTED_LANGS if l not in MODELS],
        'ai_detector': AI_MODEL is not None,
        'ai_detector_note': 'English-only (trained on HC3)',
    }), (200 if 'en' in MODELS else 503)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
