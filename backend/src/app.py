"""
app.py  —  Flask backend for Fake News Detector
Endpoints:
  POST /predict        { "url": "..." }
  POST /predict-text   { "text": "...", "title": "..." }
  GET  /health          -> status
"""

import os, re, pickle, requests

from flask       import Flask, request, jsonify
from flask_cors  import CORS
from bs4         import BeautifulSoup
from urllib.parse import urlparse

from .preprocess import clean_text
# ── near the top, alongside VECTORIZER / MODEL loading ──
import numpy as np
from scipy.sparse import hstack, csr_matrix
from .train_ai_detector import stylometric_features  # reuse feature logic

app = Flask(__name__)
CORS(app)

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, '..')
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


def run_ai_detection(text):
    """Returns None if the AI detector isn't available (fails soft)."""
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
def load_vectorizer():
    path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model():
    path = os.path.join(MODEL_DIR, 'model.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

VECTORIZER = load_vectorizer()
MODEL      = load_model()
print(f"Vectorizer: {'loaded' if VECTORIZER else 'NOT FOUND - run main.py first'}")
print(f"Model (Random Forest): {'loaded' if MODEL else 'NOT FOUND - run main.py first'}")

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
        resp = requests.get(
            'https://publish.twitter.com/oembed',
            params={'url': url, 'omit_script': 'true'},
            headers=HEADERS, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
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
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ConnectionError("Request timed out")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Could not reach the URL")
    except requests.exceptions.HTTPError:
        raise ConnectionError(f"HTTP error {resp.status_code}")

    soup = BeautifulSoup(resp.text, 'html.parser')
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

def run_prediction(text, title):
    if VECTORIZER is None or MODEL is None:
        raise RuntimeError("Model not loaded. Run main.py first.")
    combined  = title + ' ' + title + ' ' + text
    cleaned   = clean_text(combined)
    features  = VECTORIZER.transform([cleaned])
    label     = MODEL.predict(features)[0]
    proba     = MODEL.predict_proba(features)[0]
    classes   = list(MODEL.classes_)
    fake_prob = float(proba[classes.index('FAKE')]) * 100
    real_prob = float(proba[classes.index('REAL')]) * 100

    result = {
        'label':      label,
        'confidence': round(float(max(proba)) * 100, 1),
        'fake_prob':  round(fake_prob, 1),
        'real_prob':  round(real_prob, 1),
        'model_used': 'Random Forest',
    }

    # AI-text detection runs on the RAW (uncleaned) text — style signal
    # depends on stopwords/punctuation that clean_text() strips out.
    ai_result = run_ai_detection(text)
    if ai_result:
        result.update(ai_result)

    return result

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    url  = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        article = fetch_article(url)
    except (ValueError, ConnectionError) as e:
        return jsonify({'error': str(e)}), 422
    if article['word_count'] < 20:
        return jsonify({'error': 'Not enough text extracted. The site may block scraping — try pasting the text instead.'}), 422
    try:
        pred = run_prediction(article['body_text'], article['title'])
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
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text.split()) < 10:
        return jsonify({'error': 'Please provide at least 10 words of article text.'}), 400
    try:
        pred = run_prediction(text, title)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    preview = text[:500] + ('...' if len(text) > 500 else '')
    return jsonify({**pred, 'title': title or 'Pasted text', 'preview': preview,
                    'domain': 'manual input', 'word_count': len(text.split()),
                    'platform': 'manual'})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vectorizer': VECTORIZER is not None,
        'model': MODEL is not None,
        'ai_detector': AI_MODEL is not None,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)