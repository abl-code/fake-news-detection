"""
app.py  —  Flask backend for Fake News Detector
Endpoints:
  POST /predict        { "url": "...", "model": "Logistic Regression" }
  POST /predict-text   { "text": "...", "model": "Naive Bayes" }
  GET  /models         -> list of available models
  GET  /health         -> status
"""

import os, sys, re, pickle, nltk, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask       import Flask, request, jsonify
from flask_cors  import CORS
from bs4         import BeautifulSoup
from urllib.parse import urlparse

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)

STOP_WORDS = set(stopwords.words('english'))
BASE       = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE, '..')

MODEL_FILES = {
    "Naive Bayes":         "model_naive_bayes",
    "Logistic Regression": "model_logistic_regression",
    "Linear SVM":          "model_linear_svm",
    "Random Forest":       "model_random_forest",
    "Gradient Boosting":   "model_gradient_boosting",
}
DEFAULT_MODEL = "Logistic Regression"
_model_cache  = {}

def load_vectorizer():
    path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model(name):
    if name in _model_cache:
        return _model_cache[name]
    filename = MODEL_FILES.get(name)
    if filename:
        path = os.path.join(MODEL_DIR, f'{filename}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            _model_cache[name] = model
            return model
    fallback = os.path.join(MODEL_DIR, 'model.pkl')
    if os.path.exists(fallback):
        with open(fallback, 'rb') as f:
            model = pickle.load(f)
        _model_cache[name] = model
        return model
    return None

def available_models():
    found = []
    for name, filename in MODEL_FILES.items():
        if os.path.exists(os.path.join(MODEL_DIR, f'{filename}.pkl')):
            found.append(name)
    if not found and os.path.exists(os.path.join(MODEL_DIR, 'model.pkl')):
        found = list(MODEL_FILES.keys())
    return found

VECTORIZER = load_vectorizer()
print(f"Vectorizer: {'loaded' if VECTORIZER else 'NOT FOUND - run main.py first'}")
print(f"Models available: {available_models()}")

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
}

def fetch_article(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("URL must start with http:// or https://")
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

def run_prediction(text, title, model_name):
    if VECTORIZER is None:
        raise RuntimeError("Vectorizer not loaded. Run main.py first.")
    model = load_model(model_name)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' not found. Run main.py first.")
    combined  = title + ' ' + title + ' ' + text
    cleaned   = clean_text(combined)
    features  = VECTORIZER.transform([cleaned])
    label     = model.predict(features)[0]
    proba     = model.predict_proba(features)[0]
    classes   = list(model.classes_)
    fake_prob = float(proba[classes.index('FAKE')]) * 100
    real_prob = float(proba[classes.index('REAL')]) * 100
    return {
        'label':      label,
        'confidence': round(float(max(proba)) * 100, 1),
        'fake_prob':  round(fake_prob, 1),
        'real_prob':  round(real_prob, 1),
        'model_used': model_name,
    }

@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json(silent=True) or {}
    url        = (data.get('url') or '').strip()
    model_name = data.get('model', DEFAULT_MODEL)
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        article = fetch_article(url)
    except (ValueError, ConnectionError) as e:
        return jsonify({'error': str(e)}), 422
    if article['word_count'] < 20:
        return jsonify({'error': 'Not enough text extracted. The site may block scraping.'}), 422
    try:
        pred = run_prediction(article['body_text'], article['title'], model_name)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({**pred, 'title': article['title'], 'preview': article['preview'],
                    'domain': article['domain'], 'word_count': article['word_count']})

@app.route('/predict-text', methods=['POST'])
def predict_text():
    data       = request.get_json(silent=True) or {}
    text       = (data.get('text') or '').strip()
    title      = (data.get('title') or '').strip()
    model_name = data.get('model', DEFAULT_MODEL)
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text.split()) < 10:
        return jsonify({'error': 'Please provide at least 10 words of article text.'}), 400
    try:
        pred = run_prediction(text, title, model_name)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    preview = text[:500] + ('...' if len(text) > 500 else '')
    return jsonify({**pred, 'title': title or 'Pasted text', 'preview': preview,
                    'domain': 'manual input', 'word_count': len(text.split())})

@app.route('/models')
def get_models():
    models = available_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
    return jsonify({'models': models, 'default': default})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'vectorizer': VECTORIZER is not None,
                    'models': available_models()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)