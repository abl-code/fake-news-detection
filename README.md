# Fake News Detection

A full-stack machine learning system for detecting fake news and AI-generated
text. Two independent classifiers — a fake-news detector and an AI-text
detector — run in parallel behind a Flask API, with a React frontend and a
Docker Compose deployment.

This project has been through two rounds of data-leakage debugging (documented
below), and the point of documenting them isn't to hide a rough draft — it's
that catching a 99%+ "too good to be true" accuracy score, tracing it back to
a specific leakage source, and fixing it is the actual engineering work.
Anyone can report a high accuracy number; showing the audit trail behind it is
the more useful signal.

## Architecture

```
fake-news-detection/
├── backend/
│   ├── src/
│   │   ├── preprocess.py         # Text cleaning pipeline
│   │   ├── train.py              # Fake-news model training (ISOT)
│   │   ├── train_ai_detector.py  # AI-text detector training (HC3)
│   │   ├── predict.py            # CLI prediction helper
│   │   ├── app.py                # Flask API
│   │   └── check_leakage.py      # Dataset leakage audit script
│   ├── data/                     # Dataset CSVs (not included in repo)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # React UI
│   │   └── main.jsx
│   └── Dockerfile
├── test_predictions.py           # Regression suite (paragraph-length cases)
├── docker-compose.yml
└── main.py                        # Local training entry point
```

## Stack

- **Backend:** Python, Flask, scikit-learn, NLTK
- **Frontend:** React, Vite
- **Deployment:** Docker, Docker Compose, nginx
- **ML:** TF-IDF (word-level + char n-gram) → Random Forest, plus hand-crafted
  stylometric features for the AI-text detector

## How It Works

Two independent classifiers run on any submitted text (via URL scrape or
pasted text):

1. **Fake-news detector** — TF-IDF (unigrams + bigrams) → Random Forest,
   trained on the ISOT Fake News dataset (title × 2 + article body).
2. **AI-generated-text detector** — TF-IDF (word-level + char 3–5-grams) +
   stylometric features → Random Forest, trained on HC3
   (`Hello-SimpleAI/HC3`, human vs. ChatGPT answers).

Both return a label and confidence score; the AI-detector is explicitly
surfaced in the UI as a lower-confidence secondary signal (see
[Limitations](#known-limitations) below).

## Installation

```bash
git clone https://github.com/abl-code/fake-news-detection.git
cd fake-news-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r backend/requirements.txt
```

### Dataset setup

Place these files in `backend/data/`:

- [ISOT Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) → `Fake.csv`, `True.csv`
- HC3 is pulled automatically via `datasets.load_dataset("Hello-SimpleAI/HC3", "default", revision="refs/convert/parquet")` — no manual download needed.

### Train locally

```bash
python main.py
```

### Run with Docker

```bash
docker compose up --build
```

- Frontend: `http://localhost:8080`
- Backend API: `http://localhost:5000`

Model training runs at image build time (`RUN python main.py` in
`backend/Dockerfile`), so `True.csv`/`Fake.csv` must exist locally before
`docker compose up --build`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | `{ "url": "..." }` — scrapes and classifies a URL |
| `/predict-text` | POST | `{ "text": "...", "title": "..." }` — classifies pasted text |
| `/health` | GET | Model/vectorizer load status |

## Regression Testing

`test_predictions.py` runs a fixed set of paragraph-length test cases against
the trained model — real factual news, obvious fake news, *subtle* fake news
(calm tone, no sensational vocabulary), and ambiguous cases (satire, opinion,
poorly-sourced-but-real reporting) with no ground truth label.

```bash
python test_predictions.py    # prints a markdown results table
pytest test_predictions.py    # soft assertions, CI-friendly
```

Paragraph-length inputs are used deliberately: the model is trained on
`title × 2 + full article body`, so bare headlines produce out-of-distribution
TF-IDF vectors and are not a reliable way to evaluate it.

## Known Limitations

These are documented rather than engineered away, because silently tuning a
model until every test case passes produces a worse, less honest picture of
what it actually does.

### The classifier tracks rhetorical register, not claim plausibility

A regression run against 9 labeled paragraph-length test cases scored
**7/9 (77.8%)**, with two informative misses:

| Case | Expected | Predicted | Confidence | What it shows |
|---|---|---|---|---|
| Calm-toned conspiratorial claim, hedged sourcing ("a retired intelligence official says...") | FAKE | REAL | 91.0% | A false claim delivered in measured, well-sourced-sounding prose was misclassified with high confidence. |
| Hedged, cautious academic reporting ("the authors cautioned... called for larger trials") | REAL | FAKE | 57.5% | Careful, uncertainty-acknowledging language — standard in legitimate science journalism — pushed the model toward FAKE. |

Three additional cases with no objective ground truth (satire, an opinion
column, and a real-but-thinly-sourced news item) all classified as FAKE, two
at high confidence (73.5%, 90.0%). This is consistent with the same pattern:
persuasive or rhetorically loaded language resembles the FAKE class's style
even when the underlying content is legitimate journalism.

**Conclusion:** this is a TF-IDF + Random Forest text classifier, and like
essentially all classifiers of this type, it is substantially a *style*
detector — sensationalized tone, hedging patterns, and lexical choices — not
a fact-checker. It should be read as a signal about how a piece of text is
written, not a verdict on whether its claims are true. This is stated
explicitly here rather than left for a reviewer to discover independently.

### Domain specificity (ISOT)

The fake-news training data (ISOT) is almost entirely U.S. political news
from 2016–2017. Accuracy on non-U.S. or non-political topics has not been
independently validated at scale — a single non-U.S. central-bank test case
classified correctly, but this is not sufficient evidence of general
cross-domain reliability and should not be assumed.

### AI-text detector reliability (HC3)

- HC3's `human_answers` are drawn from a different corpus distribution
  (length, variance) than `chatgpt_answers` — see the leakage note below.
  Even after fixing the leakage bug this introduces, the underlying
  length/style disparity between the two classes is a property of the
  dataset's collection methodology, not something that can be fully removed
  through preprocessing.
- The UI surfaces this detector with an explicit caveat
  ("AI-text detectors are known to be unreliable, especially on edited or
  non-native English writing — treat this as a signal, not a verdict") rather
  than presenting it as authoritative.

### Two data-leakage bugs were found and fixed during development

**1. ISOT Reuters-tag leakage.** 99.2% of REAL articles in the ISOT dataset
carried a Reuters wire-service tag entirely absent from FAKE articles — the
model was substantially learning to detect that tag, not detect fake news.
Fixed with a `_strip_reuters_tag()` preprocessing step in `train.py`. This
dropped held-out accuracy from a suspicious ~99.8% into a more defensible
85–93% range, which is the expected direction and magnitude for a genuine
fix.

**2. HC3 tokenization-artifact leakage.** The AI-text detector originally
reported 100.00% held-out accuracy — implausible for this task even for
published detectors. Investigation found `human_answers` contained systematic
detokenization artifacts (`"word ."` instead of `"word."`, `"wo n't"` instead
of `"won't"`) entirely absent from `chatgpt_answers`, which the char n-gram
features were trivially learning as a near-perfect proxy for "human." Fixed
with a `_normalize_spacing()` step applied uniformly to both classes before
feature extraction.

Both leakage sources were metadata/formatting artifacts of how the datasets
were assembled, not signal the model was supposed to learn — which is part
of why validating a suspiciously high accuracy score before trusting it
matters more than the score itself.

## Tech Stack

- **Python** — backend, ML pipeline
- **scikit-learn** — TF-IDF vectorization, Random Forest classification
- **NLTK** — tokenization, stopword removal
- **Flask** + **flask-cors** — API
- **BeautifulSoup** — URL scraping
- **React** + **Vite** — frontend
- **Docker** / **docker-compose** / **nginx** — deployment

## Author

**Abiel Varghese**
[github.com/abl-code](https://github.com/abl-code)