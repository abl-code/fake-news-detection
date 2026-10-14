# Fake News Detection

A full-stack demo that classifies news article text as **REAL** or **FAKE** and optionally estimates whether English text is AI-written. Built with React, Flask, TF-IDF, and Random Forest classifiers.

**These predictions are experimental signals, not verified fact-checking.** Model confidence is a class probability estimate, not the probability that an article is true.

## Project status

- Local Docker deployment verified on **14 October 2026**.
- English news classification and optional English AI-text detection are available in the current local artifacts.
- Hindi preprocessing and model routing are implemented; trained Hindi artifacts are not included in this checkout.
- Public hosting, a public demo link, and CI/CD are not configured yet.
- Source code and trained artifacts have different distribution paths: models and datasets are excluded from Git.

## Features

- Analyze an article URL or paste article text and an optional title.
- Return REAL/FAKE probabilities, confidence, an article preview, and word count.
- Automatically route English and Hindi text to an available language model.
- Show an AI-writing estimate when all English AI-detector artifacts are available.
- Block private-network URL destinations, validate redirects, and limit downloaded responses.
- Run the frontend and API together with Docker Compose.

URL extraction depends on the source website. Paywalls, JavaScript-rendered pages, and social-media login requirements can prevent extraction; paste the article text instead.

## Architecture

```text
Browser -> nginx frontend -> /api -> Flask / Gunicorn -> trained models
             port 8080               internal port 5000
```

The browser uses `/api` on the same origin by default. nginx forwards those requests to the internal backend; port 5000 is not published by Compose.

```text
backend/
  src/app.py                 Flask routes and prediction pipeline
  src/preprocess.py          English/Hindi cleaning and language detection
  src/train.py               News dataset loading and model training
  src/train_ai_detector.py   HC3 AI-text training and style features
  src/safe_fetch.py          Public-IP validation and bounded URL fetching
  src/check_leakage.py       Dataset source/formatting audit
  src/predict.py             Standalone prediction helper
  Dockerfile                 Backend runtime image
  *.pkl                      Local trained artifacts (gitignored)
  data/                      Local training datasets (gitignored)
frontend/
  src/App.jsx                React interface
  nginx.conf                 API proxy and request limits
  Dockerfile                 Frontend build and web server
main.py                      Training entry point
requirements.txt             Pinned direct Python dependencies
docker-compose.yml            Local deployment
tests/                        API and URL-fetch protection tests
test_predictions.py           Model prediction regression cases
```

## Quick start with Docker

### Requirements

- Docker Desktop running with Linux containers on Windows, or Docker Engine with the Compose plugin on Linux.
- Trusted English model artifacts in `backend/`.
- Internet access for the first image build.

A fresh clone does **not** contain model files. Copy trusted artifacts from your training environment or train them using the instructions below before building.

| Feature | Required files in `backend/` |
| --- | --- |
| English news classification | `model.pkl`, `vectorizer.pkl` |
| Hindi news classification (optional) | `model_hi.pkl`, `vectorizer_hi.pkl` |
| English AI-text detection (optional) | `ai_detector_model.pkl`, `ai_detector_word_vec.pkl`, `ai_detector_char_vec.pkl` |

Use the same scikit-learn version for training and serving. Never load untrusted pickle files; loading a pickle can execute code.

From the repository root:

```powershell
docker compose up -d --build
docker compose ps
```

Open [http://localhost:8080](http://localhost:8080). Check [http://localhost:8080/api/health](http://localhost:8080/api/health); a ready English model returns HTTP 200 with `status: "ok"`.

Useful commands:

```powershell
docker compose logs --tail=100
docker compose down
```

Docker builds use existing artifacts. They do not train models or download training data, and raw datasets are excluded from the image.

## Free temporary demo link

You can share the locally running app using a **Cloudflare Quick Tunnel**, without buying a server or domain.

1. Start the app with Docker Compose.
2. Install `cloudflared` using the [official download instructions](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
3. In another terminal, run:

```powershell
cloudflared tunnel --url http://localhost:8080
```

Share the HTTPS `trycloudflare.com` address printed in that terminal. Your computer, internet connection, Docker containers, and tunnel must stay running. The address is temporary, and anyone with the link can access the app. Stop the tunnel with Ctrl+C when the demo ends.

Quick Tunnels are for testing and development, not an always-on production service. See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/). No tunnel or public URL is currently configured in this project.

## Local development without Docker

Use Python **3.13** and Node.js **22**, matching the Docker build environments.

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app backend.src.app run --port 5000
```

On Linux/macOS, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

In another terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Use `npm` instead of `npm.cmd` on Linux/macOS. Open the URL printed by Vite. Its development proxy forwards `/api` to Flask on port 5000.

If `frontend/.env` already sets `VITE_API_URL`, remove that setting to use the development proxy, or configure backend `CORS_ORIGINS` with the exact frontend origin. For a separately hosted backend, set `VITE_API_URL` before building the frontend; Vite embeds it in the bundle.

## Training

Training is optional when compatible artifacts are already available. It can take significant time and memory.

1. Install the root Python requirements in your virtual environment.
2. Place the English ISOT files at `backend/data/True.csv` and `backend/data/Fake.csv`. See the [ISOT dataset listing](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
3. For optional Hindi training, place HFDND CSV files in `backend/data/HFDND/`, or an ISOT-format pair in `backend/data/hi/`. Check `backend/src/train.py` for the expected CSV schema.
4. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` trains available English/Hindi news models, skipping languages whose datasets are missing, then trains the English AI-text detector. HC3 is downloaded from Hugging Face during AI training, so that step needs internet access.

News classification uses double-weighted titles plus article bodies, a 10,000-feature unigram/bigram TF-IDF vectorizer, and a 200-tree Random Forest. AI detection uses word and character TF-IDF features plus stylometric features with a separate 250-tree Random Forest.

English/Hindi cleaning uses built-in stopword lists and script-aware regular expressions. Optional fastText identification can be installed separately with its language model; the default installation uses the Unicode heuristic fallback.

## API

These are backend routes. When using Docker Compose, prefix them with `/api` in browser requests.

| Method | Route | Request / behavior |
| --- | --- | --- |
| GET | `/health` | 200 when the English model is available; otherwise 503 |
| POST | `/predict` | `{ "url": "https://example.com/article", "lang": "en" }` |
| POST | `/predict-text` | `{ "text": "article text...", "title": "optional title", "lang": "en" }` |

`lang` is optional and accepts `en` or `hi`; when omitted, language is detected automatically. Pasted text needs at least 10 words; URL extraction needs at least 20 words.

Responses include `label`, `confidence`, `fake_prob`, `real_prob`, `model_used`, `lang`, `title`, `preview`, `domain`, `word_count`, and `platform`. English responses may also include `ai_label`, `ai_probability`, and `ai_confidence`.

Invalid inputs return 400, oversized requests 413, extraction failures 422, nginx rate limits 429, and missing language models 503.

## Tests and verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest test_predictions.py -q
docker compose config --quiet
```

Build the frontend from `frontend/` with `npm.cmd run build` (or `npm run build` on Linux/macOS).

| Check | Last verified result |
| --- | --- |
| Deployment/API tests, 7 October 2026 | 9 passed |
| Combined deployment and model suite, 7 October 2026 | 16 passed, 8 skipped, 2 failed |
| Frontend production build | Passed |
| Local Compose configuration | Passed |
| Docker images and local container startup, 14 October 2026 | Passed; backend healthy |
| English prediction through nginx, 14 October 2026 | Passed |
| Public hosting and external demo link | Not yet configured |

The two model failures remain visible: a real medical-study example is classified as FAKE, and a subtle fake weather-conspiracy example as REAL. Hindi cases skip when their artifacts are absent; unlabeled cases also skip. These small regression examples are not a broad accuracy benchmark.

## Known limitations

- The classifier learns wording, source, and style patterns; it does not verify claims against evidence. Calm misinformation can pass, and cautious factual reporting can be flagged.
- English training data is concentrated in historical US political news. Generalization to current events, other countries, and other topics has not been established.
- Short headlines differ from the full articles used in training. Prefer paragraph-length inputs.
- AI-writing detection can misclassify edited text and non-native English writing. It is English-only and should not be used as proof of authorship.
- Hindi model routing exists, but trained Hindi files are missing locally. The Hindi regression examples require independent language and label review.
- Some sites block scraping or require login. Use pasted text when extraction fails.
- Training includes Reuters-tag removal and HC3 spacing normalization to reduce known source/formatting leakage. Those measures do not establish general real-world reliability.

## Deployment safeguards

- Backend runs as a non-root user with one Gunicorn worker and two threads.
- Backend port stays internal to the Compose network; local frontend binds to loopback.
- Same-origin API requests do not require open CORS; separate origins must be explicitly configured.
- Requests are limited to 128 KB; downloaded responses to 2 MB.
- URL fetching validates public destination IPs, pins connections to validated addresses, and checks redirects.
- nginx limits API traffic to 10 requests/minute per direct peer with a burst of five. Behind a tunnel, this can become a shared limit; configure trusted client-IP handling before serving many users.

## Future work

- Publish a demo link or configure persistent hosting.
- Add CI/CD after the deployment method is settled.
- Train and independently validate Hindi artifacts.
- Evaluate broader datasets and improve the documented model failure cases.

## Author

**Abiel Varghese** — [GitHub](https://github.com/abl-code)
