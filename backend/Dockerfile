FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn/pandas/scipy wheels + nltk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Bake NLTK corpora into the image so no runtime internet dependency
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords punkt punkt_tab
ENV NLTK_DATA=/usr/share/nltk_data

# Copy source + training data (True.csv/Fake.csv must exist locally before build)
COPY backend/ backend/
COPY main.py main.py

# Train fake-news RF model + AI-text detector at build time.
# NOTE: this step downloads the HC3 dataset from Hugging Face Hub,
# so the build requires internet access and will take longer than
# the fake-news-only build (dataset download + char n-gram vectorization
# on ~30k samples).
RUN python main.py

EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "backend.src.app:app"]