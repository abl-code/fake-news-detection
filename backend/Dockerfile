FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn/pandas wheels + nltk
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

# Train all 5 models at build time -> produces backend/*.pkl
RUN python main.py

EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "backend.src.app:app"]