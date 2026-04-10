# Fake News Detection Using Naive Bayes Classification

A machine learning project that detects fake news articles using Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier. Trained on real Indian news from Times of India and Indian Express paired with known fake news articles.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 94.63% |
| Precision (FAKE) | 0.91 |
| Precision (REAL) | 0.99 |
| Recall (FAKE) | 0.99 |
| Recall (REAL) | 0.90 |
| F1-Score (FAKE) | 0.95 |
| F1-Score (REAL) | 0.94 |

## Dataset

- **Real news:** Indian news articles from Times of India, Indian Express and other major outlets (~102,000 articles)
- **Fake news:** Kaggle ISOT Fake News Dataset (~23,481 articles)
- **Final training set:** 46,962 balanced articles (23,481 real + 23,481 fake)

## Project Structure
## Project Structure
fake-news-detection/
├── data/               # Dataset CSVs (not included in repo)
├── src/
│   ├── preprocess.py   # Text cleaning pipeline
│   ├── train.py        # Model training and evaluation
│   ├── predict.py      # Prediction function
│   └── init.py
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md

## How It Works

1. **Preprocessing** — raw text is lowercased, punctuation removed, tokenized and stop words stripped
2. **TF-IDF Vectorization** — text converted to numerical feature vectors (5,000 features, unigrams + bigrams)
3. **Naive Bayes** — Multinomial Naive Bayes classifier trained on the TF-IDF matrix
4. **Prediction** — any headline or article can be classified as REAL or FAKE with a confidence score

## Installation

```bash
# Clone the repository
git clone https://github.com/abl-code/fake-news-detection.git
cd fake-news-detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

After training completes the program enters interactive mode where you can type any headline and get a prediction:
Enter headline: PM Modi inaugurates new metro line in Mumbai
✅  REAL  (82.3% confidence)
Enter headline: SHOCKING: Secret government plot to control population EXPOSED
🚨  FAKE  (94.1% confidence)

## Dataset Setup

Download the datasets from Kaggle and place them in the `data/` folder:

- [Indian News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) → `news_summary.csv`, `news_summary_more.csv`
- [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) → `Fake.csv`, `True.csv`

## Tech Stack

- **Python 3.13**
- **pandas** — data loading and manipulation
- **scikit-learn** — TF-IDF vectorization and Naive Bayes classifier
- **nltk** — tokenization and stop word removal
- **matplotlib / seaborn** — visualization

## Algorithm

Naive Bayes classifies articles using Bayes' Theorem:
P(FAKE | article) = P(article | FAKE) × P(FAKE) / P(article)

Each word's contribution is computed independently (the "naive" assumption). TF-IDF weights ensure distinctive words like "BREAKING", "EXPOSED", "SHOCKING" are amplified while common words are suppressed.

## Author

**Abiel Varghese**  
3rd Year B.Tech Information Technology  
Data Mining Mini Project