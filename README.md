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