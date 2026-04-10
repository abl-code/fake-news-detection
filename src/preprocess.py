import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw news text through 5 steps:
    1. Lowercase
    2. Remove punctuation and special characters
    3. Tokenize into individual words
    4. Remove stop words (the, is, at, on...)
    5. Rejoin into a clean string
    """
    if not isinstance(text, str):
        return ''
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)