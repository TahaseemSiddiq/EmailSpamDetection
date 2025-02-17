# preprocessing_model.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

english_words = set(words.words())
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Custom stopwords
custom_stopwords = set(['aa', 'aaa', 'ab', 'ac'])

# Preprocessing class for text
class TextPreprocessor:
    def __init__(self):
        pass

    def process_text(self, text):
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove special characters and punctuation
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Convert text to lowercase
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords, single characters, numbers, and custom meaningless words
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1 and not word.isdigit() and word not in custom_stopwords]

        # Remove non-English words
        valid_tokens = [word for word in tokens if word in english_words]

        # Apply stemming and lemmatization
        stemmed_tokens = [ps.stem(word) for word in valid_tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

        # Join the tokens back into a cleaned string
        cleaned_text = " ".join(lemmatized_tokens)

        return cleaned_text

