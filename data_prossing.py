import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
import nltk

# Download stopwords (first time only)
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load stop words
stop_words = set(stopwords.words("english"))

# Keep important modal verbs (domain knowledge)
IMPORTANT_WORDS = {"should", "must", "never", "need"}
stop_words = stop_words - IMPORTANT_WORDS


def clean_text(text):
    """
    Perform text cleaning:
    - Lowercasing
    - Remove URLs
    - Remove HTML tags
    - Remove special characters
    - Remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"<.*?>", "", text)               # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)           # keep letters only
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text


def preprocess_text(text):
    """
    Full preprocessing pipeline:
    - Cleaning
    - Tokenization
    - Stop word removal
    - Lemmatization
    """
    text = clean_text(text)
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 2
    ]

    return " ".join(tokens)


# -------- Load dataset --------
df = pd.read_csv("guardian_israel_palestine_balanced_2000.csv")

# -------- Combine title + text --------
df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

# -------- Apply preprocessing --------
df["processed_text"] = df["full_text"].apply(preprocess_text)

# -------- Keep only necessary columns --------
processed_df = df[["processed_text", "label"]]

# -------- Save cleaned dataset --------
processed_df.to_csv(
    "guardian_palestine_preprocessed.csv",
    index=False,
    encoding="utf-8"
)

print("Preprocessing completed.")
print("Saved to guardian_palestine_preprocessed.csv")
