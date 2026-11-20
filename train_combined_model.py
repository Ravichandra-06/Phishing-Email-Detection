
import pandas as pd
import joblib
import os

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset/processed_emails_cleaned.csv")
df = df.dropna(subset=["Email Text", "Email Type"])

X = df["Email Text"].astype(str).str.lower()
y = df["Email Type"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Word TF-IDF
word_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

# Character TF-IDF
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=10000,
    min_df=2
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

# Combine features
X_train_combined = hstack([
    X_train_word,
    X_train_char
]).tocsr()

# Train model
model = LogisticRegression(
    solver="liblinear",
    max_iter=300,
    class_weight="balanced",
    C=1.0,
    random_state=42
)

model.fit(X_train_combined, y_train)

# Save separate candidate
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": model,
    "word_vectorizer": word_vectorizer,
    "char_vectorizer": char_vectorizer,
    "model_name": "Combined Word + Character TF-IDF"
}, "models/combined_phishing_model.pkl")

print("\nCombined model saved successfully!")
print("File: models/combined_phishing_model.pkl")
