
import pandas as pd
import re

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load cleaned dataset
df = pd.read_csv("dataset/processed_emails_cleaned.csv")

df = df.dropna(subset=["Email Text", "Email Type"])

X = df["Email Text"].astype(str).str.lower()
y = df["Email Type"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

# Same split for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def evaluate(name, train_features, test_features):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(train_features, y_train)

    predictions = model.predict(test_features)

    print(f"\n========== {name} ==========")

    print("Accuracy:", round(
        accuracy_score(y_test, predictions) * 100, 2
    ))

    print("Precision:", round(
        precision_score(y_test, predictions) * 100, 2
    ))

    print("Recall:", round(
        recall_score(y_test, predictions) * 100, 2
    ))

    print("F1 Score:", round(
        f1_score(y_test, predictions) * 100, 2
    ))

    return model


# Model 1: Word-level TF-IDF
word_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

word_model = evaluate(
    "WORD LEVEL TF-IDF",
    X_train_word,
    X_test_word
)


# Model 2: Character-level TF-IDF
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=10000,
    min_df=2
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

char_model = evaluate(
    "CHARACTER LEVEL TF-IDF",
    X_train_char,
    X_test_char
)


# Model 3: Combined word + character features
X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

combined_model = evaluate(
    "COMBINED TF-IDF",
    X_train_combined,
    X_test_combined
)

print("\n========== COMPARISON COMPLETED ==========")
