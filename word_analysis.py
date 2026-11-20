import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("dataset/processed_emails_cleaned.csv")

for label in ["Safe Email", "Phishing Email"]:

    emails = df[df["Email Type"] == label]["Email Text"].fillna("")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=30,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )

    matrix = vectorizer.fit_transform(emails)

    frequencies = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    results = pd.DataFrame({
        "Word": words,
        "Frequency": frequencies
    }).sort_values(
        by="Frequency",
        ascending=False
    )

    print("\n==========", label, "==========")
    print(results.head(20).to_string(index=False))