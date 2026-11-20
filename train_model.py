
import pandas as pd
import numpy as np
import re
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("dataset", exist_ok=True)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("dataset/processed_emails_cleaned.csv")

df = df.dropna(subset=["Email Text", "Email Type"])

emails = df["Email Text"].astype(str).str.lower()

mapping = {
    "phishing email": 1,
    "safe email": 0
}

y = (
    df["Email Type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(mapping)
)

if y.isnull().any():
    raise ValueError("Unknown email labels found!")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    emails,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 3. TF-IDF FEATURE EXTRACTION
# ==========================================

print("\nExtracting text features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_text = vectorizer.fit_transform(X_train)
X_test_text = vectorizer.transform(X_test)


# ==========================================
# 4. ADDITIONAL FEATURES
# ==========================================

keywords = [
    "verify",
    "urgent",
    "password",
    "account",
    "login",
    "bank",
    "click here",
    "winner",
    "free",
    "suspended"
]


def extract_features(email_list):

    features = []

    for email in email_list:

        email = email.lower()

        url_count = len(
            re.findall(r"https?://\S+|www\.\S+", email)
        )

        has_ip = int(
            bool(
                re.search(
                    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                    email
                )
            )
        )

        keyword_count = sum(
            email.count(word) for word in keywords
        )

        features.append([
            url_count,
            has_ip,
            keyword_count
        ])

    return csr_matrix(features)


X_train_extra = extract_features(X_train)
X_test_extra = extract_features(X_test)


# ==========================================
# 5. COMBINE FEATURES
# ==========================================

X_train_final = hstack([
    X_train_text,
    X_train_extra
])

X_test_final = hstack([
    X_test_text,
    X_test_extra
])


# ==========================================
# 6. INITIALIZE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB()

}


results = {}

print("\n========== MODEL TRAINING ==========")


# ==========================================
# 7. TRAIN AND EVALUATE MODELS
# ==========================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_final, y_train)

    predictions = model.predict(X_test_final)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    results[name] = {

        "model": model,
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1

    }

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Safe", "Phishing"],
            zero_division=0
        )
    )


# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

best_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_name]["model"]

best_predictions = results[best_name]["predictions"]

print("\n========== BEST MODEL ==========")

print("Selected Model:", best_name)


# ==========================================
# 9. CONFUSION MATRIX VALUES
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n========== CONFUSION MATRIX ==========")

print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Safe", "Phishing"],
    output_dict=True,
    zero_division=0
)


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

model_comparison = {}

for name, result in results.items():

    model_comparison[name] = {

        "accuracy": float(result["accuracy"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1_score": float(result["f1"])

    }


# ==========================================
# 12. SAVE PERFORMANCE RESULTS
# ==========================================

performance = {

    "model_name": best_name,

    "accuracy": float(
        results[best_name]["accuracy"]
    ),

    "precision": float(
        results[best_name]["precision"]
    ),

    "recall": float(
        results[best_name]["recall"]
    ),

    "f1_score": float(
        results[best_name]["f1"]
    ),

    "confusion_matrix": {

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)

    },

    "classification_report": report,

    "model_comparison": model_comparison

}


with open("models/performance.json", "w") as file:

    json.dump(
        performance,
        file,
        indent=4
    )


print("\nPerformance Results Saved Successfully!")


# ==========================================
# 13. SAVE CONFUSION MATRIX IMAGE
# ==========================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("dataset/confusion_matrix.png")

plt.close()

print("Confusion Matrix Image Saved Successfully!")


# ==========================================
# 14. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    {
        "model": best_model,
        "vectorizer": vectorizer,
        "keywords": keywords,
        "model_name": best_name
    },
    "models/phishing_model.pkl"
)

print("\nModel Saved Successfully!")


# ==========================================
# 15. FINAL OUTPUT
# ==========================================

print("\n========== GENERATED FILES ==========")

print("1. models/phishing_model.pkl")
print("2. models/performance.json")
print("3. dataset/confusion_matrix.png")

print("\n========== TRAINING COMPLETED ==========")
