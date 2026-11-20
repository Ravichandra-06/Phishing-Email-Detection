import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/processed_emails_cleaned.csv")

df["Email Text"] = (
    df["Email Text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# Reproduce the original split
X_train, X_test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Email Type"]
)

def normalize(text):
    return " ".join(str(text).split())

def fingerprint(text):
    return hashlib.md5(
        normalize(text).encode("utf-8")
    ).hexdigest()

train_hashes = set(
    X_train["Email Text"].apply(fingerprint)
)

test_hashes = X_test["Email Text"].apply(fingerprint)

overlap = test_hashes.isin(train_hashes).sum()

print("\n========== DATA LEAKAGE CHECK ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Exact overlaps:", overlap)

print("\nLeakage check completed.")