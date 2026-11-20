
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

# Load cleaned dataset
df = pd.read_csv("dataset/processed_emails.csv")

# Remove missing values
df = df.dropna(subset=["Email Text", "Email Type"])

# Prepare email text
emails = df["Email Text"].astype(str).str.lower()

# Convert labels into numbers
labels = df["Email Type"].str.strip().str.lower()

label_mapping = {
    "phishing email": 1,
    "safe email": 0
}

y = labels.map(label_mapping)

if y.isnull().any():
    raise ValueError("Unknown email categories found!")

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(emails)

# Extract URL and keyword features
suspicious_keywords = [
    "verify", "urgent", "password",
    "account", "login", "bank",
    "click here", "winner", "free",
    "suspended"
]

extra_features = []

for email in emails:

    # Count URLs
    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", email)
    )

    # Detect IP address in email
    has_ip = int(bool(
        re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", email)
    ))

    # Count suspicious keywords
    keyword_count = sum(
        email.count(word) for word in suspicious_keywords
    )

    extra_features.append([
        url_count,
        has_ip,
        keyword_count
    ])

# Convert additional features into sparse matrix
X_extra = csr_matrix(extra_features)

# Combine TF-IDF and additional features
X = hstack([X_text, X_extra])

print("========== FEATURE EXTRACTION ==========")

print("\nEmail Records:", len(emails))
print("TF-IDF Features:", X_text.shape[1])
print("Additional Features:", X_extra.shape[1])
print("Total Features:", X.shape[1])

print("\nFeature Matrix Shape:", X.shape)
print("Label Shape:", y.shape)

print("\nLabel Distribution:")
print(y.value_counts())

print("\nFeature Extraction Completed Successfully!")
