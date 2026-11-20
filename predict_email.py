
import pandas as pd
import re
import joblib

from scipy.sparse import csr_matrix, hstack

# Load trained model
data = joblib.load("models/phishing_model.pkl")

model = data["model"]
vectorizer = data["vectorizer"]
keywords = data["keywords"]

print("Phishing Email Detection System")
print("--------------------------------")

def extract_features(email):

    email = email.lower()

    # Count URLs
    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", email)
    )

    # Detect IP address
    has_ip = int(bool(
        re.search(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            email
        )
    ))

    # Count suspicious keywords
    keyword_count = sum(
        email.count(word) for word in keywords
    )

    return csr_matrix([[
        url_count,
        has_ip,
        keyword_count
    ]])

def predict_email(email):

    # Convert email to lowercase
    email = email.lower().strip()

    # TF-IDF transformation
    text_features = vectorizer.transform([email])

    # Extract additional features
    extra_features = extract_features(email)

    # Combine features
    final_features = hstack([
        text_features,
        extra_features
    ])

    # Predict category
    prediction = model.predict(final_features)[0]

    # Prediction probability
    probabilities = model.predict_proba(final_features)[0]
    print("Model classes:", model.classes_)
    print("Class probabilities:", probabilities)
    print("Additional features:", extra_features.toarray()[0])

    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "PHISHING"
    else:
        result = "SAFE"

    print("\n========== ANALYSIS RESULT ==========")

    print("Email:", email[:150])
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2), "%")

    print("\nAnalysis completed!")

# Get email from user
email = input("\nEnter email content: ")

if email.strip():
    predict_email(email)
else:
    print("Please enter valid email content.")
