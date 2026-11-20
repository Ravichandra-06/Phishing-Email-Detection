
import joblib
from scipy.sparse import hstack

# ==========================================
# LOAD TRAINED COMBINED MODEL
# ==========================================

MODEL_PATH = "models/combined_phishing_model.pkl"

try:
    data = joblib.load(MODEL_PATH)

    model = data["model"]
    word_vectorizer = data["word_vectorizer"]
    char_vectorizer = data["char_vectorizer"]

    print("\nModel loaded successfully!")

except FileNotFoundError:
    print("Error: Combined model file not found!")
    print("Please run: python train_combined_model.py")
    exit()

except Exception as error:
    print("Error loading model:", error)
    exit()


# ==========================================
# EMAIL PREDICTION FUNCTION
# ==========================================

def predict_email(email):

    email = email.strip().lower()

    if not email:
        print("Please enter a valid email.")
        return

    # Extract word-level features
    word_features = word_vectorizer.transform([email])

    # Extract character-level features
    char_features = char_vectorizer.transform([email])

    # Combine both features
    combined_features = hstack([
        word_features,
        char_features
    ])

    # Predict class
    prediction = model.predict(combined_features)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(combined_features)[0]

    phishing_index = list(model.classes_).index(1)

    phishing_probability = probabilities[phishing_index] * 100

    safe_probability = 100 - phishing_probability

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\n========== EMAIL ANALYSIS ==========")

    if phishing_probability >= 80:

        print("Result: PHISHING EMAIL")
        print("Warning: This email appears highly suspicious.")

    elif phishing_probability >= 50:

        print("Result: SUSPICIOUS EMAIL")
        print("Warning: Manual review is recommended.")

    else:

        print("Result: SAFE EMAIL")
        print("No strong phishing signals detected.")

    print("------------------------------------")

    print(f"Safe probability: {safe_probability:.2f}%")
    print(f"Phishing probability: {phishing_probability:.2f}%")

    print("------------------------------------")

    print("Model prediction:", 
          "Phishing" if prediction == 1 else "Safe")

    print("====================================")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("\n======================================")
    print("   PHISHING EMAIL DETECTION SYSTEM")
    print("   Word + Character TF-IDF Model")
    print("======================================")

    print("\nEnter an email message to analyze.")
    print("Type 'exit' to close the program.")

    while True:

        email = input("\nEnter email text: ")

        if email.strip().lower() == "exit":
            print("\nThank you for using the system!")
            break

        predict_email(email)


# ==========================================
# PROGRAM EXECUTION
# ==========================================

if __name__ == "__main__":
    main()
