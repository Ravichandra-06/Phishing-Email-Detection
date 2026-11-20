
import streamlit as st
import pandas as pd
import joblib
import json
import os

from scipy.sparse import hstack
from datetime import datetime


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# LOAD COMBINED TRAINED MODEL
# ==========================================

MODEL_PATH = "models/combined_phishing_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    data = load_model()

    model = data["model"]
    word_vectorizer = data["word_vectorizer"]
    char_vectorizer = data["char_vectorizer"]

    model_name = data.get(
        "model_name",
        "Combined Word + Character TF-IDF"
    )

except FileNotFoundError:
    st.error(
        "Combined model not found. "
        "Run python train_combined_model.py first."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# EMAIL PREDICTION FUNCTION
# ==========================================

def predict_email(email):

    email = email.strip().lower()

    if not email:
        raise ValueError("Email content cannot be empty.")

    # Word-level TF-IDF features
    word_features = word_vectorizer.transform([email])

    # Character-level TF-IDF features
    char_features = char_vectorizer.transform([email])

    # Combine both feature sets
    combined_features = hstack([
        word_features,
        char_features
    ])

    # Get prediction probabilities
    probabilities = model.predict_proba(
        combined_features
    )[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    safe_probability = (
        float(class_probabilities.get(0, 0)) * 100
    )

    phishing_probability = (
        float(class_probabilities.get(1, 0)) * 100
    )

    # Three-category classification
    if phishing_probability >= 80:

        result = "PHISHING"

    elif phishing_probability >= 50:

        result = "SUSPICIOUS"

    else:

        result = "SAFE"

    confidence = max(
        safe_probability,
        phishing_probability
    )

    return {
        "result": result,
        "safe_probability": safe_probability,
        "phishing_probability": phishing_probability,
        "confidence": confidence
    }


# ==========================================
# LOAD MODEL PERFORMANCE
# ==========================================

def load_performance():

    file_path = "models/performance.json"

    if os.path.exists(file_path):

        try:

            with open(file_path, "r") as file:
                return json.load(file)

        except (json.JSONDecodeError, OSError):
            return None

    return None


performance = load_performance()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.title("🛡️ Cyber Guardian")

    st.caption(
        "Phishing Email Detection System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Email Analyzer",
            "Prediction History",
            "Model Performance"
        ]
    )

    st.divider()

    st.subheader("System Status")

    st.success("Combined Model Loaded")

    st.caption(f"Algorithm: {model_name}")


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

if page == "Overview":

    st.title("🛡️ Security Overview")

    st.write(
        "Welcome to the Phishing Email Detection System."
    )

    st.caption(
        "Analyze email content and identify potentially "
        "suspicious messages using machine learning."
    )

    st.divider()

    history = st.session_state.history

    total = len(history)

    phishing = sum(
        item["Prediction"] == "PHISHING"
        for item in history
    )

    safe = sum(
        item["Prediction"] == "SAFE"
        for item in history
    )

    suspicious = sum(
        item["Prediction"] == "SUSPICIOUS"
        for item in history
    )

    # Dashboard metrics

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Emails", total)

    col2.metric("Phishing", phishing)

    col3.metric("Safe Emails", safe)

    col4.metric("Suspicious", suspicious)

    col5.metric(
        "Phishing Rate",
        f"{(phishing / total * 100):.1f}%"
        if total else "0%"
    )

    st.divider()

    st.subheader("Machine Learning Model")

    col1, col2 = st.columns(2)

    col1.metric(
        "Selected Algorithm",
        "Logical Aggregate"
    )

    col2.metric(
        "Feature Extraction",
        "Word + Character TF-IDF"
    )

    st.divider()

    st.subheader("Recent Analysis")

    if history:

        recent_df = pd.DataFrame(
            history
        ).tail(5)

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No emails analyzed yet. "
            "Open Email Analyzer to begin."
        )


# ==========================================
# PAGE 2: EMAIL ANALYZER
# ==========================================

elif page == "Email Analyzer":

    st.title("📧 Email Security Analyzer")

    st.write(
        "Enter email content to check whether it is "
        "potentially phishing, suspicious or safe."
    )

    st.divider()

    email = st.text_area(
        "Email Content",
        height=220,
        placeholder="Paste email content here..."
    )

    if st.button(
        "Analyze Email",
        type="primary"
    ):

        if email.strip():

            try:

                analysis = predict_email(email)

                result = analysis["result"]

                safe_probability = analysis[
                    "safe_probability"
                ]

                phishing_probability = analysis[
                    "phishing_probability"
                ]

                confidence = analysis["confidence"]

                # Save prediction history

                st.session_state.history.append({

                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "Email": email[:200],

                    "Prediction": result,

                    "Confidence": round(
                        confidence, 2
                    ),

                    "Safe Probability": round(
                        safe_probability, 2
                    ),

                    "Phishing Probability": round(
                        phishing_probability, 2
                    )

                })

                st.divider()

                st.subheader("Analysis Result")

                # Display classification

                if result == "PHISHING":

                    st.error(
                        "⚠️ Potential Phishing Email Detected"
                    )

                elif result == "SUSPICIOUS":

                    st.warning(
                        "⚠️ Suspicious Email - Manual Review Recommended"
                    )

                else:

                    st.success(
                        "✅ Classified as Safe Email"
                    )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Prediction",
                    result
                )

                col2.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.divider()

                st.subheader("Probability Analysis")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Safe Probability",
                    f"{safe_probability:.2f}%"
                )

                col2.metric(
                    "Phishing Probability",
                    f"{phishing_probability:.2f}%"
                )

                # Probability chart

                probability_df = pd.DataFrame({

                    "Class": [
                        "Safe",
                        "Phishing"
                    ],

                    "Probability": [
                        safe_probability,
                        phishing_probability
                    ]

                }).set_index("Class")

                st.bar_chart(probability_df)

                st.caption(
                    "Predictions are estimates from the trained "
                    "model, not guarantees. Verify suspicious "
                    "messages independently."
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

        else:

            st.warning(
                "Please enter email content."
            )


# ==========================================
# PAGE 3: PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "Review previously analyzed emails in this session."
    )

    st.divider()

    history = st.session_state.history

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Download history as CSV

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download CSV Report",
            data=csv_data,
            file_name="phishing_analysis_history.csv",
            mime="text/csv"
        )

        # Clear history

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Analyze an email to create history."
        )


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation information from the original "
        "word-level model training."
    )

    st.warning(
        "The performance.json and confusion matrix below "
        "belong to the original model, not the newly trained "
        "combined model. Do not interpret these metrics as "
        "combined-model evaluation results."
    )

    st.divider()

    if performance:

        # Original model evaluation metrics

        st.subheader(
            "Original Model Evaluation Metrics"
        )

        metric_columns = st.columns(4)

        metric_fields = [

            ("Accuracy", "accuracy"),

            ("Precision", "precision"),

            ("Recall", "recall"),

            ("F1 Score", "f1_score")

        ]

        for column, (label, key) in zip(
            metric_columns,
            metric_fields
        ):

            value = performance.get(key)

            if isinstance(value, (int, float)):

                column.metric(
                    label,
                    f"{value * 100:.2f}%"
                )

            else:

                column.metric(
                    label,
                    "N/A"
                )

        st.divider()

        # Confusion matrix values

        st.subheader("Confusion Matrix Analysis")

        cm_values = performance.get(
            "confusion_matrix",
            {}
        )

        if isinstance(cm_values, dict) and cm_values:

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "True Negative",
                cm_values.get(
                    "true_negative",
                    "N/A"
                )
            )

            col2.metric(
                "False Positive",
                cm_values.get(
                    "false_positive",
                    "N/A"
                )
            )

            col3.metric(
                "False Negative",
                cm_values.get(
                    "false_negative",
                    "N/A"
                )
            )

            col4.metric(
                "True Positive",
                cm_values.get(
                    "true_positive",
                    "N/A"
                )
            )

        else:

            st.info(
                "Confusion matrix counts are unavailable."
            )

        st.divider()

        # Model comparison

        st.subheader("Original Model Comparison")

        comparison = performance.get(
            "model_comparison",
            {}
        )

        if comparison:

            comparison_df = pd.DataFrame(
                comparison
            ).T

            required_columns = [

                "accuracy",
                "precision",
                "recall",
                "f1_score"

            ]

            available_columns = [

                column
                for column in required_columns
                if column in comparison_df.columns

            ]

            if available_columns:

                chart_df = comparison_df[
                    available_columns
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                st.bar_chart(chart_df)

                st.dataframe(
                    comparison_df[
                        available_columns
                    ].style.format(
                        "{:.2%}"
                    ),
                    use_container_width=True
                )

            else:

                st.info(
                    "Model comparison metrics are unavailable."
                )

        else:

            st.info(
                "Model comparison data not found."
            )

        st.divider()

        # Classification report

        st.subheader("Classification Report")

        report = performance.get(
            "classification_report",
            {}
        )

        if report:

            report_df = pd.DataFrame(
                report
            ).T

            st.dataframe(
                report_df.style.format("{:.2f}"),
                use_container_width=True
            )

        else:

            st.info(
                "Classification report not found."
            )

        st.divider()

        # Confusion matrix image

        st.subheader("Original Model Confusion Matrix")

        confusion_path = "dataset/confusion_matrix.png"

        if os.path.exists(confusion_path):

            st.image(
                confusion_path,
                caption="Actual vs Predicted Email Classification",
                use_container_width=True
            )

        else:

            st.info(
                "Confusion matrix image not found."
            )

        st.divider()

        st.subheader("Selected Model")

        st.success(
            performance.get(
                "model_name",
                "Original trained model"
            )
        )

    else:

        st.warning(
            "Original model performance data not found."
        )


# ==========================================
# END OF APPLICATION
# ==========================================
