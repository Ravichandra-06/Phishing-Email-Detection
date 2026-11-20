
# Phishing Email Detection Using Machine Learning

## 1. Project Overview

Phishing Email Detection is a machine learning-based application designed to identify potentially harmful phishing emails and distinguish them from legitimate emails.

The project uses a Logical Aggregate approach with Word-Level and Character-Level TF-IDF feature extraction and Logistic Regression to analyze email content and estimate phishing probability.

The application provides an interactive dashboard built using Streamlit.

## 2. Objectives

- Detect phishing emails using machine learning.
- Identify legitimate and suspicious email messages.
- Apply text preprocessing and feature extraction.
- Combine word-level and character-level TF-IDF features.
- Display prediction results and probability scores.
- Evaluate model performance using machine learning metrics.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- SciPy
- Joblib

## 4. Machine Learning Model

**Selected Algorithm:** Logical Aggregate

**Feature Extraction:** Word + Character TF-IDF

**Classifier:** Logistic Regression

The model combines word-level and character-level text features into a single feature representation before classification.

## 5. Project Features

- Email text analysis
- Phishing probability prediction
- Safe and suspicious email classification
- Interactive Streamlit dashboard
- Prediction history
- Model performance visualization
- Dataset preprocessing and analysis
- Combined TF-IDF feature extraction

## 6. Project Structure

```text
Phishing-Email-Detection/
│
├── app.py
├── data_preprocessing.py
├── feature_extraction.py
├── train_model.py
├── predict_email.py
├── eda.py
├── clean_dataset.py
├── check_word.py
├── audit_dataset.py
├── word_analysis.py
├── check_duplicates.py
├── check_leakage.py
├── compare_features.py
├── train_combined_model.py
├── predict_combined.py
│
├── dataset/
│   ├── Phishing_Email.csv
│   ├── processed_emails.csv
│   ├── processed_emails_cleaned.csv
│   ├── confusion_matrix.png
│   ├── email_distribution.png
│   └── email_length.png
│
├── models/
│   ├── phishing_model.pkl
│   ├── combined_phishing_model.pkl
│   └── performance.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/Ravichandra-06/Phishing-Email-Detection.git
```

Navigate to the project folder:

```bash
cd Phishing-Email-Detection
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment on macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the dashboard.

## 9. Model Workflow

1. Dataset collection
2. Data preprocessing
3. Text cleaning
4. Word and character TF-IDF feature extraction
5. Logical Aggregate feature combination
6. Logistic Regression classification
7. Phishing probability prediction
8. Model evaluation and visualization

## 10. Dataset

The project uses an email dataset containing safe and phishing email examples.

The dataset is cleaned and preprocessed before model training.

## 11. Future Enhancements

- Improve prediction accuracy.
- Reduce false-positive predictions.
- Support email file uploads.
- Add real-time email analysis.
- Explore additional machine learning models.

## 12. Author

**GitHub:** [Ravichandra-06](https://github.com/Ravichandra-06)

## 13. Disclaimer

This project is developed for educational and academic purposes. Predictions are model estimates and should not be treated as a guarantee that an email is safe or malicious.
