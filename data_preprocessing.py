
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Phishing_Email.csv")

print("========== DATASET INFORMATION ==========")

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column information
print("\nColumn Names:")
print(df.columns.tolist())

# 1. Identify missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 2. Handle missing values
df = df.dropna(subset=["Email Text", "Email Type"])

print("\nMissing Values After Treatment:")
print(df.isnull().sum())

# 3. Identify duplicates
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("Duplicates After Treatment:", df.duplicated().sum())

# 4. Check data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 5. Data transformation
df["Email Text"] = df["Email Text"].astype(str).str.strip()
df["Email Type"] = df["Email Type"].astype(str).str.strip()

# Remove empty email text
df = df[df["Email Text"] != ""]

# Save cleaned dataset
df.to_csv("dataset/processed_emails.csv", index=False)

print("\n========== FINAL DATASET ==========")
print("Final Shape:", df.shape)

print("\nEmail Categories:")
print(df["Email Type"].value_counts())

print("\nCleaned dataset saved successfully!")
