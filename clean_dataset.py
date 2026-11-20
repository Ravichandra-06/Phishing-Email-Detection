import pandas as pd
import os

# Load original dataset
input_file = "dataset/processed_emails.csv"

df = pd.read_csv(input_file)

print("Original dataset:", len(df))

# Remove missing values
df = df.dropna(subset=["Email Text", "Email Type"]).copy()

# Standardize labels
df["Email Type"] = (
    df["Email Type"].astype(str).str.strip()
)

# Create normalized text for duplicate detection
df["normalized_text"] = (
    df["Email Text"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 1. Remove empty and placeholder emails
before = len(df)

df = df[
    ~df["normalized_text"].isin(["", "empty"])
].copy()

print("Placeholder records removed:", before - len(df))

# 2. Remove records with conflicting labels
label_counts = (
    df.groupby("normalized_text")["Email Type"].nunique()
)

conflicting_texts = label_counts[
    label_counts > 1
].index

conflict_rows = df[
    df["normalized_text"].isin(conflicting_texts)
].copy()

df = df[
    ~df["normalized_text"].isin(conflicting_texts)
].copy()

print("Conflicting records removed:", len(conflict_rows))

# 3. Remove duplicate email text
before = len(df)

df = df.drop_duplicates(
    subset=["normalized_text"],
    keep="first"
).copy()

print("Duplicate records removed:", before - len(df))

# 4. Separate extremely large records for manual inspection
extreme_records = df[
    df["Email Text"].str.len() > 1_000_000
].copy()

df = df[
    df["Email Text"].str.len() <= 1_000_000
].copy()

# Create output folder
os.makedirs("dataset", exist_ok=True)

# Save cleaned dataset
df[["Email Text", "Email Type"]].to_csv(
    "dataset/processed_emails_cleaned.csv",
    index=False
)

# Save records requiring manual review
conflict_rows.to_csv(
    "dataset/conflicting_emails_review.csv",
    index=False
)

extreme_records.to_csv(
    "dataset/extreme_emails_review.csv",
    index=False
)

print("\n========== CLEANING REPORT ==========")
print("Original records:", pd.read_csv(input_file).shape[0])
print("Cleaned records:", len(df))
print("Conflicting records:", len(conflict_rows))
print("Extreme records:", len(extreme_records))

print("\nFinal label distribution:")
print(df["Email Type"].value_counts())

print("\nCleaned dataset saved successfully!")