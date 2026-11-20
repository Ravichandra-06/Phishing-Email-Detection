import pandas as pd

df = pd.read_csv("dataset/processed_emails_cleaned.csv")

df["Email Text"] = (
    df["Email Text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

print("\n========== DUPLICATE ANALYSIS ==========")

print("Total records:", len(df))
print("Unique email texts:", df["Email Text"].nunique())

print("\nDuplicate emails:", df.duplicated(
    subset=["Email Text"]
).sum())

print("\nLabel distribution:")
print(df["Email Type"].value_counts())

print("\nEmails containing Enron:")
print(df["Email Text"].str.contains(
    "enron", na=False
).sum())