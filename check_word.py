import pandas as pd

df = pd.read_csv("dataset/processed_emails_cleaned.csv")

df["Email Text"] = df["Email Text"].astype(str).str.lower()

word = "hello"

matching = df[df["Email Text"].str.contains(
    rf"\b{word}\b",
    regex=True,
    na=False
)]

print("\n========== HELLO WORD ANALYSIS ==========")

print("Total emails containing hello:", len(matching))

print("\nLabel distribution:")
print(matching["Email Type"].value_counts())

print("\nSample emails containing hello:")

for _, row in matching.head(10).iterrows():
    print("\nLabel:", row["Email Type"])
    print("Email:", row["Email Text"][:300])
    print("--------------------------------")