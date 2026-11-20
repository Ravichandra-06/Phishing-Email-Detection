import pandas as pd
import re

df = pd.read_csv("dataset/processed_emails_cleaned.csv")

def clean_preview(text):
    text = str(text)

    # Hide email addresses
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)

    # Hide URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Hide phone-like numbers
    text = re.sub(r'\b\d{10,}\b', '[NUMBER]', text)

    return text[:350].replace("\n", " ")

print("\n========== DATASET LABEL AUDIT ==========")

for label in df["Email Type"].unique():

    samples = df[df["Email Type"] == label].sample(
        n=min(10, len(df[df["Email Type"] == label])),
        random_state=42
    )

    print(f"\n\n========== {label} ==========")

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\nExample {i}:")
        print(clean_preview(row["Email Text"]))
        print("-" * 60)
        