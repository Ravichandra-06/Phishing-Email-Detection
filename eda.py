
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("dataset/processed_emails.csv")

print("========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Dataset shape
print("\nDataset Shape:")
print(df.shape)

# 2. Dataset information
print("\nDataset Information:")
print(df.info())

# 3. Email category distribution
print("\nEmail Categories:")
print(df["Email Type"].value_counts())

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Email text length
df["Email Length"] = df["Email Text"].str.len()

print("\nEmail Length Statistics:")
print(df["Email Length"].describe())

# 7. Email category count chart
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Email Type")

plt.title("Phishing vs Safe Emails")
plt.xlabel("Email Category")
plt.ylabel("Number of Emails")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("dataset/email_distribution.png")
plt.show()

# 8. Email length distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Email Length",
    hue="Email Type",
    bins=50
)

plt.title("Email Length Distribution")
plt.xlabel("Email Length")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("dataset/email_length.png")
plt.show()

print("\nEDA Completed Successfully!")
