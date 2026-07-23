import pandas as pd

# Load datasets
fake_df = pd.read_csv("../Dataset/Fake.csv")
true_df = pd.read_csv("../Dataset/True.csv")

# Display first 5 rows
print("Fake News Dataset")
print(fake_df.head())

print("\nTrue News Dataset")
print(true_df.head())

# Assign labels
fake_df["label"] = 0   # Fake News
true_df["label"] = 1   # True News

# Merge datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head())

print("Dataset Shape:", df.shape)

print(df.columns)

print(df.isnull().sum())

print(df["label"].value_counts())


df.to_csv("../Dataset/merged_news.csv", index=False)
print("Merged dataset saved successfully!")