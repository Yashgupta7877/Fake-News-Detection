import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download("stopwords")

# Load merged dataset
df = pd.read_csv("../Dataset/merged_news.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and apply stemming
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)

print("\nStarting text preprocessing...")
df["clean_text"] = df["text"].apply(clean_text)
print("Text preprocessing completed!")

# Compare original and cleaned text
print("\n================ ORIGINAL TEXT ================\n")
print(df["text"].iloc[0])

print("\n================ CLEANED TEXT ================\n")
print(df["clean_text"].iloc[0])

# Save cleaned dataset
df.to_csv("../Dataset/cleaned_news.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Saved as: ../Dataset/cleaned_news.csv")