import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Load Cleaned Dataset
# -------------------------------------------------

df = pd.read_csv("../Dataset/cleaned_news.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# -------------------------------------------------
# Features (X) and Labels (y)
# -------------------------------------------------

print(df.columns)
print(df["clean_text"].head())
print(df["clean_text"].dtype)
print("Missing values:", df["clean_text"].isnull().sum())

X = df["clean_text"].fillna("").astype(str)
y = df["label"]

# -------------------------------------------------
# Bag of Words
# -------------------------------------------------

bow = CountVectorizer(max_features=5000)

X_bow = bow.fit_transform(X)

print("\nBag of Words Shape:")
print(X_bow.shape)

# -------------------------------------------------
# TF-IDF
# -------------------------------------------------

tfidf = TfidfVectorizer(max_features=5000)

X_tfidf = tfidf.fit_transform(X)

print("\nTF-IDF Shape:")
print(X_tfidf.shape)

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

print("\nTraining Labels:", y_train.shape)
print("Testing Labels :", y_test.shape)

print("\nFeature Engineering Completed Successfully!")