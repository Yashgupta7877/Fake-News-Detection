import joblib
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv("../Dataset/cleaned_news.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ---------------------------------------
# Features and Labels
# ---------------------------------------

X = df["clean_text"].fillna("").astype(str)
y = df["label"]

# ---------------------------------------
# TF-IDF Feature Extraction
# ---------------------------------------

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(X)

print("TF-IDF Shape:", X.shape)

# ---------------------------------------
# Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ---------------------------------------
# Models
# ---------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    )
}

results = []

# ---------------------------------------
# Train & Evaluate
# ---------------------------------------

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}...")
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append([name, accuracy])

    print(f"\n{name} Accuracy : {accuracy * 100:.2f}%")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

# ---------------------------------------
# Model Comparison
# ---------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results.sort(key=lambda x: x[1], reverse=True)

for model_name, accuracy in results:
    print(f"{model_name:<25} {accuracy * 100:.2f}%")


# Save the best model (Random Forest)
os.makedirs("../models", exist_ok=True)

best_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

best_model.fit(X_train, y_train)

joblib.dump(best_model, "../models/fake_news_model.pkl")
joblib.dump(tfidf, "../models/tfidf_vectorizer.pkl")

print("\nBest model and vectorizer saved successfully!")