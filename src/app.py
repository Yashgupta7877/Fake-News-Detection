import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords", quiet=True)

# Load model and vectorizer
model = joblib.load("../models/fake_news_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is likely **Fake** or **True**.")

# Input
news = st.text_area("News Article", height=200)

# Predict button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        if prediction == 0:
            st.error(f"❌ Prediction: FAKE NEWS\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ Prediction: TRUE NEWS\n\nConfidence: {confidence:.2f}%")