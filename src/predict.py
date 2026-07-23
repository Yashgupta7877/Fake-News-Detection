import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    cleaned = []
    for word in words:
        if word not in stop_words:
            cleaned.append(stemmer.stem(word))

    return " ".join(cleaned)

# Load model
model = joblib.load("../models/fake_news_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

print("====================================")
print(" Fake News Detection System")
print("====================================")

while True:

    news = input("\nEnter News Text:\n")

    if news.lower() == "exit":
        break

    # Clean the text before prediction
    news = clean_text(news)

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        print("\nPrediction : FAKE NEWS")
    else:
        print("\nPrediction : TRUE NEWS")