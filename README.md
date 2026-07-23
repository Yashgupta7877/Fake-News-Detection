# 📰 Fake News Detection System

## 📌 Project Overview

The Fake News Detection System is a Machine Learning project that classifies news articles as **Fake** or **True**. It uses Natural Language Processing (NLP) techniques and a Random Forest classifier trained on a labeled news dataset.

---

## 🚀 Features

- Detects Fake or True news articles
- Text preprocessing using NLP
- TF-IDF feature extraction
- Random Forest machine learning model
- Streamlit-based web application
- Displays prediction with confidence score

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Joblib
- Matplotlib

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── Dataset/
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── output/
│   └── model_accuracy.png
│
├── src/
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── text_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Preprocessing
3. Text Cleaning
4. TF-IDF Feature Extraction
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Save Best Model
9. Streamlit Web Application

---

## 🤖 Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest ⭐
- Multi-Layer Perceptron (Neural Network)

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run src/app.py
```

---

## 📈 Results

The Random Forest model achieved the highest accuracy on the test dataset and was selected as the final model.

---

## ⚠️ Limitations

- The model was trained on a specific fake/true news dataset.
- It performs best on news similar to the training data.
- It may not generalize well to modern or region-specific news articles.

---

## 🔮 Future Improvements

- Train on a larger and more diverse dataset.
- Add support for multilingual news.
- Integrate live news APIs for real-time analysis.
- Deploy the application online.

---

## 📸 Application Screenshots

### Home Page

![Home](output/screenshots/Landing_page.png)

### True News Prediction

![True Prediction](output/screenshots/TRUE.png)

### Fake News Prediction

![Fake Prediction](output/screenshots/FAKE.png)

## 👨‍💻 Author

Yash Gupta

B.Tech Student

Machine Learning Project