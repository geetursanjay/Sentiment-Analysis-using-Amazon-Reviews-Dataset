# 📈 Amazon Product Review Sentiment Analysis

This project performs sentiment analysis on Amazon-style product reviews using **Natural Language Processing (NLP)** and **Machine Learning (Logistic Regression)**.

---

## 📂 Dataset

- File: `amazon_reviews_balanced.csv`
- Samples: 200 reviews (100 Positive, 100 Negative)
- Format: CSV with `Review` and `Sentiment` columns
- Source: Synthesized for this project

---

## 💡 Objectives

- Preprocess product reviews (text cleaning, stopword removal)
- Convert text to vectors using TF-IDF
- Train a classifier to predict sentiment
- Evaluate model performance (Precision, Recall, F1, Accuracy)

---

## 🔧 Technologies Used

- Python 3.x
- Jupyter Notebook / Google Colab
- Libraries: `pandas`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🧠 Model

- **Algorithm**: Logistic Regression
- **Accuracy Achieved**: 93%
- **Precision (Negative/Positive)**: 0.95 / 0.90
- **Recall (Negative/Positive)**: 0.90 / 0.95

---

## 🧪 Try It Yourself

```python
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]
