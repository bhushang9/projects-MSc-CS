# 🧠 Mental Health Analytics System (Version 2)

A machine learning–based web application that predicts the likelihood of requiring mental health treatment based on workplace, demographic, and personal factors, enhanced with sentiment analysis of user journal entries. This Version 2 is a refined, stable, and production-ready upgrade over the initial implementation, focusing on correctness, consistency, and deployment reliability.

---

## 📌 Project Overview

This project analyzes mental health survey data to:
- Predict whether an individual may require mental health treatment
- Provide confidence scores for predictions
- Analyze emotional sentiment from optional user journal entries
  
Version 2 is a refined and stable implementation focused on correctness, clean ML pipelines, and deployability using Streamlit.\

---

## ✨ Key Features

- End-to-end ML pipeline (preprocessing → training → inference)
- Robust data cleaning and normalization
- Multiple ML models with automatic best-model selection
- Confidence-based predictions
- Journal sentiment analysis using NLP
- Interactive Streamlit web application
- Downloadable prediction report (CSV)
- Clean and consistent artifact management
  
---

## 🧾 Input Features Used for Prediction

The model is trained on the following features:

| Feature        | Description                            |
| -------------- | -------------------------------------- |
| Age            | Normalized age of the user             |
| Gender         | male / female / trans                  |
| family_history | Family history of mental health issues |
| benefits       | Employer mental health benefits        |
| care_options   | Awareness of care options              |
| anonymity      | Workplace anonymity protection         |
| leave          | Ease of taking mental health leave     |
| work_interfere | Impact of mental health on work        |

---

## 🤖 Machine Learning Models Used

During training, the following models are evaluated:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM) (with probability enabled)
- K-Nearest Neighbors (KNN)

The best-performing model (based on validation accuracy) is automatically selected and saved for deployment.

---

## 🔍 Sentiment Analysis (NLP)

- Uses TextBlob
- Analyzes optional journal entries
- Outputs:
 - Sentiment label: Positive / Neutral / Negative
 - Polarity score (−1 to +1)

Sentiment does not affect prediction (kept interpretable)

---

## 🔄 Why Version 2?

Version 2 was created to fix limitations identified in the earlier experimental version:

Removed in V2:
❌ Bagging & Boosting ensembles
❌ Over-engineered pipelines
❌ Feature mismatch issues
❌ Inference-time encoder failures

Why they were removed:
- They increased complexity without improving reliability
- Caused deployment instability
- Not suitable for a real-time Streamlit app
- Reduced interpretability for academic evaluation

Added in V2:
✔ Stable feature alignment
✔ Safe label encoding
✔ Consistent scaling
✔ Deployment-ready inference pipeline

---

## ⚙️ Tech Stack

Python 3.11
Pandas, NumPy
Scikit-learn
TextBlob (NLP)
Streamlit
Joblib

---
