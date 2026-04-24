"""
modelling.py - Basic Model Training with MLflow Autolog

Melatih model machine learning (Scikit-Learn) untuk sentiment analysis
review PUBG Mobile menggunakan MLflow Tracking UI yang disimpan secara lokal.
- Tanpa hyperparameter tuning
- Menggunakan MLflow autolog

Author: Muhammad Rizal Nurfirdaus
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "pubg_mobile_reviews_preprocessed.csv")

# MLflow tracking lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("PUBG_Mobile_Sentiment_Analysis")

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("  MODELLING - Sentiment Analysis PUBG Mobile Reviews")
print("  MLflow Autolog (Basic)")
print("=" * 60)
print()

print("[1/4] Loading preprocessed data...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")
print(f"      Sentiment distribution:")
print(df['sentiment'].value_counts().to_string())
print()

# ============================================================
# 2. Feature Engineering - TF-IDF
# ============================================================
print("[2/4] Feature Engineering (TF-IDF)...")
X = df['review_processed']
y = df['sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)
print(f"      TF-IDF shape: {X_tfidf.shape}")
print()

# ============================================================
# 3. Train-Test Split
# ============================================================
print("[3/4] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print()

# ============================================================
# 4. Model Training with MLflow Autolog
# ============================================================
print("[4/4] Training Logistic Regression with MLflow Autolog...")

# Enable autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogisticRegression_Autolog"):
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n      Accuracy: {accuracy:.4f}")
    print(f"\n      Classification Report:")
    print(report)

    print(f"\n      Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Log additional info
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1, 2)")
    mlflow.log_param("test_size", 0.2)

print()
print("=" * 60)
print("  DONE! Cek MLflow UI di http://127.0.0.1:5000")
print("=" * 60)
