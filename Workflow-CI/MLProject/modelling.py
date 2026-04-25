"""
modelling.py - MLflow Project CI Model Training

Melatih model machine learning (Scikit-Learn) untuk sentiment analysis
review PUBG Mobile menggunakan MLflow Tracking UI.
Script ini dijalankan oleh GitHub Actions CI pipeline.

Author: Muhammad Rizal Nurfirdaus
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "pubg_mobile_reviews_preprocessed.csv")
RUN_ID_PATH = os.path.join(BASE_DIR, "run_id.txt")

# MLflow tracking URI — default ke port 5000, bisa di-override via env var
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("PUBG_Mobile_Sentiment_CI")

print("=" * 60)
print("  MODELLING - Workflow CI")
print(f"  MLflow Tracking: {TRACKING_URI}")
print("=" * 60)
print()

# ============================================================
# 1. Load Data
# ============================================================
print("[1/4] Loading preprocessed data...")
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset tidak ditemukan di {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")
print(f"      Sentiment distribution:")
print(df['sentiment'].value_counts().to_string())
print()

# ============================================================
# 2. Feature Engineering - TF-IDF
# ============================================================
print("[2/4] Feature Engineering (TF-IDF)...")
X = df['review_processed'].astype(str)
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

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogisticRegression_CI") as run:
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n      Accuracy: {accuracy:.4f}")
    print(f"\n      Classification Report:")
    print(report)

    # Log additional params
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1, 2)")
    mlflow.log_param("test_size", 0.2)

    # Simpan Run ID ke file untuk digunakan CI pipeline
    run_id = run.info.run_id
    with open(RUN_ID_PATH, "w") as f:
        f.write(run_id)
    print(f"\n      Run ID: {run_id}")
    print(f"      Run ID saved to: {RUN_ID_PATH}")

print()
print("=" * 60)
print("  DONE!")
print("=" * 60)
