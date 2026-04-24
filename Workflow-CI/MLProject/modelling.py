"""
modelling.py - MLflow Project CI Model Training

Melatih model machine learning (Scikit-Learn) untuk sentiment analysis
review PUBG Mobile menggunakan MLflow Tracking UI.
Diset khusus untuk port 2026 agar tidak bentrok.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "pubg_mobile_reviews_preprocessed.csv")

# Set tracking URI to port 5000
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("PUBG_Mobile_Sentiment_CI")

print("=" * 60)
print("  MODELLING - Workflow CI (Port 5000)")
print("=" * 60)

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# 2. Feature Engineering
X = df['review_processed'].astype(str)
y = df['sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model Training with MLflow Autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogisticRegression_CI") as run:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Menuliskan RUN_ID ke sebuah file agar dapat diambil oleh GitHub Actions
    run_id = run.info.run_id
    workspace = os.environ.get("GITHUB_WORKSPACE", ".")
    with open(os.path.join(workspace, "run_id.txt"), "w") as f:
        f.write(run_id)
        
    print(f"Run ID: {run_id} berhasil disimpan.")

print("DONE!")

