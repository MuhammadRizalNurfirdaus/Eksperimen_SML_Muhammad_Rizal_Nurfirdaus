"""
modelling_tuning.py - Skilled/Advanced Model Training with Manual MLflow Logging

Melatih model machine learning dengan hyperparameter tuning menggunakan
MLflow Tracking UI. Menggunakan manual logging (bukan autolog) dengan
metriks yang sama + artefak tambahan.

Tahapan:
1. Load data preprocessed
2. TF-IDF Vectorization
3. Hyperparameter tuning (GridSearchCV)
4. Manual MLflow logging (params, metrics, artifacts)

Author: Muhammad Rizal Nurfirdaus
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "pubg_mobile_reviews_preprocessed.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "mlflow_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# MLflow tracking via DagsHub (Advance)
import dagshub
dagshub.init(repo_owner='MuhammadRizalNurfirdaus', repo_name='Eksperimen_SML_Muhammad_Rizal_Nurfirdaus', mlflow=True)
mlflow.set_experiment("PUBG_Mobile_Sentiment_Tuning")

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("  MODELLING TUNING - Sentiment Analysis PUBG Mobile")
print("  Manual MLflow Logging + Hyperparameter Tuning (Skilled)")
print("=" * 60)
print()

print("[1/5] Loading preprocessed data...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")
print(f"      Sentiment distribution:")
print(df['sentiment'].value_counts().to_string())
print()

# ============================================================
# 2. Feature Engineering - TF-IDF
# ============================================================
print("[2/5] Feature Engineering (TF-IDF)...")
X = df['review_processed']
y = df['sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)
print(f"      TF-IDF shape: {X_tfidf.shape}")

# ============================================================
# 3. Train-Test Split
# ============================================================
print("[3/5] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print()


def save_confusion_matrix(y_true, y_pred, labels, filepath):
    """Simpan confusion matrix sebagai gambar."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)


def save_classification_report_json(y_true, y_pred, filepath):
    """Simpan classification report sebagai JSON."""
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def train_and_log_model(model_name, model, param_grid, X_train, X_test,
                        y_train, y_test):
    """Train model dengan GridSearchCV dan log ke MLflow secara manual."""

    print(f"\n--- Training: {model_name} ---")

    with mlflow.start_run(run_name=f"{model_name}_Tuning"):

        # GridSearchCV
        print(f"    Running GridSearchCV...")
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_best_score = grid_search.best_score_

        # Predictions
        y_pred = best_model.predict(X_test)

        # === MANUAL LOGGING ===

        # Log Parameters (manual - bukan autolog)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # Log Metrics (manual - sama dengan yang ada di autolog)
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        prec_weighted = precision_score(y_test, y_pred, average='weighted')
        rec_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec_macro)
        mlflow.log_metric("recall_macro", rec_macro)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_weighted", prec_weighted)
        mlflow.log_metric("recall_weighted", rec_weighted)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("cv_best_score", cv_best_score)
        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("test_samples", X_test.shape[0])

        # Log Model (manual)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # === ARTEFAK TAMBAHAN (untuk Advance) ===

        # Artifact 1: Confusion Matrix image
        cm_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")
        labels = ['negatif', 'netral', 'positif']
        save_confusion_matrix(y_test, y_pred, labels, cm_path)
        mlflow.log_artifact(cm_path)

        # Artifact 2: Classification Report JSON
        report_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_classification_report.json")
        save_classification_report_json(y_test, y_pred, report_path)
        mlflow.log_artifact(report_path)

        # Print results
        print(f"    Best Params: {best_params}")
        print(f"    CV Score:    {cv_best_score:.4f}")
        print(f"    Accuracy:    {acc:.4f}")
        print(f"    F1 Macro:    {f1_macro:.4f}")
        print(f"    F1 Weighted: {f1_weighted:.4f}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'best_params': best_params
        }


# ============================================================
# 4. Define Models & Hyperparameter Grids
# ============================================================
print("[4/5] Hyperparameter Tuning with GridSearchCV + Manual MLflow Logging...")

models_config = [
    {
        'name': 'LogisticRegression',
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    },
    {
        'name': 'LinearSVC',
        'model': LinearSVC(random_state=42, max_iter=2000),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge']
        }
    }
]

# ============================================================
# 5. Train All Models
# ============================================================
print("[5/5] Training models...")
results = []

for config in models_config:
    result = train_and_log_model(
        model_name=config['name'],
        model=config['model'],
        param_grid=config['params'],
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )
    results.append(result)

# Summary
print()
print("=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Model':<25} {'Accuracy':<12} {'F1 Macro':<12}")
print("-" * 60)
for r in results:
    print(f"  {r['model_name']:<25} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f}")

best = max(results, key=lambda x: x['f1_macro'])
print(f"\n  Best Model: {best['model_name']} (F1 Macro: {best['f1_macro']:.4f})")
print()
print("=" * 60)
print("  DONE! Cek MLflow UI di http://127.0.0.1:5000")
print("=" * 60)
