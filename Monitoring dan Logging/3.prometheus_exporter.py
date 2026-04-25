"""
prometheus_exporter.py - ML Model Serving + Prometheus Metrics Exporter

Serves the PUBG Mobile Sentiment Analysis model via Flask API
and exposes 12+ Prometheus metrics for monitoring.

Metrics exposed:
  1.  ml_prediction_total           - Total predictions (Counter)
  2.  ml_prediction_latency_seconds - Prediction latency (Histogram)
  3.  ml_prediction_errors_total    - Total errors (Counter)
  4.  ml_model_accuracy             - Model accuracy (Gauge)
  5.  ml_prediction_by_class        - Predictions per class (Counter)
  6.  ml_model_confidence_score     - Confidence scores (Histogram)
  7.  ml_request_payload_size_bytes  - Request payload size (Histogram)
  8.  ml_active_requests            - In-flight requests (Gauge)
  9.  ml_model_info                 - Model metadata (Info)
  10. ml_feature_count              - Features per request (Histogram)
  11. ml_model_last_prediction_time - Last prediction timestamp (Gauge)
  12. ml_uptime_seconds             - Server uptime (Gauge)

Usage:
  python 3.prometheus_exporter.py
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)

# ============================================================
# Prometheus Metrics (12 metrics)
# ============================================================

# 1. Total predictions counter
PREDICTION_TOTAL = Counter(
    'ml_prediction_total',
    'Total number of predictions made',
    ['model_name']
)

# 2. Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Time spent processing prediction requests',
    ['model_name'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# 3. Prediction errors counter
PREDICTION_ERRORS = Counter(
    'ml_prediction_errors_total',
    'Total number of prediction errors',
    ['model_name', 'error_type']
)

# 4. Model accuracy gauge
MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy on test set',
    ['model_name']
)

# 5. Predictions by class counter
PREDICTION_BY_CLASS = Counter(
    'ml_prediction_by_class',
    'Number of predictions per sentiment class',
    ['model_name', 'sentiment_class']
)

# 6. Model confidence score histogram
CONFIDENCE_SCORE = Histogram(
    'ml_model_confidence_score',
    'Distribution of model confidence scores',
    ['model_name'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# 7. Request payload size histogram
REQUEST_PAYLOAD_SIZE = Histogram(
    'ml_request_payload_size_bytes',
    'Size of incoming request payloads in bytes',
    ['model_name'],
    buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192)
)

# 8. Active requests gauge
ACTIVE_REQUESTS = Gauge(
    'ml_active_requests',
    'Number of currently active prediction requests',
    ['model_name']
)

# 9. Model info
MODEL_INFO = Info(
    'ml_model',
    'Information about the ML model'
)

# 10. Feature count histogram
FEATURE_COUNT = Histogram(
    'ml_feature_count',
    'Number of input features (text length) per request',
    ['model_name'],
    buckets=(1, 5, 10, 20, 50, 100, 200, 500)
)

# 11. Last prediction timestamp gauge
LAST_PREDICTION_TIME = Gauge(
    'ml_model_last_prediction_timestamp',
    'Unix timestamp of the last prediction',
    ['model_name']
)

# 12. Uptime gauge
UPTIME_SECONDS = Gauge(
    'ml_uptime_seconds',
    'Server uptime in seconds'
)

# ============================================================
# Global State
# ============================================================
MODEL_NAME = 'pubg_sentiment_lr'
START_TIME = time.time()
model = None
vectorizer = None
accuracy = 0.0

# ============================================================
# Load and Train Model
# ============================================================
def load_and_train_model():
    """Load data, train TF-IDF + LogisticRegression, and set global model."""
    global model, vectorizer, accuracy

    # Find the preprocessed dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)

    # Look for the preprocessed CSV in multiple locations
    possible_paths = [
        os.path.join(repo_root, "pubg_mobile_reviews_preprocessing", "pubg_mobile_reviews_preprocessed.csv"),
        os.path.join(repo_root, "Workflow-CI", "MLProject", "pubg_mobile_reviews_preprocessed.csv"),
        os.path.join(repo_root, "Membangun_model", "pubg_mobile_reviews_preprocessed.csv"),
    ]

    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        print("ERROR: Could not find preprocessed dataset!")
        print("Searched in:", possible_paths)
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Determine text and label columns
    text_col = None
    for col in ['clean_text', 'cleaned_text', 'text', 'content', 'review']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        # Use the first text-like column
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'sentiment':
                text_col = col
                break

    print(f"Using text column: {text_col}")
    print(f"Dataset shape: {df.shape}")

    # Clean data
    df = df.dropna(subset=[text_col, 'sentiment'])
    X = df[text_col].astype(str)
    y = df['sentiment']

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Set Prometheus metrics
    MODEL_ACCURACY.labels(model_name=MODEL_NAME).set(accuracy)
    MODEL_INFO.info({
        'model_type': 'LogisticRegression',
        'vectorizer': 'TfidfVectorizer',
        'max_features': '5000',
        'ngram_range': '1_2',
        'dataset_size': str(len(df)),
        'accuracy': f'{accuracy:.4f}',
        'version': '1.0.0'
    })

    print(f"Model loaded and ready for serving!")
    return model, vectorizer

# ============================================================
# Routes
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'uptime_seconds': time.time() - START_TIME
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with Prometheus instrumentation."""
    ACTIVE_REQUESTS.labels(model_name=MODEL_NAME).inc()

    try:
        start_time = time.time()

        # Parse input
        data = request.get_json(force=True)
        texts = data.get('inputs', data.get('texts', []))

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            PREDICTION_ERRORS.labels(
                model_name=MODEL_NAME, error_type='empty_input'
            ).inc()
            return jsonify({'error': 'No input texts provided'}), 400

        # Track payload size (metric 7)
        payload_size = len(request.get_data())
        REQUEST_PAYLOAD_SIZE.labels(model_name=MODEL_NAME).observe(payload_size)

        # Track feature count per text (metric 10)
        for t in texts:
            word_count = len(str(t).split())
            FEATURE_COUNT.labels(model_name=MODEL_NAME).observe(word_count)

        # Vectorize and predict
        X_input = vectorizer.transform(texts)
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)

        # Calculate latency (metric 2)
        latency = time.time() - start_time
        PREDICTION_LATENCY.labels(model_name=MODEL_NAME).observe(latency)

        # Increment prediction counter (metric 1)
        PREDICTION_TOTAL.labels(model_name=MODEL_NAME).inc(len(texts))

        # Track predictions by class (metric 5)
        for pred in predictions:
            PREDICTION_BY_CLASS.labels(
                model_name=MODEL_NAME, sentiment_class=str(pred)
            ).inc()

        # Track confidence scores (metric 6)
        for prob_row in probabilities:
            max_confidence = float(np.max(prob_row))
            CONFIDENCE_SCORE.labels(model_name=MODEL_NAME).observe(max_confidence)

        # Update last prediction timestamp (metric 11)
        LAST_PREDICTION_TIME.labels(model_name=MODEL_NAME).set(time.time())

        # Build response
        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            conf = float(np.max(probabilities[i]))
            results.append({
                'text': text,
                'sentiment': str(pred),
                'confidence': round(conf, 4)
            })

        return jsonify({
            'predictions': results,
            'latency_ms': round(latency * 1000, 2),
            'count': len(results)
        })

    except Exception as e:
        PREDICTION_ERRORS.labels(
            model_name=MODEL_NAME, error_type=type(e).__name__
        ).inc()
        return jsonify({'error': str(e)}), 500

    finally:
        ACTIVE_REQUESTS.labels(model_name=MODEL_NAME).dec()


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    # Update uptime (metric 12)
    UPTIME_SECONDS.set(time.time() - START_TIME)
    return generate_latest(REGISTRY), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'service': 'PUBG Mobile Sentiment Analysis - ML Monitoring',
        'endpoints': {
            '/predict': 'POST - Make predictions',
            '/metrics': 'GET - Prometheus metrics',
            '/health': 'GET - Health check'
        },
        'model': MODEL_NAME,
        'accuracy': accuracy
    })


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  ML Model Serving + Prometheus Exporter")
    print("  Port: 8000")
    print("=" * 60)

    # Load model
    load_and_train_model()

    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=False)
