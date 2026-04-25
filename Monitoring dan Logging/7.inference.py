"""
inference.py - Traffic Generator for ML Model Monitoring

Sends varied prediction requests to the ML model serving endpoint
to generate realistic traffic patterns for Prometheus/Grafana monitoring.

Usage:
  python 7.inference.py
"""

import requests
import time
import random
import json
import sys

# ============================================================
# Configuration
# ============================================================
MODEL_URL = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"

# Sample review texts for generating traffic
POSITIVE_REVIEWS = [
    "game ini sangat seru dan grafisnya bagus sekali",
    "saya suka bermain pubg mobile setiap hari",
    "update terbaru sangat keren fiturnya mantap",
    "grafis game ini sangat realistis dan smooth",
    "gameplay yang seru dan menantang saya ketagihan",
    "map baru sangat bagus desainnya detail",
    "mode permainan baru sangat menyenangkan",
    "kontrol game sangat responsif dan nyaman",
    "efek suara dan musik sangat immersive",
    "game terbaik yang pernah saya mainkan di hp",
    "recommended banget buat pecinta battle royale",
    "sangat puas dengan performa game ini",
    "skin dan kostum barunya keren-keren semua",
    "anti cheat sudah membaik game jadi fair",
    "voice chat berfungsi dengan baik untuk teamwork",
]

NEGATIVE_REVIEWS = [
    "banyak cheater dan hacker yang merusak permainan",
    "lag terus tidak bisa main dengan lancar",
    "update terbaru membuat game semakin berat",
    "terlalu banyak bug setelah update baru",
    "server sering down dan tidak stabil",
    "matchmaking tidak adil pemula vs pro player",
    "terlalu banyak iklan mengganggu pengalaman bermain",
    "game ini terlalu pay to win tidak adil",
    "baterai cepat habis saat bermain game ini",
    "desync parah peluru tidak kena padahal sudah aim",
    "game crash terus menerus sangat menjengkelkan",
    "customer service tidak membantu sama sekali",
    "ban system tidak adil player jujur kena ban",
    "grafis downgrade di update terbaru mengecewakan",
    "loading masuk game sangat lama membosankan",
]

NEUTRAL_REVIEWS = [
    "game ini biasa saja tidak terlalu istimewa",
    "cukup menghibur untuk mengisi waktu luang",
    "ada kelebihan dan kekurangan masing masing",
    "game standar battle royale pada umumnya",
    "perlu beberapa perbaikan minor tapi lumayan",
    "tidak buruk tapi juga tidak luar biasa",
    "game ini oke untuk dimainkan sesekali saja",
    "performa di hp saya cukup stabil",
    "fitur game lengkap tapi bisa lebih baik lagi",
    "game ini sudah mature tapi butuh inovasi baru",
]


def check_health():
    """Check if the model server is running."""
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Server healthy | Model loaded: {data.get('model_loaded')}")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Server not reachable at", HEALTH_URL)
    return False


def send_prediction(texts):
    """Send a prediction request and return the response."""
    try:
        resp = requests.post(
            MODEL_URL,
            json={"inputs": texts},
            timeout=10
        )
        return resp.json(), resp.status_code
    except Exception as e:
        return {"error": str(e)}, 500


def generate_traffic(duration_seconds=120, requests_per_second=2):
    """Generate varied traffic for monitoring dashboards."""
    print(f"\n{'='*60}")
    print(f"  Traffic Generator - PUBG Sentiment Model")
    print(f"  Duration: {duration_seconds}s | Rate: ~{requests_per_second} req/s")
    print(f"{'='*60}\n")

    all_reviews = POSITIVE_REVIEWS + NEGATIVE_REVIEWS + NEUTRAL_REVIEWS
    total_requests = 0
    total_errors = 0
    start_time = time.time()

    while (time.time() - start_time) < duration_seconds:
        # Randomly select 1-5 texts per request
        batch_size = random.randint(1, 5)
        texts = random.sample(all_reviews, min(batch_size, len(all_reviews)))

        # Occasionally send bad requests to trigger errors (5% chance)
        if random.random() < 0.05:
            texts = []  # Empty input to trigger error

        result, status_code = send_prediction(texts)
        total_requests += 1

        if status_code == 200:
            preds = result.get('predictions', [])
            latency = result.get('latency_ms', 0)
            sentiments = [p['sentiment'] for p in preds]
            print(f"  [{total_requests:4d}] ✅ {len(preds)} predictions | "
                  f"latency: {latency:.1f}ms | sentiments: {sentiments}")
        else:
            total_errors += 1
            print(f"  [{total_requests:4d}] ❌ Error: {result.get('error', 'unknown')}")

        # Random delay between requests
        delay = 1.0 / requests_per_second + random.uniform(-0.2, 0.3)
        time.sleep(max(0.1, delay))

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"  Total requests: {total_requests}")
    print(f"  Errors: {total_errors}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Avg rate: {total_requests/elapsed:.1f} req/s")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Check server health first
    if not check_health():
        print("\nPlease start the model server first:")
        print("  python 3.prometheus_exporter.py")
        sys.exit(1)

    # Parse optional arguments
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 2

    generate_traffic(duration_seconds=duration, requests_per_second=rate)
