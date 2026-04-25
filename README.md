# End-to-End Machine Learning: Analisis Sentimen PUBG Mobile

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-orange)

## Deskripsi Proyek

Proyek ini merupakan **Sistem Machine Learning End-to-End** untuk melakukan analisis sentimen terhadap ulasan game **PUBG Mobile** dari Google Play Store. Proyek ini dibangun untuk memenuhi kriteria tingkat **Advance (4 pts)** yang mencakup seluruh siklus hidup Machine Learning (MLOps).

### Kriteria yang Diselesaikan:
1. **Dataset & Preprocessing (Kriteria 1)**: Scraping >15,000 ulasan menggunakan `google_play_scraper`, pembersihan data teks, penghapusan *stopwords*, dan pelabelan otomatis.
2. **Membangun Model Machine Learning (Kriteria 2)**: Eksperimen model (Logistic Regression & TF-IDF) dan *hyperparameter tuning* dengan tracking komprehensif menggunakan **MLflow** (DagsHub terintegrasi).
3. **Continuous Integration (CI) Pipeline (Kriteria 3)**: Otomatisasi *training*, *tracking*, ekstraksi Run ID terbaik, *build* Docker Image, dan *push* ke Docker Hub menggunakan **GitHub Actions**.
4. **Monitoring & Logging (Kriteria 4)**: Sistem pemantauan performa model secara real-time menggunakan **Prometheus** (mengekspos 12 metriks kustom) dan **Grafana** (12 panel visualisasi dan 3 *alert rules* otomatis).

---

## Struktur Repositori

```text
Eksperimen_SML_Muhammad_Rizal_Nurfirdaus/
├── .github/workflows/              # GitHub Actions CI/CD pipelines
├── Membangun_model/                # Script training, tuning, & integrasi MLflow (DagsHub)
├── Monitoring dan Logging/         # Flask Exporter, Prometheus config, Grafana JSON, & Bukti
├── preprocessing/                  # Notebook eksperimen & script preprocessing otomatis
├── pubg_mobile_reviews_preprocessing/ # Dataset hasil pembersihan (Preprocessed)
├── pubg_mobile_reviews_raw/        # Dataset mentah (Raw CSV)
├── Workflow-CI/                    # Konfigurasi MLProject & CI workflow file
├── scraping_pubgmobile.py          # Script scraping data ulasan awal
├── requirements.txt                # Dependensi Python
└── README.md                       # Dokumentasi Utama
```

---

## 🚀 Panduan Instalasi (Git Clone)

Ikuti langkah-langkah di bawah ini untuk mengkloning repositori dan mengatur *environment* lokal Anda.

### Untuk Pengguna Windows

Buka **Command Prompt (CMD)** atau **PowerShell**, lalu jalankan perintah berikut secara berurutan:

```cmd
:: 1. Clone repositori
git clone https://github.com/MuhammadRizalNurfirdaus/Eksperimen_SML_Muhammad_Rizal_Nurfirdaus.git

:: 2. Masuk ke direktori proyek
cd Eksperimen_SML_Muhammad_Rizal_Nurfirdaus

:: 3. Buat Virtual Environment
python -m venv venv

:: 4. Aktivasi Virtual Environment
venv\Scripts\activate

:: 5. Install semua dependensi
pip install -r requirements.txt
```

### Untuk Pengguna Linux / macOS

Buka **Terminal**, lalu jalankan perintah berikut secara berurutan:

```bash
# 1. Clone repositori
git clone https://github.com/MuhammadRizalNurfirdaus/Eksperimen_SML_Muhammad_Rizal_Nurfirdaus.git

# 2. Masuk ke direktori proyek
cd Eksperimen_SML_Muhammad_Rizal_Nurfirdaus

# 3. Buat Virtual Environment
python3 -m venv venv

# 4. Aktivasi Virtual Environment
source venv/bin/activate

# 5. Install semua dependensi
pip install -r requirements.txt
```

---

## 🛠️ Cara Menjalankan Pipeline

### 1. Preprocessing Otomatis
Untuk membersihkan data mentah menjadi data siap *train*:
```bash
python preprocessing/automate_Muhammad_Rizal_Nurfirdaus.py
```

### 2. Training Model & MLflow Tracking
Jalankan MLflow UI di satu terminal:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Buka terminal baru (pastikan *venv* aktif), masuk ke folder `Membangun_model`, dan jalankan:
```bash
cd Membangun_model
python modelling.py
```

### 3. Monitoring (Prometheus & Grafana)
*Pastikan Prometheus dan Grafana sudah terinstal di sistem Anda.*
Jalankan *exporter* untuk melayani model dan mengekspos metriks:
```bash
cd "Monitoring dan Logging"
python 3.prometheus_exporter.py
```
*(Server akan berjalan di `http://localhost:8000`)*

Untuk mensimulasikan trafik *request* masuk agar Grafana menampilkan data:
```bash
python 7.inference.py
```

---

## 🐳 Docker (CI/CD Artifacts)

Jika Anda ingin langsung menjalankan model hasil pipeline GitHub Actions tanpa *setup* manual, Anda bisa menarik *image* dari Docker Hub:

```bash
# Pull image dari Docker Hub
docker pull mrizaln/pubg_sentiment_model:latest

# Jalankan container di port 8080
docker run -p 8080:8080 mrizaln/pubg_sentiment_model:latest
```

---

## Author

**Muhammad Rizal Nurfirdaus**
- Peserta Dicoding Sains Machine Learning
