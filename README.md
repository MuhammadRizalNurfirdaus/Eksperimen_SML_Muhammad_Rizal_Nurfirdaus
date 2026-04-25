# Eksperimen SML - Muhammad Rizal Nurfirdaus 021

## Deskripsi

Proyek eksperimen **Sains, Machine Learning (SML)** yang melakukan analisis sentimen terhadap review game **PUBG Mobile** dari Google Play Store. Dataset dikumpulkan menggunakan scraping `google_play_scraper` dan diproses melalui pipeline preprocessing untuk menghasilkan data yang siap digunakan untuk pelatihan model.

## Struktur Repository

```
Eksperimen_SML_Muhammad_Rizal_Nurfirdaus/
├── .github/workflows/
│   ├── preprocessing.yml              # GitHub Actions untuk automated preprocessing
│   └── mlflow_ci.yml                  # GitHub Actions untuk CI model training + Docker
├── pubg_mobile_reviews_raw/
│   └── pubg_mobile_reviews.csv         # Dataset mentah (raw)
├── preprocessing/
│   ├── Eksperimen_Muhammad_Rizal_Nurfirdaus.ipynb  # Notebook eksperimen
│   └── automate_Muhammad_Rizal_Nurfirdaus.py       # Script otomatis preprocessing
├── pubg_mobile_reviews_preprocessing/
│   └── pubg_mobile_reviews_preprocessed.csv        # Dataset hasil preprocessing
├── Membangun_model/
│   ├── modelling.py                    # Script training model (lokal)
│   ├── modelling_tuning.py             # Script training model dengan tuning
│   ├── requirements.txt                # Dependencies model
│   └── mlflow_artifacts/               # Artefak hasil training
├── Workflow-CI/
│   ├── .workflow/
│   │   └── mlflow_ci.yml               # Copy workflow CI
│   └── MLProject/
│       ├── modelling.py                # Script training untuk CI
│       ├── conda.yaml                  # Conda environment
│       ├── MLproject                   # MLflow project config
│       ├── pubg_mobile_reviews_preprocessed.csv  # Dataset preprocessing
│       └── Docker_Hub_Link.txt         # Tautan ke Docker Hub
├── scraping_pubgmobile.py              # Script scraping data
├── requirements.txt                    # Dependencies
└── README.md
```

## Dataset

- **Sumber**: Google Play Store (review PUBG Mobile)
- **Jumlah**: ~15000 review
- **Kolom**: `review`, `rating`, `date`, `userName`
- **Bahasa**: Indonesia

## Tahapan Preprocessing

1. **Data Loading** — Memuat dataset CSV
2. **Data Cleaning** — Lowercase, hapus URL, emoji, special chars, angka
3. **Stopword Removal** — Menghapus stopwords Bahasa Indonesia (NLTK + custom)
4. **Sentiment Labeling** — Rating 1-2: negatif, 3: netral, 4-5: positif
5. **Export** — Menyimpan hasil ke CSV

## Workflow CI (MLflow Project)

### Cara Kerja

Pipeline CI otomatis berjalan ketika ada perubahan pada `Workflow-CI/MLProject/**` atau trigger manual (`workflow_dispatch`).

**Tahapan Pipeline:**
1. **Setup** — Checkout repo, install Python 3.11 dan dependencies
2. **MLflow Server** — Jalankan MLflow tracking server di port 5000
3. **Train Model** — Jalankan `mlflow run .` untuk melatih model Logistic Regression
4. **Extract Run ID** — Baca Run ID dari `run_id.txt` yang disimpan oleh `modelling.py`
5. **Upload Artifacts** — Simpan mlruns dan artefak ke GitHub Actions Artifacts
6. **Build Docker** — Bangun Docker image menggunakan `mlflow models build-docker`
7. **Push to Docker Hub** — Push image ke Docker Hub

### GitHub Secrets yang Diperlukan

| Secret | Keterangan |
|--------|-----------|
| `DOCKER_USERNAME` | Username Docker Hub |
| `DOCKER_PASSWORD` | Access token Docker Hub |

### Docker Hub

Docker Image: `<DOCKER_USERNAME>/pubg_sentiment_model:latest`

```bash
# Pull image
docker pull <DOCKER_USERNAME>/pubg_sentiment_model:latest

# Run container
docker run -p 8080:8080 <DOCKER_USERNAME>/pubg_sentiment_model:latest
```

## Cara Menjalankan

### Prerequisites

```bash
pip install -r requirements.txt
```

### Jalankan Preprocessing Otomatis

```bash
python preprocessing/automate_Muhammad_Rizal_Nurfirdaus.py
```

### Jalankan Model Training (Lokal)

```bash
cd Membangun_model
mlflow server --host 127.0.0.1 --port 5000 &
python modelling.py
```

### Notebook Eksperimen

Buka `preprocessing/Eksperimen_Muhammad_Rizal_Nurfirdaus.ipynb` di Jupyter Notebook atau Google Colab.

## GitHub Actions

### 1. Preprocessing Workflow
Workflow otomatis akan berjalan ketika:
- Ada push ke branch `main` yang mengubah dataset raw atau script automate
- Trigger manual via `workflow_dispatch`

### 2. MLflow CI Pipeline
Workflow otomatis akan berjalan ketika:
- Ada push yang mengubah file di `Workflow-CI/MLProject/`
- Trigger manual via `workflow_dispatch`

Pipeline akan melatih model, menyimpan artefak, dan membuat Docker image.

## Author

**Muhammad Rizal Nurfirdaus**
