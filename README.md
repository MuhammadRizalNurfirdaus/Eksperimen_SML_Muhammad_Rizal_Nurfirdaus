# Eksperimen SML - Muhammad Rizal Nurfirdaus

## Deskripsi

Proyek eksperimen **Sains, Machine Learning (SML)** yang melakukan analisis sentimen terhadap review game **PUBG Mobile** dari Google Play Store. Dataset dikumpulkan menggunakan scraping `google_play_scraper` dan diproses melalui pipeline preprocessing untuk menghasilkan data yang siap digunakan untuk pelatihan model.

## Struktur Repository

```
Eksperimen_SML_Muhammad_Rizal_Nurfirdaus/
├── .github/workflows/
│   └── preprocessing.yml              # GitHub Actions untuk automated preprocessing
├── pubg_mobile_reviews_raw/
│   └── pubg_mobile_reviews.csv         # Dataset mentah (raw)
├── preprocessing/
│   ├── Eksperimen_Muhammad_Rizal_Nurfirdaus.ipynb  # Notebook eksperimen
│   └── automate_Muhammad_Rizal_Nurfirdaus.py       # Script otomatis preprocessing
├── pubg_mobile_reviews_preprocessing/
│   └── pubg_mobile_reviews_preprocessed.csv        # Dataset hasil preprocessing
├── scraping_pubgmobile.py              # Script scraping data
├── requirements.txt                    # Dependencies
└── README.md
```

## Dataset

- **Sumber**: Google Play Store (review PUBG Mobile)
- **Jumlah**: ~1000 review
- **Kolom**: `review`, `rating`, `date`, `userName`
- **Bahasa**: Indonesia

## Tahapan Preprocessing

1. **Data Loading** — Memuat dataset CSV
2. **Data Cleaning** — Lowercase, hapus URL, emoji, special chars, angka
3. **Stopword Removal** — Menghapus stopwords Bahasa Indonesia (NLTK + custom)
4. **Sentiment Labeling** — Rating 1-2: negatif, 3: netral, 4-5: positif
5. **Export** — Menyimpan hasil ke CSV

## Cara Menjalankan

### Prerequisites

```bash
pip install -r requirements.txt
```

### Jalankan Preprocessing Otomatis

```bash
python preprocessing/automate_Muhammad_Rizal_Nurfirdaus.py
```

### Notebook Eksperimen

Buka `preprocessing/Eksperimen_Muhammad_Rizal_Nurfirdaus.ipynb` di Jupyter Notebook atau Google Colab.

## GitHub Actions

Workflow otomatis akan berjalan ketika:
- Ada push ke branch `main` yang mengubah dataset raw atau script automate
- Trigger manual via `workflow_dispatch`

Workflow akan menjalankan preprocessing dan meng-commit hasil ke repository.

## Author

**Muhammad Rizal Nurfirdaus**
