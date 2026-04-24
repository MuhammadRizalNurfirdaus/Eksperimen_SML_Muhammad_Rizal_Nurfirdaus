"""
automate_Muhammad_Rizal_Nurfirdaus.py

Script otomatis untuk melakukan preprocessing data review PUBG Mobile
dari Google Play Store. Script ini mengotomasi seluruh tahapan preprocessing
yang dilakukan di notebook Eksperimen, sehingga menghasilkan dataset
yang siap dilatih untuk model sentiment analysis.

Tahapan:
1. Load raw dataset
2. Cleaning teks (lowercase, hapus special chars, hapus angka)
3. Remove stopwords Bahasa Indonesia
4. Labeling sentimen berdasarkan rating
5. Export hasil ke folder output
"""

import os
import re
import pandas as pd
import numpy as np
import nltk

# Download NLTK data jika belum ada
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords


# ============================================================
# Konfigurasi Path
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "pubg_mobile_reviews_raw", "pubg_mobile_reviews.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "pubg_mobile_reviews_preprocessing")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pubg_mobile_reviews_preprocessed.csv")


# ============================================================
# Custom Stopwords Bahasa Indonesia
# ============================================================
CUSTOM_STOPWORDS_ID = {
    'yang', 'di', 'dan', 'ini', 'itu', 'dengan', 'untuk', 'tidak', 'ada',
    'pada', 'ke', 'dari', 'adalah', 'juga', 'akan', 'sudah', 'bisa', 'saya',
    'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka', 'nya', 'lah', 'lagi',
    'kan', 'ya', 'dong', 'sih', 'nih', 'deh', 'aja', 'gak', 'ga', 'gk',
    'nggak', 'ngga', 'udah', 'udh', 'yg', 'dg', 'dgn', 'tp', 'tapi',
    'jadi', 'jd', 'krn', 'karena', 'karna', 'dlm', 'dalam', 'sm', 'sama',
    'mau', 'mo', 'lg', 'sdh', 'blm', 'belum', 'kalau', 'kalo', 'klo',
    'bgt', 'banget', 'jgn', 'jangan', 'hrs', 'harus', 'bs', 'gw', 'gue',
    'lo', 'lu', 'emg', 'emang', 'mmg', 'memang', 'krna', 'trs', 'terus',
    'dr', 'ke', 'se', 'si', 'sang', 'para', 'pun', 'pula', 'lalu',
    'kemudian', 'oleh', 'dulu', 'masih', 'atau', 'serta', 'maupun',
    'melainkan', 'tetapi', 'namun', 'sedangkan', 'sementara', 'meskipun',
    'walaupun', 'apabila', 'jika', 'bila', 'ketika', 'saat', 'sambil',
    'seraya', 'supaya', 'agar', 'biarpun', 'walau', 'sekalipun',
    'seandainya', 'andai', 'seolah', 'bagaikan', 'seperti', 'bak',
    'ibarat', 'daripada', 'alih', 'sebab', 'hal', 'sering', 'selalu',
    'setiap', 'tiap', 'per', 'dll', 'dsb', 'dst', 'tsb', 'dlsb', 'oh',
    'ah', 'eh', 'wah', 'nah', 'hm', 'hmm', 'lho', 'kok', 'tuh', 'nih',
    'tau', 'tahu', 'mana', 'apa', 'siapa', 'bagaimana', 'gimana',
    'kapan', 'dimana', 'kemana', 'berapa', 'kenapa', 'mengapa'
}


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV.

    Args:
        path: Path ke file CSV

    Returns:
        DataFrame berisi data review
    """
    print(f"[1/5] Loading data dari: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    return df


def clean_text(text: str) -> str:
    """
    Membersihkan teks review:
    - Konversi ke lowercase
    - Hapus URL
    - Hapus mention dan hashtag
    - Hapus emoji dan special characters
    - Hapus angka
    - Hapus whitespace berlebih

    Args:
        text: Teks review mentah

    Returns:
        Teks yang sudah dibersihkan
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Hapus URL
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Hapus mention dan hashtag
    text = re.sub(r'@\w+|#\w+', '', text)

    # Hapus emoji dan karakter non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus special characters, hanya simpan huruf dan spasi
    text = re.sub(r'[^a-z\s]', '', text)

    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Menghapus stopwords dari teks menggunakan gabungan
    NLTK stopwords Indonesia dan custom stopwords.

    Args:
        text: Teks yang sudah di-clean

    Returns:
        Teks tanpa stopwords
    """
    if not isinstance(text, str) or text == "":
        return ""

    # Gabungkan NLTK stopwords dengan custom
    try:
        nltk_stopwords = set(stopwords.words('indonesian'))
    except OSError:
        nltk_stopwords = set()

    all_stopwords = nltk_stopwords | CUSTOM_STOPWORDS_ID

    words = text.split()
    filtered = [w for w in words if w not in all_stopwords and len(w) > 1]

    return ' '.join(filtered)


def label_sentiment(rating: int) -> str:
    """
    Melabeli sentimen berdasarkan rating:
    - Rating 1-2: negatif
    - Rating 3: netral
    - Rating 4-5: positif

    Args:
        rating: Rating 1-5

    Returns:
        Label sentimen (positif/netral/negatif)
    """
    if rating <= 2:
        return 'negatif'
    elif rating == 3:
        return 'netral'
    else:
        return 'positif'


def preprocess_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline lengkap preprocessing data review PUBG Mobile.

    Tahapan:
    1. Load data
    2. Hapus duplikat dan missing values
    3. Clean teks review
    4. Remove stopwords
    5. Label sentimen
    6. Simpan ke CSV

    Args:
        input_path: Path ke file CSV raw
        output_path: Path untuk menyimpan hasil preprocessing

    Returns:
        DataFrame hasil preprocessing
    """
    # Step 1: Load data
    df = load_data(input_path)

    # Step 2: Hapus duplikat dan missing values
    print("[2/5] Membersihkan duplikat dan missing values...")
    initial_rows = len(df)
    df = df.dropna(subset=['review'])
    df = df.drop_duplicates(subset=['review'])
    removed = initial_rows - len(df)
    print(f"      Dihapus: {removed} baris (duplikat/missing)")
    print(f"      Sisa: {len(df)} baris")

    # Step 3: Cleaning teks
    print("[3/5] Cleaning teks review...")
    df['review_clean'] = df['review'].apply(clean_text)

    # Step 4: Remove stopwords
    print("[4/5] Menghapus stopwords...")
    df['review_processed'] = df['review_clean'].apply(remove_stopwords)

    # Hapus baris dengan review kosong setelah preprocessing
    df = df[df['review_processed'].str.strip().str.len() > 0]
    print(f"      Baris valid setelah preprocessing: {len(df)}")

    # Step 5: Labeling sentimen
    print("[5/5] Labeling sentimen...")
    df['sentiment'] = df['rating'].apply(label_sentiment)

    # Buat DataFrame output
    df_output = df[['review', 'review_clean', 'review_processed', 'rating', 'sentiment', 'date', 'userName']].copy()
    df_output = df_output.reset_index(drop=True)

    # Simpan ke CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False)
    print(f"\n✅ Preprocessing selesai!")
    print(f"   Output: {output_path}")
    print(f"   Total data: {len(df_output)} baris")
    print(f"\n   Distribusi Sentimen:")
    print(df_output['sentiment'].value_counts().to_string())

    return df_output


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  AUTOMATE PREPROCESSING - PUBG Mobile Reviews")
    print("  Muhammad Rizal Nurfirdaus")
    print("=" * 60)
    print()

    result = preprocess_pipeline(RAW_DATA_PATH, OUTPUT_PATH)

    print()
    print("=" * 60)
    print("  DONE!")
    print("=" * 60)
