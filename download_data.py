"""
download_data.py — Load Dataset dari Local File
================================================
Membaca dataset "dataset_final_2.csv" dari folder data/.
"""

import os

from config import DATA_DIR

# ──────────────────────────────────────────────
# Konstanta Dataset
# ──────────────────────────────────────────────
DATASET_FILE = os.path.join(DATA_DIR, "dataset_final_2.csv")


def download_dataset() -> str:
    """
    Load dataset dari file lokal (dataset_final_2.csv).

    Alur:
    1. Cek apakah file dataset_final_2.csv ada di folder data/.
    2. Return path absolut ke file CSV.

    Returns:
        str: Path absolut ke file CSV yang siap dipakai.

    Raises:
        FileNotFoundError: Jika dataset_final_2.csv tidak ditemukan.
    """

    if os.path.exists(DATASET_FILE):
        print(f"[OK] Dataset ditemukan: {DATASET_FILE}")
        return DATASET_FILE
    else:
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {DATASET_FILE}"
        )


# ──────────────────────────────────────────────
# Jika dijalankan langsung: python download_data.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    path = download_dataset()
    print(f"\n[SUCCESS] Dataset siap digunakan: {path}")
