"""
download_data.py
=====================================
Versi Lokal (tanpa download Kaggle)
Langsung memakai dataset yang sudah ada
"""

import os

from config import DATA_DIR, RAW_CSV


# ==================================================
# Path dataset lokal
# ==================================================
LOCAL_DATASET = r"C:\Users\raiha\Downloads\pba2026-vectorsvibe-main\pba2026-vectorsvibe-main\data\dataset_final_2 (1).csv"


# ==================================================
def download_dataset():
    """
    Tidak download lagi.
    Langsung pakai dataset lokal.
    """

    print("Menggunakan dataset lokal...")

    if os.path.exists(LOCAL_DATASET):
        print("Dataset ditemukan:")
        print(LOCAL_DATASET)
        return LOCAL_DATASET

    raise FileNotFoundError(
        "Dataset tidak ditemukan di path berikut:\n"
        + LOCAL_DATASET
    )


# ==================================================
if __name__ == "__main__":

    path = download_dataset()

    print("\nDataset siap dipakai:")
    print(path)