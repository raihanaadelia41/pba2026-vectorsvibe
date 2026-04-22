"""
download_data.py
=====================================
Versi Windows Friendly (tanpa emoji)
Auto Download Dataset dari Kaggle
"""

import os
import shutil
import glob

from config import DATA_DIR, RAW_CSV


KAGGLE_DATASET = "cttrhnn/cyberbullying-bahasa-indonesia"
EXPECTED_FILENAME = "dataset.csv"


# ==================================================
def download_dataset():
    """
    Download dataset otomatis
    """

    # jika file sudah ada
    if os.path.exists(RAW_CSV):
        print("Dataset sudah tersedia:")
        print(RAW_CSV)
        return RAW_CSV

    os.makedirs(DATA_DIR, exist_ok=True)

    # ==================================================
    # Coba pakai kagglehub
    # ==================================================
    try:
        print("Download dataset dari Kaggle...")

        import kagglehub

        download_path = kagglehub.dataset_download(
            KAGGLE_DATASET
        )

        print("Dataset berhasil didownload ke:")
        print(download_path)

        # cari file dataset.csv
        csv_files = glob.glob(
            os.path.join(
                download_path,
                "**",
                EXPECTED_FILENAME
            ),
            recursive=True
        )

        # kalau tidak ada, cari csv lain
        if not csv_files:
            csv_files = glob.glob(
                os.path.join(
                    download_path,
                    "**",
                    "*.csv"
                ),
                recursive=True
            )

        if len(csv_files) == 0:
            raise FileNotFoundError(
                "File CSV tidak ditemukan"
            )

        src = csv_files[0]

        shutil.copy2(src, RAW_CSV)

        print("Dataset berhasil disimpan:")
        print(RAW_CSV)

        return RAW_CSV

    except Exception as e:

        print("Kagglehub gagal:")
        print(e)

        print("Mencoba fallback opendatasets...")

        return fallback_download()


# ==================================================
def fallback_download():

    try:
        import opendatasets as od

        url = (
            "https://www.kaggle.com/datasets/"
            "cttrhnn/cyberbullying-bahasa-indonesia"
        )

        od.download(
            url,
            data_dir=DATA_DIR
        )

        csv_files = glob.glob(
            os.path.join(
                DATA_DIR,
                "**",
                "*.csv"
            ),
            recursive=True
        )

        if len(csv_files) == 0:
            raise FileNotFoundError(
                "CSV tidak ditemukan"
            )

        src = csv_files[0]

        if os.path.abspath(src) != os.path.abspath(RAW_CSV):
            shutil.copy2(src, RAW_CSV)

        print("Dataset tersedia di:")
        print(RAW_CSV)

        return RAW_CSV

    except Exception as e:

        raise RuntimeError(
            "Gagal download dataset.\n\n"
            + str(e)
            + "\n\nDownload manual dari:\n"
            + "https://www.kaggle.com/datasets/"
            + "cttrhnn/cyberbullying-bahasa-indonesia"
        )


# ==================================================
if __name__ == "__main__":

    path = download_dataset()

    print("\nDataset siap dipakai:")
    print(path)