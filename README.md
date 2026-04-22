# NLP Project
## Klasifikasi Cyberbullying Bahasa Indonesia

### 📌 Deskripsi Project
Project ini bertujuan untuk melakukan klasifikasi teks dalam Bahasa Indonesia untuk mendeteksi apakah suatu kalimat termasuk cyberbullying atau tidak menggunakan pendekatan Natural Language Processing (NLP). Dalam tugas ini, akan dilakukan perbandingan performa algoritma machine learning dan deep learning untuk menentukan model terbaik.

---

## 👥 Anggota Kelompok

| No | Nama | NIM | Akun GitHub |
|----|------|-----|-------------|
| 1 | Aisyah Musfirah | 123450084 | aisyahmusfirah |
| 2 | Anggi Puspita Ningrum | 123450012 | AnggiPuspita012 |
| 3 | Raihana Adelia Putri | 123450041 | raihanaadelia41 |

---

## 📊 Dataset

**Nama Dataset:** Cyberbullying Bahasa Indonesia

**Sumber:** [Kaggle — Cyberbullying Bahasa Indonesia](https://www.kaggle.com/datasets/cttrhnn/cyberbullying-bahasa-indonesia)

Dataset ini berisi kumpulan teks berbahasa Indonesia yang diklasifikasikan berdasarkan kandungan *cyberbullying*-nya.

---

## 🎯 Deskripsi Repositori

Repositori ini berisi implementasi sistem **deteksi *cyberbullying* berbahasa Indonesia** menggunakan pendekatan NLP berbasis machine learning dan deep learning. Proyek ini membangun dan membandingkan beberapa algoritma untuk menentukan model terbaik dalam mengenali konten perundungan siber.

Dataset yang digunakan telah dilengkapi dengan label biner:
- **Bullying**
- **Non-Bullying**

---

## 🤖 Model yang Dikembangkan

### 1. Naive Bayes
Model probabilistik yang memanfaatkan label biner untuk menghitung peluang teks termasuk kelas *bullying* atau *non-bullying* berdasarkan distribusi kata.

### 2. Logistic Regression
Model klasifikasi biner yang mempelajari bobot fitur untuk memprediksi probabilitas teks termasuk *bullying* atau *non-bullying*.

### 3. Support Vector Machine (SVM)
Model yang mencari batas pemisah optimal untuk membedakan teks *bullying* dan *non-bullying* secara maksimal.

### 4. Bi-LSTM (Bidirectional LSTM)
Model deep learning yang membaca teks dari dua arah (forward dan backward) sehingga mampu memahami konteks kalimat dengan lebih baik dalam mengklasifikasikan *bullying* dan *non-bullying*.

---

## ⚙️ Pipeline Pemrosesan

```
Raw Data
    ↓
Preprocessing
(case folding, cleaning, tokenisasi, stopword removal, stemming)
    ↓
Feature Extraction:
    - TF-IDF            → untuk Naive Bayes, Logistic Regression, & SVM
    - Embedding Layer   → untuk Bi-LSTM
    ↓
Training Model:
    - Naive Bayes
    - Logistic Regression
    - SVM
    - Bi-LSTM
    ↓
Evaluasi & Perbandingan Model
```

**Metrik evaluasi:** Accuracy, Precision, Recall, dan F1-Score

### Deploy Hugging Face [ML Models]
Link : https://huggingface.co/spaces/aishsahi/prediksi-komen-lagi

### Deploy Hugging Fase [DL Models]
Link : https://huggingface.co/spaces/raihana41/cyberbully-app

---

## 📁 Struktur Folder

```
pba2026-vectorsvibe/
├── README.md
├── data/
│   ├── raw/            (dataset asli dari Kaggle)
│   └── processed/      (data setelah preprocessing)
├── notebooks/          (file .ipynb Jupyter Notebook)
├── src/                (kode Python (.py))
├── models/             (model yang sudah dilatih)
└── reports/            (laporan dan grafik hasil)
```

---

## 🗓️ Checkpoint

| Checkpoint | Tanggal | Target | Status |
|------------|---------|--------|--------|
| Checkpoint 1 | 18 Maret 2026 | Pemilihan dan pelaporan dataset | Done |

---
