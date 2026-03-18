# NLP Project
## Klasifikasi Cyberbullying Bahasa Indonesia

### 📌 Deskripsi Project
Project ini bertujuan untuk melakukan klasifikasi teks dalam Bahasa Indonesia untuk mendeteksi apakah suatu kalimat termasuk cyberbullying atau tidak menggunakan pendekatan Natural Language Processing (NLP). Dalam tugas ini, akan dilakukan perbandingan performa beberapa algoritma machine learning untuk menentukan model terbaik.

---

## 👥 Anggota Kelompok

| No | Nama | NIM | Akun GitHub |
|----|------|-----|--------------------|
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

Repositori ini berisi implementasi sistem **deteksi *cyberbullying* berbahasa Indonesia** menggunakan pendekatan *machine learning* klasik berbasis teks. Dalam tugas ini membangun dan membandingkan tiga algoritma klasifikasi untuk menentukan model terbaik dalam mengenali konten perundungan siber pada teks berbahasa Indonesia.

Dataset yang digunakan telah dilengkapi dengan label biner:
- **Bullying** — Teks yang mengandung unsur perundungan siber
- **Non-Bullying** — Teks yang tidak mengandung unsur perundungan siber

---

## 🤖 Model yang Dikembangkan

### 1. Naive Bayes
Algoritma klasifikasi probabilistik yang memanfaatkan label biner pada dataset untuk mempelajari peluang suatu teks termasuk ke dalam kelas bullying atau non-bullying. Model ini menggunakan distribusi kata dan mengasumsikan setiap fitur bersifat independen.

### 2. Logistic Regression
Algoritma klasifikasi biner yang memanfaatkan label pada data untuk memodelkan probabilitas suatu teks termasuk ke dalam kelas bullying atau non-bullying. Model ini mempelajari bobot setiap fitur kata untuk membedakan kedua kelas.

### 3. Support Vector Machine (SVM)
Algoritma klasifikasi yang menggunakan data berlabel biner untuk menemukan batas pemisah (hyperplane) optimal antara kelas bullying dan non-bullying dalam ruang fitur, sehingga mampu memisahkan kedua kelas secara maksimal.

---

## ⚙️ Pipeline Pemrosesan

```
Raw Text
   ↓
Preprocessing (case folding, tokenisasi, stopword removal, stemming)
   ↓
Feature Extraction (TF-IDF)
   ↓
Training (Naive Bayes / Logistic Regression / SVM)
   ↓
Evaluasi & Perbandingan Model
```

**Metrik evaluasi:** Accuracy, Precision, Recall, dan F1-Score

---

## 📁 Struktur Folder

```
pba2026-vectorsvibe/
├── README.md
├── data/
├── notebooks/
├── src/
└── reports/
```

---

## 🗓️ Checkpoint

| Checkpoint | Tanggal | Target | Status |
|------------|---------|--------|--------|
| Checkpoint 1 | 18 Maret 2026 | Pemilihan dan pelaporan dataset | Done |

---

