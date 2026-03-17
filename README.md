

## 👥 Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1 | Aisyah Musfirah | 123450084 |
| 2 | Anggi Puspita Ningrum | 123450012 |
| 3 | Raihana Adelia Putri | 123450041 |

---

## 📊 Dataset

**Nama Dataset:** Cyberbullying Bahasa Indonesia

**Sumber:** [Kaggle — Cyberbullying Bahasa Indonesia](https://www.kaggle.com/datasets/cttrhnn/cyberbullying-bahasa-indonesia)

Dataset ini berisi kumpulan teks berbahasa Indonesia yang diklasifikasikan berdasarkan kandungan *cyberbullying*-nya.

---

## 🎯 Deskripsi Repositori

Repositori ini berisi implementasi sistem **deteksi *cyberbullying* berbahasa Indonesia** menggunakan pendekatan *machine learning* klasik berbasis teks. Proyek ini membangun dan membandingkan tiga algoritma klasifikasi untuk menentukan model terbaik dalam mengenali konten perundungan siber pada teks berbahasa Indonesia.

Dataset yang digunakan telah dilengkapi dengan label biner:
- **Bullying** — Teks yang mengandung unsur perundungan siber
- **Non-Bullying** — Teks yang tidak mengandung unsur perundungan siber

---

## 🤖 Model yang Dikembangkan

### 1. Naive Bayes
Algoritma probabilistik berbasis Teorema Bayes yang menghitung peluang suatu teks termasuk kelas *Bullying* atau *Non-Bullying* berdasarkan kemunculan kata-kata di dalamnya. Model ini mengasumsikan setiap fitur (kata) bersifat independen satu sama lain. Naive Bayes dipilih karena ringan, cepat dilatih, dan terbukti efektif untuk klasifikasi teks.

### 2. Logistic Regression
Algoritma klasifikasi linear yang memodelkan probabilitas kelas menggunakan fungsi sigmoid. Model ini mempelajari bobot setiap fitur kata untuk membedakan teks *bullying* dari yang bukan. Logistic Regression dipilih karena mudah diinterpretasikan, stabil, dan memberikan performa yang baik pada data teks berdimensi tinggi.

### 3. Support Vector Machine (SVM)
Algoritma yang bekerja dengan mencari *hyperplane* optimal untuk memisahkan dua kelas secara maksimal. Pada data teks yang direpresentasikan sebagai vektor TF-IDF berdimensi tinggi, SVM dikenal unggul karena mampu menemukan batas keputusan yang optimal meskipun ruang fiturnya sangat besar.

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
| Checkpoint 1 | 18 Maret 2026 | Pemilihan dan pelaporan dataset | ✅ |

---

