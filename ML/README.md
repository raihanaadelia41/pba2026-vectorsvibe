---
title: Prediksi Kategori Komentar
description: Model NLP untuk prediksi kategori komentar Instagram menggunakan PyCaret
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Prediksi Kategori Komentar Instagram

Model machine learning untuk prediksi kategori komentar Instagram berdasarkan konten teks.

## Dataset
- **Sumber**: Dataset Instagram Comments
- **Ukuran**: 650+ baris
- **Kategori**: Berbagai kategori komentar
- **Preprocessing**: Text cleaning, normalisasi leetspeak, ekspansi slang

## Model
- **Algoritma**: Logistic Regression (dari PyCaret AutoML)
- **Framework**: PyCaret Classification
- **Preprocessing**: TF-IDF untuk feature extraction

## Cara Penggunaan
1. Masukkan teks komentar pada kolom input
2. Klik "Submit" 
3. Model akan memprediksi kategori komentar

## Teknologi
- **Gradio**: UI/Interface
- **PyCaret**: AutoML Framework
- **Scikit-learn**: Machine Learning
- **NLTK & Sastrawi**: NLP Indonesia

---
Author: NLP Workshop
