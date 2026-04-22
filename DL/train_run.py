"""
train_run.py
=========================================
Versi FINAL cocok dengan:
- config.py
- preprocess.py
- dataset.py

Dataset: Cyberbullying Instagram
"""

import os
import torch

from config import (
    DEVICE,
    SAMPLE_SIZE,
    VOCAB_SIZE,
    MAX_LEN,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LR,
    LSTM_PATIENCE,
    BILSTM_MODEL_PATH,
    BILSTM_ATT_MODEL_PATH,
    VOCAB_PATH,
    PLOT_DIR,
)

from download_data import download_dataset
from preprocess import load_and_clean, show_cleaning_examples
from dataset import Vocabulary, get_dataloaders

from models import (
    BiLSTMClassifier,
    BiLSTMAttentionClassifier,
    count_parameters
)

from train import (
    set_seed,
    train_model,
    evaluate_lstm,
    get_criterion,
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report,
    compare_models
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==================================================
def main():

    set_seed(42)

    print("Device :", DEVICE)

    if SAMPLE_SIZE is None:
        print("Sample Size : Full Dataset")
    else:
        print(f"Sample Size : {SAMPLE_SIZE:,}")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ==================================================
    # 1. DOWNLOAD DATASET
    # ==================================================
    section("1. Download Dataset")

    csv_path = download_dataset()

    # ==================================================
    # 2. PREPROCESS
    # ==================================================
    section("2. Preprocessing")

    df = load_and_clean(csv_path, sample_size=SAMPLE_SIZE)

    show_cleaning_examples(df, n=3)

    label_counts = df["label"].value_counts().to_dict()

    # ==================================================
    # 3. PLOT DISTRIBUSI KELAS
    # ==================================================
    plt.figure(figsize=(8, 4))

    df["label"].value_counts().plot(
        kind="bar",
    )

    plt.title("Distribusi Kelas")
    plt.xlabel("Label")
    plt.ylabel("Jumlah")

    save_plot = os.path.join(PLOT_DIR, "distribusi_kelas.png")

    plt.tight_layout()
    plt.savefig(save_plot, dpi=150)
    plt.close()

    print("Plot disimpan:", save_plot)

    # ==================================================
    # 4. BUILD VOCAB
    # ==================================================
    section("3. Build Vocabulary")

    vocab = Vocabulary()

    vocab.build_vocab(
        df["cleaned_text"].tolist(),
        max_size=VOCAB_SIZE
    )

    vocab.save(VOCAB_PATH)

    # ==================================================
    # 5. DATALOADER
    # ==================================================
    section("4. Build DataLoader")

    train_loader, val_loader, test_loader = get_dataloaders(
        df=df,
        vocab=vocab,
        max_len=MAX_LEN,
        batch_size=LSTM_BATCH_SIZE
    )

    # ==================================================
    # 6. TRAIN BILSTM
    # ==================================================
    section("5. Training BiLSTM")

    bilstm = BiLSTMClassifier(
        vocab_size=len(vocab)
    )

    print("Total Parameter:", count_parameters(bilstm))

    hist_bilstm = train_model(
        model=bilstm,
        train_loader=train_loader,
        val_loader=val_loader,
        model_type="lstm",
        save_path=BILSTM_MODEL_PATH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
        device=DEVICE,
        label_counts=label_counts
    )

    plot_training_curves(
        hist_bilstm,
        "BiLSTM"
    )

    criterion = get_criterion(label_counts)

    _, _, preds1, labels1 = evaluate_lstm(
        bilstm,
        test_loader,
        criterion,
        DEVICE
    )

    plot_confusion_matrix(
        labels1,
        preds1,
        "BiLSTM"
    )

    result1 = print_classification_report(
        labels1,
        preds1,
        "BiLSTM"
    )

    result1["training_time_min"] = hist_bilstm["total_time"] / 60

    # ==================================================
    # 7. TRAIN ATTENTION
    # ==================================================
    section("6. Training BiLSTM + Attention")

    model_att = BiLSTMAttentionClassifier(
        vocab_size=len(vocab)
    )

    hist_att = train_model(
        model=model_att,
        train_loader=train_loader,
        val_loader=val_loader,
        model_type="lstm_att",
        save_path=BILSTM_ATT_MODEL_PATH,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE,
        device=DEVICE,
        label_counts=label_counts
    )

    plot_training_curves(
        hist_att,
        "BiLSTM + Attention"
    )

    _, _, preds2, labels2 = evaluate_lstm(
        model_att,
        test_loader,
        criterion,
        DEVICE
    )

    plot_confusion_matrix(
        labels2,
        preds2,
        "BiLSTM + Attention"
    )

    result2 = print_classification_report(
        labels2,
        preds2,
        "BiLSTM + Attention"
    )

    result2["training_time_min"] = hist_att["total_time"] / 60

    # ==================================================
    # 8. KOMPARASI
    # ==================================================
    section("7. Perbandingan Model")

    all_results = {
        "BiLSTM": result1,
        "BiLSTM+Attention": result2
    }

    compare_models(all_results)

    print("\nRingkasan Akhir")
    print("-" * 60)

    for name, m in all_results.items():
        print(
            f"{name:<20}"
            f"{m['accuracy']:>10.4f}"
            f"{m['f1_macro']:>12.4f}"
            f"{m['f1_weighted']:>14.4f}"
        )

    print("\nTraining selesai.")
    print("Model tersimpan di folder models/")


# ==================================================
if __name__ == "__main__":
    main()