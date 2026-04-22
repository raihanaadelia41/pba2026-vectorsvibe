"""
dataset.py — Vocabulary, PyTorch Dataset, dan DataLoader Builder
================================================================
Versi khusus klasifikasi Cyberbullying Instagram menggunakan BiLSTM

Isi file:
- Vocabulary
- InstagramDataset
- get_dataloaders()
"""

import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    VOCAB_SIZE,
    MAX_LEN,
    LSTM_BATCH_SIZE,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
    VOCAB_PATH,
)


# ──────────────────────────────────────────────
# 📖 VOCABULARY
# ──────────────────────────────────────────────
class Vocabulary:
    """
    Vocabulary untuk model BiLSTM

    Token khusus:
    <PAD> = 0
    <UNK> = 1
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }

        self.idx2word = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

    def build_vocab(self, texts, max_size=VOCAB_SIZE):
        counter = Counter()

        for text in texts:
            counter.update(str(text).split())

        most_common = counter.most_common(max_size - 2)

        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary dibuat: {len(self.word2idx):,} kata")

    def text_to_indices(self, text, max_len=MAX_LEN):
        tokens = str(text).split()[:max_len]

        indices = [
            self.word2idx.get(token, self.UNK_IDX)
            for token in tokens
        ]

        # padding
        indices += [self.PAD_IDX] * (max_len - len(indices))

        return indices

    def __len__(self):
        return len(self.word2idx)

    def save(self, path=VOCAB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

        print(f"Vocabulary disimpan ke {path}")

    @classmethod
    def load(cls, path=VOCAB_PATH):
        vocab = cls()

        with open(path, "r", encoding="utf-8") as f:
            vocab.word2idx = json.load(f)

        vocab.idx2word = {
            idx: word for word, idx in vocab.word2idx.items()
        }

        print(f"Vocabulary dimuat: {len(vocab):,} kata")

        return vocab


# ──────────────────────────────────────────────
# 📦 DATASET
# ──────────────────────────────────────────────
class InstagramDataset(Dataset):
    """
    Dataset PyTorch untuk komentar Instagram
    """

    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        indices = self.vocab.text_to_indices(text, self.max_len)

        length = min(len(str(text).split()), self.max_len)
        length = max(length, 1)

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


# ──────────────────────────────────────────────
# 🔄 DATALOADER BUILDER
# ──────────────────────────────────────────────
def get_dataloaders(
    df: pd.DataFrame,
    vocab: Vocabulary,
    max_len: int = MAX_LEN,
    batch_size: int = LSTM_BATCH_SIZE,
):
    """
    Membuat DataLoader:
    train / validation / test
    """

    texts = df["cleaned_text"].astype(str).tolist()
    labels = df["label_encoded"].tolist()

    # split train dan temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # split val dan test
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=(1 - relative_val),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    # dataset
    train_ds = InstagramDataset(train_texts, train_labels, vocab, max_len)
    val_ds = InstagramDataset(val_texts, val_labels, vocab, max_len)
    test_ds = InstagramDataset(test_texts, test_labels, vocab, max_len)

    # dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    print("Data berhasil dibagi:")
    print(f"Train : {len(train_ds)}")
    print(f"Val   : {len(val_ds)}")
    print(f"Test  : {len(test_ds)}")

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 🚀 TEST CEPAT
# ──────────────────────────────────────────────
if __name__ == "__main__":

    df = pd.read_csv("data/clean_dataset.csv")

    vocab = Vocabulary()
    vocab.build_vocab(df["Teks_Bersih"].astype(str).tolist())
    vocab.save()

    train_loader, _, _ = get_dataloaders(df, vocab)

    for batch in train_loader:
        x, y, l = batch

        print("Input Shape :", x.shape)
        print("Label Shape :", y.shape)
        print("Length Shape:", l.shape)

        break