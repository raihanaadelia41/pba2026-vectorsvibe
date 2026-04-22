"""
config.py
====================================================
Konfigurasi utama project klasifikasi cyberbullying
Dataset berada di folder:
task_2/data/dataset_final_2.csv
"""

import os
import torch

# ==================================================
# PATH FOLDER
# ==================================================

# folder DL/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# folder task_2/
ROOT_DIR = os.path.dirname(BASE_DIR)

# folder data sejajar dengan DL
DATA_DIR = os.path.join(ROOT_DIR, "data")

# folder output di dalam DL
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

# buat folder otomatis
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==================================================
# FILE DATASET
# ==================================================

RAW_CSV = os.path.join(DATA_DIR, "dataset_final_2.csv")

CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_dataset.csv")

# ==================================================
# DATASET COLUMN
# ==================================================

TEXT_COL = "Teks_Bersih"
LABEL_COL = "Kategori"

LABEL_LIST = [
    "Non-bullying",
    "Bullying"
]

NUM_CLASSES = len(LABEL_LIST)

# ==================================================
# GENERAL
# ==================================================

RANDOM_SEED = 42

# None = pakai semua data
SAMPLE_SIZE = None

TEST_SIZE = 0.10
VAL_SIZE = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# BiLSTM PARAMETER
# ==================================================

VOCAB_SIZE = 10000
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.30
MAX_LEN = 50

LSTM_EPOCHS = 15
LSTM_BATCH_SIZE = 32
LSTM_LR = 1e-3
LSTM_PATIENCE = 3

# ==================================================
# DISTILBERT PARAMETER
# ==================================================

BERT_MODEL = "distilbert-base-uncased"
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5
BERT_EPOCHS = 5
BERT_PATIENCE = 2

# ==================================================
# MODEL SAVE PATH
# ==================================================

BILSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bilstm_bullying.pt"
)

BILSTM_ATT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bilstm_attention_bullying.pt"
)

DISTILBERT_MODEL_DIR = os.path.join(
    MODEL_DIR,
    "distilbert"
)

VOCAB_PATH = os.path.join(
    MODEL_DIR,
    "vocab.json"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.json"
)

# ==================================================
# VISUALIZATION FILE
# ==================================================

LOSS_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "loss_curve.png"
)

CONF_MATRIX_PATH = os.path.join(
    PLOT_DIR,
    "confusion_matrix.png"
)

REPORT_PATH = os.path.join(
    PLOT_DIR,
    "classification_report.txt"
)