import gradio as gr
import torch
import json
import torch.nn as nn

# =========================
# LOAD VOCAB
# =========================
with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 50

# =========================
# MODEL BILSTM
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(
            128,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out[:, -1, :]
        return self.fc(out)

# =========================
# LOAD MODEL
# =========================
model = BiLSTMClassifier(len(vocab))
model.load_state_dict(torch.load("models/bilstm_bullying.pt", map_location="cpu"))
model.eval()

# =========================
# PREPROCESS
# =========================
def text_to_tensor(text):
    text = text.lower().split()
    ids = [vocab.get(word, UNK_IDX) for word in text[:MAX_LEN]]
    ids += [PAD_IDX] * (MAX_LEN - len(ids))
    return torch.tensor([ids])

# =========================
# PREDICT
# =========================
def predict(text):
    x = text_to_tensor(text)

    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)[0]

    labels = ["Non-bullying", "Bullying"]

    return {
        labels[0]: float(prob[0]),
        labels[1]: float(prob[1]),
    }

# =========================
# UI
# =========================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar Instagram..."),
    outputs=gr.Label(num_top_classes=2),
    title="Cyberbullying Detector Indonesia",
    description="Prediksi komentar bullying atau non-bullying menggunakan BiLSTM."
)

demo.launch()