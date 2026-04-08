import gradio as gr
import pandas as pd
import joblib
import os
from pathlib import Path

# Tentukan path model relatif terhadap script ini
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR.parent / "nlp_pipeline_final.pkl"

print(f"Script directory: {SCRIPT_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")

# Try load model with pycaret fallback ke joblib
model = None
try:
    from pycaret.classification import load_model, predict_model
    # PyCaret: load tanpa .pkl extension
    model = load_model(str(SCRIPT_DIR.parent / "nlp_pipeline_final"))
    print("Model loaded successfully with PyCaret")
except Exception as e:
    print(f"PyCaret load failed: {e}, trying joblib...")
    try:
        if MODEL_PATH.exists():
            model = joblib.load(str(MODEL_PATH))
            print("Model loaded successfully with joblib")
        else:
            print(f"Model file not found at: {MODEL_PATH}")
    except Exception as e2:
        print(f"Joblib load also failed: {e2}")

def predict_sentiment(text):
    """Prediksi kategori komentar dari teks input."""
    if model is None:
        return "Model tidak berhasil di-load. Pastikan file model ada."
    
    try:
        from pycaret.classification import predict_model
        data = pd.DataFrame([text], columns=["cleaned_text"])
        prediction = predict_model(model, data=data)
        return prediction["prediction_label"][0]
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Prediksi Kategori Komentar Instagram",
    description="Masukkan komentar yang ingin diprediksi kategorinya"
)

interface.launch(share=True)