from pathlib import Path

import numpy as np
import torch

from src.model import simpleECGNN
from src.preprocessing import preprocess_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "results/models/best_simple_cnn.pt"
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path=MODEL_PATH):
    model = simpleECGNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()


def predict(signal_arr, fs, threshold=THRESHOLD):
    signal = preprocess_signal(np.asarray(signal_arr), fs)
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        sttc_probability = torch.sigmoid(logits).item()

    prediction = "STTC" if sttc_probability >= threshold else "Normal"
    return {
        "prediction": prediction,
        "sttc_probability": round(sttc_probability, 4),
    }
