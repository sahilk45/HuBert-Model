# ============================================================
# HuBERT Emotion Recognition - Inference Script
# ============================================================

import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# ------------------ CHANGE THESE TWO PATHS ------------------

MODEL_PATH = "./hubert_large_bandpass"   # Folder where model is saved
AUDIO_PATH = "test_audio.wav"            # Any unseen audio file

# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

# Load model
model = HubertForSequenceClassification.from_pretrained(MODEL_PATH)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

# Load audio
y, _ = librosa.load(AUDIO_PATH, sr=16000)

# Extract features
inputs = feature_extractor(
    y,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

input_values = inputs["input_values"].to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(input_values=input_values)
    probabilities = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probabilities, dim=1).item()

emotion = IDX_TO_EMOTION[pred_id]
confidence = probabilities[0][pred_id].item()

print("\n===== Emotion Prediction =====")
print("Audio File       :", AUDIO_PATH)
print("Predicted Emotion:", emotion)
print("Confidence       :", round(confidence * 100, 2), "%")