
# 🎙️ Speech Emotion Recognition using HuBERT, Wav2Vec2 & Bi-LSTM


## 📌 Project Overview

This repository presents a comprehensive study on **Speech Emotion Recognition (SER)** using three different deep learning approaches:

1. **HuBERT (Large)**
2. **Wav2Vec2 (Base) – Fine-tuned**
3. **Bi-LSTM with Augmented Audio**

The goal is to compare self-supervised transformer-based models and traditional recurrent neural networks for multi-class emotion classification on augmented speech datasets.

---


# 🧠 1️⃣ HuBERT Large Model

### 🔹 Model
- Pretrained HuBERT Large
- Transformer-based encoder
- Self-supervised speech representation learning
- Fine-tuned for emotion classification

### 🔹 Key Characteristics
- Contextual speech embeddings
- Masked prediction objective
- Deep transformer layers
- Strong generalization capability

### 🔹 Use Case
High-level acoustic feature learning for emotion recognition.

---

# 🎧 2️⃣ Wav2Vec2 (Base) – Bandpass Audio Fine-Tuning

### 🔹 Pretrained Model
`facebook/wav2vec2-base`

### 🔹 Training Strategy
- Fine-tuned for **6-class emotion classification**
- Feature extractor initially frozen
- Unfrozen after 3 epochs
- Early stopping (patience = 5)
- AdamW optimizer
- Linear warmup scheduler

### 🔹 Dataset Structure
aug_bandpass_audio/

├── Anger/

├── Disgust/

├── Fear/

├── Happy/

├── Neutral/

└── Sad/


### 🔹 Configuration

- Sampling Rate: 16kHz
- Batch Size: 8
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Epochs: 50 (Early Stopping enabled)
- Loss: CrossEntropyLoss

### 🔹 Outputs Saved

- `best_model.pth`
- `training_curves.png`
- `confusion_matrix.png`
- `training_history.csv`
- `finetuned_wav2vec2_model/` (Full pretrained model directory)

---

# 🔁 3️⃣ Bi-LSTM with Augmented Audio

### 🔹 Model Architecture

- Input: 120-D features  
  (40 MFCC + Delta + Delta-Delta)
- Bi-LSTM Layer 1:
  - Hidden Size: 256
  - Num Layers: 2
  - Bidirectional
- Bi-LSTM Layer 2:
  - Hidden Size: 256
  - Bidirectional
- Fully Connected Layers:
  - Dense (128)
  - Output (6 classes)

### 🔹 Dataset

- Total Samples: **8100**
- Emotions:
  - Anger
  - Disgust
  - Fear
  - Happy
  - Sad
  - Neutral

### 🔹 Data Split

- Train: 6480
- Validation: 810
- Test: 810

### 🔹 Feature Extraction

- MFCC (40 coefficients)
- Delta
- Delta-Delta
- Total Feature Dimension: 120
- Sampling Rate: 16kHz
- Max Duration: 10 seconds

### 🔹 Training Configuration

- Epochs: 50
- Batch Size: 32
- Learning Rate: 0.001
- Dropout: 0.3
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Early Stopping: Yes



---

# 📊 Model Comparison

| Model        | Type                  | Feature Learning Style | Strength |
|-------------|----------------------|------------------------|----------|
| HuBERT Large | Transformer (SSL)     | Self-supervised        | Deep contextual embeddings |
| Wav2Vec2     | Transformer (SSL)     | Fine-tuned pretrained  | Strong generalization |
| Bi-LSTM      | Recurrent Neural Net  | MFCC-based features    | Lightweight & interpretable |

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchaudio
- HuggingFace Transformers
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# 🚀 How to Run

## 1️⃣ Install Dependencies

```bash
pip install torch torchaudio transformers librosa datasets soundfile scikit-learn pandas matplotlib seaborn tqdm
```

## 2️⃣ Update Dataset Path

Update DATA_PATH in notebooks to your local dataset folder:

```bash
aug_bandpass_audio/
```

## 3️⃣ Run Notebooks

Execute:

- Hubert Large Model.ipynb
- wav2vec_bandpass.ipynb
- bilstm_augmentd.ipynb


---

## 📌 End-to-End Pipeline Summary

This project implements a complete end-to-end Speech Emotion Recognition pipeline:

1. **Dataset Preparation**
   - Organized into emotion-wise folders
   - Includes augmented audio (bandpass, pitch shift, time masking)
   - Stratified train-validation-test split

2. **Feature Extraction**
   - HuBERT / Wav2Vec2 raw waveform embeddings
   - MFCC + Delta + Delta-Delta features (Bi-LSTM)
   - Sampling rate normalization (16kHz)

3. **Model Training**
   - Transformer-based fine-tuning (HuBERT, Wav2Vec2)
   - Recurrent architecture training (Bi-LSTM)
   - Early stopping and learning rate scheduling
   - Gradient clipping for stability

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Per-emotion performance analysis

5. **Inference**
   - Single-audio prediction support
   - Confidence score output
   - Full model checkpoint saving for deployment

---
## 📊 Key Takeaways

- Self-supervised transformer models (HuBERT, Wav2Vec2) provide strong contextual representations.
- Fine-tuning pretrained models significantly improves performance over training from scratch.
- MFCC-based Bi-LSTM offers a computationally lighter alternative.
- Data augmentation improves generalization and robustness.

---
