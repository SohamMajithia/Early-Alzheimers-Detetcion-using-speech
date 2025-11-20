# Early-Alzheimers-Detetcion-using-speech
# 🧠 Early Alzheimer's Detection from Speech

A machine learning system that screens for **early Alzheimer's risk using short speech recordings.**  
This project uses **synthetically generated data**, **acoustic biomarkers**, and a **browser-based Streamlit interface** to perform inference.

---

## 🎯 Objective

To detect speech patterns that correlate with **early cognitive decline**, specifically:
- increased pauses
- hesitation markers
- slowed speech rate
- reduced lexical diversity
- abnormal spectral + prosodic characteristics

This system provides **a screening aid**, not a clinical diagnostic tool.

---

## 🏗️ Project Pipeline

### 1️⃣ Synthetic Speech Dataset (Notebook)

The notebook:
- creates speech templates with **fillers, hesitation markers, pauses**
- assigns each sample a **cognitive label (Normal vs AD-like)**
- converts text transcripts into speech using **Text-to-Speech (TTS)**
- extracts acoustic features for ML classification

📌 **Dataset is not uploaded** due to size, but **it can be recreated fully from the notebook.**
    A public dataset was not availbale that could help me detect early alzheimers from speech , so I made my own .

---

### 2️⃣ Feature Engineering
A total of 43 acoustic features are extracted from evry single  audio  file.
Extracted features include:

| Feature Group | Examples |
|--------------|----------|
| **MFCC (28)** | timbral + spectral features |
| **Chroma** | pitch class distribution |
| **Spectral Features** | centroid, rolloff |
| **Prosodic Markers** | silence, hesitation (through TTS patterns) |
| **ZCR** | voice activity measure |

Feature extraction uses `librosa`.

---

### 3️⃣ Model

A `RandomForestClassifier` is trained on the synthetic feature set.  
The final outputs saved are:

- `best_rf_model.pkl`
- `scaler.pkl`

These files are required by the Streamlit app.

---

## 💻 Streamlit Web App (`app.py`)

The UI allows users to **upload a speech sample** (`wav` or `mp3`) and receive a predicted cognitive risk score.

#### 🧐 Classification Threshold
The app uses a **sensitive screening threshold**:

