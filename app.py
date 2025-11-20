import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd
import os

# --- Configuration (UPDATED FOR STREAMLIT CLOUD) ---
# The model and scaler files must be in the same GitHub directory as app.py
MODEL_PATH = 'best_rf_model.pkl'
SCALER_PATH = 'scaler.pkl'
NUM_MFCC = 28
AD_THRESHOLD = 0.40 # Threshold for high AD detection sensitivity (PAD >= 0.40)

# --- Global Assets (Model and Scaler) ---
@st.cache_resource
def load_assets():
    """Loads the model and scaler using relative paths."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Asset not found: {MODEL_PATH} or {SCALER_PATH}. Ensure they are in your GitHub repo.")
        return None, None

model, scaler = load_assets()

# --- Feature Extraction Function ---
def extract_features_streamlit(file_path):
    """Extracts acoustic features (28 MFCCs, Chroma, Spectral, ZCR)."""
    try:
        # Load audio data. You should use a lower default sample rate like 22050 
        # for deployment if sr=None is causing issues, but sr=None works with pydub/ffmpeg.
        y, sr = librosa.load(file_path, duration=5, offset=0.5, sr=None) 
        
        # Calculate features (must match the features used for training)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        features = np.hstack([mfccs, chroma, spec_centroid, spec_rolloff, zcr])
        return pd.DataFrame([features])

    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# --- Main Streamlit App ---
st.set_page_config(page_title="🧠 AD Speech Detector", layout="centered")
st.title("🗣️ Early AD Detection from Speech")
st.markdown(f"**Screening Threshold:** AD risk flagged if probability $\\geq {AD_THRESHOLD}$.")

# Use a temporary directory for file operations that Streamlit Cloud can write to
temp_dir = "/tmp"

uploaded_file = st.file_uploader("Upload Audio File (.wav, .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None and model is not None and scaler is not None:
    temp_audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format='audio/wav')

    if st.button('Analyze Speech'):
        with st.spinner('Extracting features and predicting...'):
            feature_vector = extract_features_streamlit(temp_audio_path)

            if feature_vector is not None:
                # Scale the input features using the loaded scaler
                scaled_features = scaler.transform(feature_vector)

                # Get probability for the AD class (Class 1)
                proba = model.predict_proba(scaled_features)[0]
                ad_probability = proba[1]

                st.divider()

                # Classification based on the sensitive 0.40 threshold
                if ad_probability >= AD_THRESHOLD:
                    st.error("### ⚠️ HIGH RISK: AD-Like Speech Patterns Detected")
                    st.metric(label="Probability (AD)", value=f"{ad_probability*100:.2f}%")
                    st.info("High risk classification based on sensitivity setting.")
                else:
                    st.success("### ✅ LOW RISK: Cognitively Normal Speech")
                    st.metric(label="Probability (CN)", value=f"{proba[0]*100:.2f}%")

            # Clean up the temporary file (optional but good practice)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
