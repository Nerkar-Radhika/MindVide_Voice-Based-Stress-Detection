import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="MindVibe AI", page_icon="🎙️", layout="centered")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        selector = joblib.load("selector.pkl")
        return model, scaler, selector, True
    except Exception as e:
        return None, None, None, False

model, scaler, selector, loaded = load_assets()

def extract_features(file_path, gender):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=20)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean, mfcc_std = np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0)
        delta = librosa.feature.delta(mfcc)
        delta_mean, delta_std = np.mean(delta.T, axis=0), np.std(delta.T, axis=0)
        delta2 = librosa.feature.delta(mfcc, order=2)
        delta2_mean, delta2_std = np.mean(delta2.T, axis=0), np.std(delta2.T, axis=0)
        zcr_feat = [np.mean(librosa.feature.zero_crossing_rate(y)), np.std(librosa.feature.zero_crossing_rate(y))]
        rms_feat = [np.mean(librosa.feature.rms(y=y)), np.std(librosa.feature.rms(y=y))]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean, chroma_std = np.mean(chroma.T, axis=0), np.std(chroma.T, axis=0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_mean, mel_std = np.mean(mel.T, axis=0), np.std(mel.T, axis=0)
        contrast_m = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean, tonnetz_std = np.mean(tonnetz.T, axis=0), np.std(tonnetz.T, axis=0)
        rolloff_feat = [np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)), np.std(librosa.feature.spectral_rolloff(y=y, sr=sr))]
        bw_feat = [np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)), np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))]
        g_val = 1 if gender == "Female" else 0

        features = np.concatenate([
            mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std,
            zcr_feat, rms_feat, chroma_mean, chroma_std, mel_mean, mel_std,
            contrast_m, tonnetz_mean, tonnetz_std, rolloff_feat, bw_feat, [g_val]
        ])
        scaled = scaler.transform([features])
        selected = selector.transform(scaled)
        return selected, features
    except Exception:
        return None, None

def predict(file_path, gender):
    result, _ = extract_features(file_path, gender)
    if result is None:
        return None, None, None, None

    probability = model.predict_proba(result)[0]
    stressed_p = probability[1]
    unstressed_p = probability[0]

    OPTIMAL_THRESHOLD = 0.25
    prediction = 1 if stressed_p >= OPTIMAL_THRESHOLD else 0
    confidence = max(probability) * 100
    return prediction, confidence, stressed_p * 100, unstressed_p * 100

def show_results(prediction, confidence, stressed_prob, unstressed_prob):
    st.divider()
    st.subheader("📊 Analysis Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", "🔴 STRESSED" if prediction == 1 else "🟢 CALM")
    col2.metric("Stressed %", f"{stressed_prob:.1f}%")
    col3.metric("Unstressed %", f"{unstressed_prob:.1f}%")

    st.progress(stressed_prob / 100)

    if prediction == 1:
        st.error("⚠️ Stress detected!")
        st.markdown("""
        **Stress Relief Tips:**
        - 🫁 Try 4-7-8 breathing — inhale 4 sec, hold 7 sec, exhale 8 sec
        - 💧 Drink a glass of water
        - 🚶 Take a short walk
        - 🎵 Listen to calming music
        - 🧘 Try 5 minute meditation
        """)
    else:
        st.success("✅ You sound calm and relaxed!")
        st.markdown("""
        **Keep it up!**
        - 😊 Great job managing your stress!
        - 🌿 Stay consistent with healthy habits
        - ☀️ Keep a positive mindset
        """)

# --- UI ---
st.title("🎙️ MindVibe")
st.caption("Voice Based Stress Detection | Hindi + English + Marathi | Accuracy: 97.42%")

if not loaded:
    st.error("❌ Model files not found! Please check your directory.")
    st.stop()

st.divider()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("👤 Select Gender", ["Male", "Female"])
with col2:
    language = st.selectbox("🌐 Select Language", ["Hindi", "English", "Marathi"])

st.divider()

st.write("### 🎯 Analyze Your Stress")
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Voice"])

with tab1:
    st.write("Upload a voice recording in any language")
    audio_file = st.file_uploader("Choose audio file", type=["wav", "mp3", "m4a", "ogg"])
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        if st.button("🔍 Analyze Stress", type="primary", key="upload_btn"):
            with st.spinner("Analyzing your voice..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    p, c, sp, up = predict(tmp.name, gender)
                    if p is not None:
                        show_results(p, c, sp, up)
                    else:
                        st.error("❌ Could not process audio. Please try again.")
                os.unlink(tmp.name)

with tab2:
    st.write("Record your voice directly")
    audio_bytes = audio_recorder(
        text="Click mic to record",
        recording_color="#E24B4A",
        neutral_color="#185FA5"
    )
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("🔍 Analyze Stress", type="primary", key="record_btn"):
            with st.spinner("Analyzing your voice..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    p, c, sp, up = predict(tmp.name, gender)
                    if p is not None:
                        show_results(p, c, sp, up)
                    else:
                        st.error("❌ Could not process audio. Please try again.")
                os.unlink(tmp.name)

st.divider()
st.caption("⚠️ This tool is for informational purposes only and is not a medical diagnosis.")
