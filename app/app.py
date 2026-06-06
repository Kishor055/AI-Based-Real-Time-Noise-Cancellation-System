import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.audio import load_audio
from utils.metrics import evaluate_all
from realtime.processor import process_audio

st.set_page_config(page_title="AI Noise Cancellation", layout="wide")

st.title("🎧 AI Noise Cancellation System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Filter Type", ["LMS", "NLMS", "Wiener", "DL"])

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    st.success("File uploaded successfully")

    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    signal, sr = load_audio("temp.wav")

    st.write(f"Sample Rate: {sr}")
    st.write(f"Length: {len(signal)} samples")

    # Add synthetic noise
    noise = np.random.randn(len(signal)) * 0.01
    noisy_signal = signal + noise

    # Process
    if mode in ["LMS", "NLMS"]:
        filtered = process_audio(noisy_signal, mode=mode, noise_ref=noise)
    else:
        filtered = process_audio(noisy_signal, mode=mode)

    # ---------------- METRICS ----------------
    metrics = evaluate_all(signal, filtered)

    st.subheader("📊 Metrics")
    st.json(metrics)

    # ---------------- PLOT ----------------
    st.subheader("📈 Waveforms")

    fig, ax = plt.subplots(3, 1, figsize=(10, 6))

    ax[0].plot(signal)
    ax[0].set_title("Original")

    ax[1].plot(noisy_signal)
    ax[1].set_title("Noisy")

    ax[2].plot(filtered)
    ax[2].set_title("Filtered")

    plt.tight_layout()
    st.pyplot(fig)
