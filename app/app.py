import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import io
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim

# Add root folder to python path to ensure imports work when running streamlit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import load_audio, save_audio, normalize_audio
from utils.metrics import evaluate_all
from realtime.processor import process_audio
from ml_model.model import DenoiseNet, WEIGHTS_PATH

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SoundShield AI | Noise Cancellation System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS FOR RICH & SIMPLE AESTHETICS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App Background */
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Custom Cards */
.metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #334155;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transition: all 0.2s ease-in-out;
    text-align: center;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.1);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 4px 0;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* Main Header Styling */
.main-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 1rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid #334155;
}
.main-header .logo {
    font-size: 2.5rem;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header .subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 4px 0 0 0;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid #1e293b !important;
}

/* Alert styling customization */
.stAlert {
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER SECTION ----------------
st.markdown("""
<div class="main-header">
    <span class="logo">🎧</span>
    <div class="title-container">
        <h1>SoundShield AI</h1>
        <p class="subtitle">Enterprise Real-Time Noise Cancellation & Speech Enhancement Dashboard</p>
    </div>
</div>
""", unsafe_allow_html=True)



# ---------------- NOISE GENERATOR HELPERS ----------------
def generate_pink_noise(length):
    """
    Generate pink noise (1/f spectral density)
    """
    uneven = length % 2
    X = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1)
    y = (np.fft.irfft(X / S))[:length]
    return y / (np.max(np.abs(y)) + 1e-8)

def generate_hum(length, sr, hum_freq=50):
    """
    Generate electrical power line hum (hum frequency + harmonics)
    """
    t = np.arange(length) / sr
    hum = (
        np.sin(2 * np.pi * hum_freq * t) +
        0.5 * np.sin(2 * np.pi * 2 * hum_freq * t) +
        0.25 * np.sin(2 * np.pi * 3 * hum_freq * t)
    )
    return hum / (np.max(np.abs(hum)) + 1e-8)

def mix_noise_at_snr(clean_signal, noise_signal, target_snr_db):
    """
    Mix clean signal and noise signal to match the target SNR in decibels
    """
    s_power = np.mean(clean_signal ** 2)
    n_power = np.mean(noise_signal ** 2)
    
    if s_power < 1e-8:
        return clean_signal + noise_signal, noise_signal
        
    target_n_power = s_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_n_power / (n_power + 1e-8))
    
    scaled_noise = noise_signal * scale
    mixed = clean_signal + scaled_noise
    return mixed, scaled_noise


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.markdown("### ⚙️ Denoising Engine Settings")

filter_mode = st.sidebar.selectbox(
    "Select Denoising Filter",
    ["LMS", "NLMS", "Wiener", "DL (Deep Learning)"],
    help="LMS/NLMS are adaptive filters. Wiener is frequency-domain. DL uses a Deep Autoencoder model."
)

st.sidebar.markdown("---")

# Filter Specific Hyperparameters
st.sidebar.markdown("### 🔧 Filter Hyperparameters")
config_params = {}

if filter_mode == "LMS":
    config_params["filter_order"] = st.sidebar.slider("Filter Order (M)", min_value=8, max_value=256, value=64, step=8, help="Number of filter coefficients")
    config_params["mu"] = st.sidebar.slider("Step Size (μ)", min_value=0.001, max_value=0.100, value=0.010, step=0.001, format="%.3f", help="Controls adaptation speed and stability")

elif filter_mode == "NLMS":
    config_params["filter_order"] = st.sidebar.slider("Filter Order (M)", min_value=8, max_value=256, value=64, step=8)
    config_params["mu"] = st.sidebar.slider("Step Size (μ)", min_value=0.01, max_value=1.00, value=0.08, step=0.01, format="%.2f", help="Normalized learning step size")
    config_params["epsilon"] = st.sidebar.number_input("Regularization (ε)", min_value=1e-8, max_value=1e-3, value=1e-6, format="%.1e", help="Stabilizer against division by zero")

elif filter_mode == "Wiener":
    config_params["alpha"] = st.sidebar.slider("Noise Scaling (α)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Over-subtraction factor for spectral scaling")

elif filter_mode == "DL (Deep Learning)":
    # Check DL status
    dl_model_loaded = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(WEIGHTS_PATH):
        dl_model_loaded = True
        st.sidebar.success(f"🤖 Model Loaded ({device.upper()})")
    else:
        st.sidebar.warning("⚠️ No Weights Found! Go to 'Deep Learning Center' tab to train the model first.")

st.sidebar.markdown("---")

# Noise Simulation Settings
st.sidebar.markdown("### 🔊 Noise Simulation Control")
simulate_noise = st.sidebar.checkbox("Inject Simulated Noise", value=True, help="Mix simulated noise into your clean audio signal.")

if simulate_noise:
    noise_type = st.sidebar.selectbox(
        "Noise Type",
        ["White Gaussian Noise", "Pink Noise (1/f)", "AC Hum (50 Hz Electrical Hum)"]
    )
    snr_level = st.sidebar.slider(
        "Target SNR (dB)",
        min_value=-10,
        max_value=30,
        value=10,
        step=2,
        help="Signal-to-Noise ratio. Lower means more noise."
    )
else:
    st.sidebar.info("Processing raw audio signal as uploaded.")


# ---------------- MAIN UI WORKSPACE TABS ----------------
tab1, tab2, tab3 = st.tabs(["🚀 Audio Denoising Playground", "🎤 Record Live Mic", "🧠 Deep Learning Center"])

# ==========================================
# TAB 1: AUDIO DENOISING PLAYGROUND
# ==========================================
with tab1:
    st.markdown("### 🎛️ Upload or Select Audio")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])
        
    with col2:
        # Provide sample preset selection
        preset_files = ["None", "Sample 1 (Clean)", "Sample 2 (Clean)"]
        selected_preset = st.selectbox("Or choose a pre-loaded sample", preset_files)
        
    # Determine the audio source file path
    audio_path = None
    if uploaded_file is not None:
        # Save upload to a temp location
        audio_path = "temp_uploaded.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
    elif selected_preset == "Sample 1 (Clean)":
        audio_path = "data/clean/sample1.wav"
    elif selected_preset == "Sample 2 (Clean)":
        audio_path = "data/clean/sample2.wav"

    if audio_path is not None:
        st.markdown("---")
        
        # Load audio signal
        try:
            signal, sr = load_audio(audio_path)
            
            # Sub-sample if files are too long to prevent dashboard lag (limit to 10 seconds max)
            max_samples = 10 * sr
            if len(signal) > max_samples:
                signal = signal[:max_samples]
                st.warning("⚠️ File is longer than 10 seconds. Truncated to first 10 seconds to optimize visualization performance.")
                
            st.success(f"Successfully loaded file | Sample Rate: **{sr} Hz** | Duration: **{len(signal)/sr:.2f} seconds**")
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            signal = None

        if signal is not None:
            # ---------------- NOISE SIMULATION ----------------
            if simulate_noise:
                if noise_type == "White Gaussian Noise":
                    noise_ref = np.random.randn(len(signal)).astype(np.float32)
                elif noise_type == "Pink Noise (1/f)":
                    noise_ref = generate_pink_noise(len(signal))
                else: # AC Hum
                    noise_ref = generate_hum(len(signal), sr, hum_freq=50)
                
                # Mix
                noisy_signal, final_noise = mix_noise_at_snr(signal, noise_ref, snr_level)
            else:
                noisy_signal = signal.copy()
                final_noise = np.random.randn(len(signal)).astype(np.float32) * 1e-4  # minimal reference

            # ---------------- PROCESSING ENGINE ----------------
            with st.spinner(" Denoising audio signal..."):
                start_time = time.time()
                
                # Convert active filter name to processor format
                proc_mode = "DL" if filter_mode.startswith("DL") else filter_mode
                
                # Map configuration
                processor_config = {
                    "lms": {"filter_order": config_params.get("filter_order", 32), "mu": config_params.get("mu", 0.01)},
                    "nlms": {"filter_order": config_params.get("filter_order", 32), "mu": config_params.get("mu", 0.1), "epsilon": config_params.get("epsilon", 1e-6)},
                    "wiener": {"alpha": config_params.get("alpha", 1.0)}
                }
                
                # Process signal
                filtered_signal = process_audio(
                    noisy_signal,
                    mode=proc_mode,
                    noise_ref=final_noise,
                    config=processor_config
                )
                
                latency = (time.time() - start_time) * 1000

            # ---------------- METRICS COMPUTATION ----------------
            metrics = evaluate_all(signal, filtered_signal)
            noisy_metrics = evaluate_all(signal, noisy_signal)
            
            # Compute SNR improvement
            snr_improvement = metrics["SNR"] - noisy_metrics["SNR"]

            # ---------------- DISPLAY PREMIUM METRIC CARDS ----------------
            st.markdown("### 📊 Denoising Performance Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">SNR Improvement</div>
                    <div class="metric-value">{snr_improvement:+.2f} dB</div>
                    <div class="metric-label">Processed SNR: {metrics['SNR']:.2f} dB</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Squared Error</div>
                    <div class="metric-value">{metrics['MSE']:.6f}</div>
                    <div class="metric-label">Noisy MSE: {noisy_metrics['MSE']:.6f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Peak SNR (PSNR)</div>
                    <div class="metric-value">{metrics['PSNR']:.2f} dB</div>
                    <div class="metric-label">Noisy PSNR: {noisy_metrics['PSNR']:.2f} dB</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Engine Latency</div>
                    <div class="metric-value">{latency:.1f} ms</div>
                    <div class="metric-label">Length: {len(signal)} samples</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ---------------- AUDIO PLAYBACK INDICATORS ----------------
            st.markdown("### 🔊 Listen to the Difference")
            play_col1, play_col2, play_col3 = st.columns(3)
            
            with play_col1:
                st.markdown("##### 🟢 Original Clean Signal")
                st.audio(signal, sample_rate=sr)
                
            with play_col2:
                st.markdown("##### 🔴 Input Noisy Signal")
                st.audio(noisy_signal, sample_rate=sr)
                
            with play_col3:
                st.markdown("##### 🔵 Output Denoised Signal")
                st.audio(filtered_signal, sample_rate=sr)
                
                # Provide download button
                out_buffer = io.BytesIO()
                sf.write(out_buffer, filtered_signal, sr, format="WAV")
                st.download_button(
                    label="📥 Download Denoised Audio (.wav)",
                    data=out_buffer.getvalue(),
                    file_name="denoised_audio.wav",
                    mime="audio/wav"
                )

            st.markdown("---")

            # ---------------- MATPLOTLIB PLOTS ----------------
            st.markdown("### 📈 Visual Comparison: Waveforms & Spectrograms")
            
            with st.spinner("Generating high-fidelity spectrum plots..."):
                fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex='col')
                
                t_arr = np.arange(len(signal)) / sr
                
                # Colors
                c_clean = "#10b981"
                c_noisy = "#ef4444"
                c_filtered = "#3b82f6"
                
                # --- Waveforms (Left Column) ---
                axes[0, 0].plot(t_arr, signal, color=c_clean, alpha=0.85, linewidth=0.8)
                axes[0, 0].set_title("Original Clean Waveform", fontsize=11, color="#8a99ad", weight="bold")
                axes[0, 0].set_ylabel("Amplitude")
                axes[0, 0].grid(True, alpha=0.15)
                
                axes[1, 0].plot(t_arr, noisy_signal, color=c_noisy, alpha=0.85, linewidth=0.8)
                axes[1, 0].set_title("Noisy Waveform (Input)", fontsize=11, color="#8a99ad", weight="bold")
                axes[1, 0].set_ylabel("Amplitude")
                axes[1, 0].grid(True, alpha=0.15)
                
                axes[2, 0].plot(t_arr, filtered_signal, color=c_filtered, alpha=0.85, linewidth=0.8)
                axes[2, 0].set_title(f"Denoised Waveform ({filter_mode})", fontsize=11, color="#8a99ad", weight="bold")
                axes[2, 0].set_ylabel("Amplitude")
                axes[2, 0].set_xlabel("Time (seconds)")
                axes[2, 0].grid(True, alpha=0.15)
                
                # --- Spectrograms (Right Column) ---
                # Generate spectrograms using matplotlib specgram
                NFFT = 512
                noverlap = 256
                
                pxx0, freqs0, bins0, im0 = axes[0, 1].specgram(
                    signal, NFFT=NFFT, Fs=sr, noverlap=noverlap, cmap="viridis"
                )
                axes[0, 1].set_title("Original Clean Spectrogram", fontsize=11, color="#8a99ad", weight="bold")
                axes[0, 1].set_ylabel("Frequency (Hz)")
                
                pxx1, freqs1, bins1, im1 = axes[1, 1].specgram(
                    noisy_signal, NFFT=NFFT, Fs=sr, noverlap=noverlap, cmap="viridis"
                )
                axes[1, 1].set_title("Noisy Spectrogram", fontsize=11, color="#8a99ad", weight="bold")
                axes[1, 1].set_ylabel("Frequency (Hz)")
                
                pxx2, freqs2, bins2, im2 = axes[2, 1].specgram(
                    filtered_signal, NFFT=NFFT, Fs=sr, noverlap=noverlap, cmap="viridis"
                )
                axes[2, 1].set_title(f"Denoised Spectrogram ({filter_mode})", fontsize=11, color="#8a99ad", weight="bold")
                axes[2, 1].set_ylabel("Frequency (Hz)")
                axes[2, 1].set_xlabel("Time (seconds)")
                
                # Styling plot colors & borders
                for r in range(3):
                    for c in range(2):
                        axes[r, c].set_facecolor("#0a0f18")
                        axes[r, c].tick_params(colors="#8a99ad")
                        for spine in axes[r, c].spines.values():
                            spine.set_color("#2d3748")
                
                # Make figures clean and transparent
                fig.patch.set_facecolor("none")
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("💡 Please upload a WAV audio file or choose a pre-loaded sample from the options above.")


# ==========================================
# TAB 2: RECORD LIVE MICROPHONE
# ==========================================
with tab2:
    st.markdown("### 🎤 Record Audio Live from your Local Mic")
    st.markdown("Record a clip directly from your microphone to test how the noise cancellation behaves on real-world inputs.")
    
    rec_duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=10, value=4)
    rec_sr = 16000 # target sample rate
    
    rec_col1, rec_col2 = st.columns([1, 2])
    
    with rec_col1:
        start_rec_btn = st.button("🎙️ Start Recording", use_container_width=True)
        
    with rec_col2:
        rec_status = st.empty()

    if start_rec_btn:
        rec_status.info("🎙️ Recording in progress... Please speak into your mic.")
        progress_bar = st.progress(0)
        
        # We record chunk by chunk to update progress bar visually
        recorded_chunks = []
        chunk_len = int(0.2 * rec_sr) # 200 ms chunks
        num_steps = int(rec_duration / 0.2)
        
        try:
            # Simple record call using sounddevice
            # To show real-time progress we can record in a single sounddevice block and wait, or simulate progress bar
            # Wait, running sd.rec in a separate thread is cleaner, but simple sd.rec with sd.wait is most reliable:
            recording = sd.rec(
                int(rec_duration * rec_sr),
                samplerate=rec_sr,
                channels=1,
                dtype="float32"
            )
            
            # Progress bar simulation while recording
            for step in range(num_steps):
                time.sleep(0.2)
                progress_bar.progress((step + 1) / num_steps)
                
            sd.wait() # Make sure it is finished
            
            raw_recording = recording.flatten()
            
            # Normalize
            raw_recording = normalize_audio(raw_recording)
            
            st.session_state["mic_recording"] = raw_recording
            st.session_state["mic_sr"] = rec_sr
            rec_status.success("✅ Recording complete!")
            
        except Exception as e:
            rec_status.error(f"❌ Recording failed: {e}. Check if you have a default recording device configured.")

    # Show recording options if a recording exists in session state
    if "mic_recording" in st.session_state:
        mic_audio = st.session_state["mic_recording"]
        mic_sr = st.session_state["mic_sr"]
        
        st.markdown("---")
        st.markdown("### ⚙️ Live Denoising Controls")
        
        # Option to inject simulated noise to mic
        mic_noise_inject = st.checkbox("Inject Simulated Noise into Mic Recording", value=False)
        
        if mic_noise_inject:
            mic_noise_type = st.selectbox("Mic Noise Type", ["White Gaussian Noise", "Pink Noise (1/f)", "AC Hum (50 Hz)"], key="mic_noise")
            mic_snr = st.slider("Mic Target SNR (dB)", min_value=-5, max_value=25, value=10, key="mic_snr_slider")
            
            if mic_noise_type == "White Gaussian Noise":
                n_ref = np.random.randn(len(mic_audio)).astype(np.float32)
            elif mic_noise_type == "Pink Noise (1/f)":
                n_ref = generate_pink_noise(len(mic_audio))
            else:
                n_ref = generate_hum(len(mic_audio), mic_sr, hum_freq=50)
                
            mic_input, mic_noise_final = mix_noise_at_snr(mic_audio, n_ref, mic_snr)
        else:
            mic_input = mic_audio.copy()
            mic_noise_final = np.random.randn(len(mic_audio)).astype(np.float32) * 1e-4

        # Denoise button for mic recording
        if st.button("✨ Apply Active Filter on Recording", use_container_width=True):
            with st.spinner("Processing mic recording..."):
                proc_mode = "DL" if filter_mode.startswith("DL") else filter_mode
                
                processor_config = {
                    "lms": {"filter_order": config_params.get("filter_order", 32), "mu": config_params.get("mu", 0.01)},
                    "nlms": {"filter_order": config_params.get("filter_order", 32), "mu": config_params.get("mu", 0.1), "epsilon": config_params.get("epsilon", 1e-6)},
                    "wiener": {"alpha": config_params.get("alpha", 1.0)}
                }
                
                mic_filtered = process_audio(
                    mic_input,
                    mode=proc_mode,
                    noise_ref=mic_noise_final,
                    config=processor_config
                )
                
                st.session_state["mic_filtered"] = mic_filtered
                st.session_state["mic_input_mixed"] = mic_input

        # Playback mic results
        if "mic_input_mixed" in st.session_state and "mic_filtered" in st.session_state:
            st.markdown("---")
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.markdown("##### 🎙️ Mic Input (Noisy/Recorded)")
                st.audio(st.session_state["mic_input_mixed"], sample_rate=mic_sr)
                
            with m_col2:
                st.markdown("##### 🔊 Mic Output (Denoised)")
                st.audio(st.session_state["mic_filtered"], sample_rate=mic_sr)
                
                out_buffer = io.BytesIO()
                sf.write(out_buffer, st.session_state["mic_filtered"], mic_sr, format="WAV")
                st.download_button(
                    label="📥 Download Denoised Mic Audio (.wav)",
                    data=out_buffer.getvalue(),
                    file_name="denoised_mic_audio.wav",
                    mime="audio/wav",
                    key="mic_download"
                )


# ==========================================
# TAB 3: DEEP LEARNING CENTER
# ==========================================
with tab3:
    st.markdown("### 🧠 Deep Learning Autoencoder Control Console")
    st.markdown("Our Deep Learning mode utilizes a neural network autoencoder (`DenoiseNet`) trained on clean-noisy speech pairs to reconstruct clean speech signals.")
    
    # Model details
    st.markdown("#### 📐 Model Architecture Specs")
    st.code("""
DenoiseNet(
  (encoder): Sequential(
    (0): Linear(in_features=1024, out_features=512)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256)
  )
  (decoder): Sequential(
    (0): Linear(in_features=256, out_features=512)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=1024)
  )
)
    """, language="python")

    st.markdown("---")
    st.markdown("#### 🏋️ Live Model Training Dashboard")
    st.markdown("You can retrain the neural network using the clean samples inside the dataset directory (`data/clean`).")

    # Hyperparameter Inputs
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        train_epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=15, step=5)
    with h_col2:
        train_lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    with h_col3:
        train_batch = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)

    train_btn = st.button("🚀 Begin Training DenoiseNet", use_container_width=True)
    
    status_msg = st.empty()
    progress_bar = st.empty()
    chart_msg = st.empty()

    if train_btn:
        status_msg.info("🚀 Preparing dataset for training...")
        
        # Load Dataset
        import soundfile as sf
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "clean")
        
        if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
            status_msg.error(f"❌ Dataset path '{data_path}' not found or empty! Put wav files inside it.")
        else:
            chunks_clean = []
            chunks_noisy = []
            
            try:
                for file in os.listdir(data_path):
                    if not file.endswith(".wav"):
                        continue
                        
                    sig, _ = sf.read(os.path.join(data_path, file))
                    if sig.ndim > 1:
                        sig = np.mean(sig, axis=1)
                        
                    sig = sig / (np.max(np.abs(sig)) + 1e-8)
                    
                    # Create noisy version (synthetic noise for training pairs)
                    noise = np.random.randn(len(sig)) * 0.02
                    noisy = sig + noise
                    
                    # Create 1024-size chunks
                    chunk_size = 1024
                    for i in range(0, len(sig), chunk_size):
                        clean_c = sig[i:i+chunk_size]
                        noisy_c = noisy[i:i+chunk_size]
                        
                        if len(clean_c) < chunk_size:
                            pad = chunk_size - len(clean_c)
                            clean_c = np.pad(clean_c, (0, pad))
                            noisy_c = np.pad(noisy_c, (0, pad))
                            
                        chunks_clean.append(clean_c)
                        chunks_noisy.append(noisy_c)
                
                X = np.array(chunks_noisy, dtype=np.float32)
                Y = np.array(chunks_clean, dtype=np.float32)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                X_tensor = torch.tensor(X).to(device)
                Y_tensor = torch.tensor(Y).to(device)
                
                dataset_size = X.shape[0]
                
                # Initialize model
                model = DenoiseNet().to(device)
                optimizer = optim.Adam(model.parameters(), lr=train_lr)
                criterion = nn.MSELoss()
                
                status_msg.success(f"Dataset Loaded: {dataset_size} training samples. Starting training on {device.upper()}...")
                
                losses = []
                progress = progress_bar.progress(0)
                
                # Create a chart placeholder
                chart_placeholder = chart_msg.empty()
                
                # Training Loop
                for epoch in range(train_epochs):
                    model.train()
                    epoch_loss = 0.0
                    
                    indices = np.random.permutation(dataset_size)
                    
                    for i in range(0, dataset_size, train_batch):
                        batch_idx = indices[i:i+train_batch]
                        bx = X_tensor[batch_idx]
                        by = Y_tensor[batch_idx]
                        
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                    avg_loss = epoch_loss / (dataset_size / train_batch)
                    losses.append(avg_loss)
                    
                    # Update status
                    status_msg.info(f"🔄 Epoch {epoch+1}/{train_epochs} | Training Loss: **{avg_loss:.6f}**")
                    progress.progress((epoch + 1) / train_epochs)
                    
                    # Update training chart live!
                    chart_placeholder.line_chart(losses)
                
                # Save trained weights
                os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": losses[-1],
                    "epoch": train_epochs - 1
                }, WEIGHTS_PATH)
                
                status_msg.success("🎉 Training Completed and Model Saved Successfully! Deep Learning mode is now fully enabled.")
                
            except Exception as e:
                status_msg.error(f"❌ Training failed: {e}")
