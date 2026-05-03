import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly
import os

EPS = 1e-8


# ---------------- LOAD AUDIO ----------------
def load_audio(path, target_sr=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    signal, sr = sf.read(path, dtype="float32")

    # Convert to mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Normalize
    signal = normalize_audio(signal)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        signal = resample_audio(signal, sr, target_sr)
        sr = target_sr

    return signal, sr


# ---------------- SAVE AUDIO ----------------
def save_audio(path, signal, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    signal = np.clip(signal, -1.0, 1.0).astype(np.float32)
    sf.write(path, signal, sr)


# ---------------- NORMALIZATION ----------------
def normalize_audio(signal):
    max_val = np.max(np.abs(signal)) + EPS
    return signal / max_val


# ---------------- RESAMPLING ----------------
def resample_audio(signal, original_sr, target_sr):
    """
    Better resampling using polyphase filtering
    """
    gcd = np.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    return resample_poly(signal, up, down)


# ---------------- CHUNKING ----------------
def split_chunks(signal, chunk_size=1024):
    num_chunks = int(np.ceil(len(signal) / chunk_size))

    padded = np.pad(signal, (0, num_chunks * chunk_size - len(signal)))

    return padded.reshape(num_chunks, chunk_size)


def merge_chunks(chunks, original_length):
    signal = chunks.reshape(-1)
    return signal[:original_length]


# ---------------- PLAYBACK ----------------
def play_audio(signal, sr):
    sd.play(signal.astype(np.float32), sr)
    sd.wait()


# ---------------- RECORD AUDIO ----------------
def record_audio(duration=5, sr=16000):
    try:
        print("🎤 Recording...")

        recording = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        print("✅ Recording complete")

        return recording.flatten()

    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return np.zeros(int(duration * sr))


# ---------------- REAL-TIME STREAM ----------------
def stream_audio(callback, sr=16000, block_size=1024):
    """
    Correct real-time streaming (low latency)
    """

    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(f"⚠️ Stream status: {status}")

        chunk = indata[:, 0]

        try:
            processed = callback(chunk)

            # Ensure correct shape
            if processed.ndim == 1:
                processed = processed.reshape(-1, 1)

            outdata[:] = processed

        except Exception as e:
            print(f"❌ Processing error: {e}")
            outdata[:] = indata  # fallback (pass-through)

    stream = sd.Stream(
        callback=audio_callback,
        samplerate=sr,
        blocksize=block_size,
        channels=1,
        dtype="float32"
    )

    stream.start()
    return stream
